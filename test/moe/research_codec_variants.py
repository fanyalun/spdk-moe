#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Offline, bitwise verified codec comparisons; never runs an inference target."""

import argparse
import ctypes as c
import ctypes.util
import hashlib
import json
from pathlib import Path
import random
import struct
import time

from research_reversible import Block

MIB = 1048576
VARIANTS = ('baseline', 'lattice_zstd1', 'lattice_zstd9', 'fp32_byte_zstd9',
            'integer_byte_zstd9', 'integer_bit_zstd9', 'integer_bit_lz4',
            'split', 'split_fast', 'split_zstd9', 'sub1024', 'sub4096')


class Experiment:
    def __init__(self, library):
        self.lib = c.CDLL(str(library.resolve()))
        p, u, s, i = c.c_void_p, c.c_uint, c.c_size_t, c.c_int
        signatures = {
            'moe_codec_encode': ([p, u, p, s, p], i),
            'moe_codec_decode': ([p, s, p, p], i),
            'research_pack': ([p, p, u, u, u, u], None),
            'research_integers': ([p, p, u, p], None),
            'research_restore_integers': ([p, p, u, p], None),
            'research_shuffle': ([p, p, u, i, i], None),
            'research_split_encode': ([p, u, p, s, p, p, p], c.c_int64),
            'research_split_decode': ([p, s, p, u, p, p, p], i),
            'research_split_decode_fast': ([p, s, p, u, p, p, p], i),
            'research_sub_encode': ([p, u, u, p, s], c.c_int64),
            'research_sub_decode': ([p, s, p, u, u], i),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, result
        for name in ('packed', 'output', 'integers', 'positive', 'negative'):
            setattr(self, name, c.create_string_buffer(MIB + 32))
        for name in ('lattice', 'transform', 'frame', 'stage'):
            setattr(self, name, c.create_string_buffer(2 * MIB + 64))
        self.compressed = c.create_string_buffer(4 * MIB)
        self.zstd = c.CDLL(ctypes.util.find_library('zstd'))
        for name, args in [('ZSTD_compress', [p, s, p, s, i]),
                           ('ZSTD_decompress', [p, s, p, s]), ('ZSTD_isError', [s])]:
            f = getattr(self.zstd, name)
            f.argtypes, f.restype = args, s
        self.zstd.ZSTD_getErrorName.argtypes = [s]
        self.zstd.ZSTD_getErrorName.restype = c.c_char_p
        self.zstd.ZSTD_versionString.restype = c.c_char_p
        self.lz4 = c.CDLL(ctypes.util.find_library('lz4'))
        self.lz4.LZ4_compress_default.argtypes = [p, p, i, i]
        self.lz4.LZ4_decompress_safe.argtypes = [p, p, i, i]
        self.lz4.LZ4_compress_default.restype = i
        self.lz4.LZ4_decompress_safe.restype = i

    def zcheck(self, result):
        if self.zstd.ZSTD_isError(result):
            raise RuntimeError(self.zstd.ZSTD_getErrorName(result).decode())
        return result

    def encode(self, variant, words, b):
        lib = self.lib
        meta = dict(mode=variant, words=words, block=b, groups=None, stage_size=0,
                    index_bytes=24 if variant == 'baseline' else 64)
        source, size = self.lattice, b.bytes
        if variant.startswith('integer_') and b.bits == 32:
            meta['mode'] = 'baseline'
        elif variant.startswith('integer_'):
            lib.research_integers(self.packed, self.integers, words, c.byref(b))
            bit = int('_bit_' in variant)
            lib.research_shuffle(self.integers, self.transform, words, bit, 0)
            source, size = self.transform, 32 * ((words + 7) // 8) if bit else words * 4
        elif variant.startswith('fp32_byte_'):
            lib.research_shuffle(self.packed, self.transform, words, 0, 0)
            source, size = self.transform, words * 4
        elif variant.startswith('split'):
            groups = (Block * 2)()
            size = lib.research_split_encode(self.packed, words, self.frame, len(self.frame),
                                            groups, self.positive, self.negative)
            assert size >= 0, ('split encode', size)
            meta['groups'] = groups
            source = self.frame
        elif variant.startswith('sub'):
            sub = int(variant[3:])
            size = lib.research_sub_encode(self.packed, words, sub, self.frame, len(self.frame))
            assert size >= 0, ('sub encode', size)
            source = self.frame
        meta['stage_size'] = size
        if meta['mode'] != 'baseline' and 'zstd' in variant:
            level = int(variant.split('zstd')[1])
            size = self.zcheck(self.zstd.ZSTD_compress(
                self.compressed, len(self.compressed), source, size, level))
            source = self.compressed
        elif meta['mode'] != 'baseline' and variant.endswith('_lz4'):
            size = self.lz4.LZ4_compress_default(source, self.compressed, size, len(self.compressed))
            assert size > 0, 'LZ4 compression failed'
            source = self.compressed
        return source.raw[:size], meta

    def decode(self, encoded, size, meta):
        lib, mode, words, b = self.lib, meta['mode'], meta['words'], meta['block']
        source = encoded
        if 'zstd' in mode:
            result = self.zcheck(self.zstd.ZSTD_decompress(
                self.stage, len(self.stage) - 16, encoded, size))
            assert result == meta['stage_size']
            c.memset(c.addressof(self.stage) + result, 0, 16)
            source = self.stage
        elif mode.endswith('_lz4'):
            result = self.lz4.LZ4_decompress_safe(encoded, self.stage, size, meta['stage_size'])
            assert result == meta['stage_size']
            source = self.stage
        if mode == 'baseline' or mode.startswith('lattice_'):
            assert lib.moe_codec_decode(source, b.bytes + 16, self.output, c.byref(b)) == 0
        elif mode.startswith('integer_'):
            lib.research_shuffle(source, self.integers, words, int('_bit_' in mode), 1)
            lib.research_restore_integers(self.integers, self.output, words, c.byref(b))
        elif mode.startswith('fp32_byte_'):
            lib.research_shuffle(source, self.output, words, 0, 1)
        elif mode.startswith('split'):
            decode = lib.research_split_decode_fast if mode == 'split_fast' else lib.research_split_decode
            assert decode(source, meta['stage_size'], self.output, words,
                          meta['groups'], self.positive, self.negative) == 0
        else:
            assert lib.research_sub_decode(source, meta['stage_size'], self.output, words,
                                          int(mode[3:])) == 0

    def self_test(self):
        rng = random.Random(71)
        matrix = c.create_string_buffer(struct.pack('<384I', *range(384)))
        packed = [row * 128 + col + j for col in (0, 64)
                  for row in range(3) for j in range(64)]
        for first in (0, 1, 63, 64, 191, 192, 193, 373):
            self.lib.research_pack(matrix, self.packed, 3, 128, first, 11)
            assert self.packed.raw[:44] == struct.pack('<11I', *packed[first:first + 11])
        cases = [struct.pack('<9I', 0, 0x80000000, 0x7f800000, 0xff800000,
                             0x7fc00001, 0x7fa12345, 1, 0x80000001, 0xffffffff),
                 bytes(4 * 13), struct.pack('<13I', *([0x80000000] * 13)),
                 bytes(rng.randrange(256) for _ in range(4 * 257))]
        for sign in (1, -1):
            cases.append(b''.join(struct.pack('<f', sign * rng.randrange(1024) / 1024)
                                  for _ in range(259)))
        cases.append(b''.join(struct.pack('<f', rng.randrange(-1024, 1024) / 1024)
                              for _ in range(259)))
        for data in cases:
            c.memmove(self.packed, data, len(data))
            b = Block()
            assert self.lib.moe_codec_encode(self.packed, len(data) // 4, self.lattice,
                                            len(self.lattice), c.byref(b)) == 0
            for variant in VARIANTS:
                payload, meta = self.encode(variant, len(data) // 4, b)
                encoded = c.create_string_buffer(payload, len(payload) + 16)
                self.decode(encoded, len(payload), meta)
                assert self.output.raw[:len(data)] == data, ('self-test', variant)
        print('PASS: all variants, signed zeros, NaN payloads, infinities, raw fallback, tails', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experts', default='0,128,255')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--d-model', type=int, default=2048)
    parser.add_argument('--d-ff', type=int, default=7168)
    parser.add_argument('--cpu', type=int, default=24)
    args = parser.parse_args()
    import os
    os.sched_setaffinity(0, {args.cpu})
    assert not args.output.exists(), 'Use a fresh output path'
    assert args.d_model % 64 == 0 and args.d_ff % 64 == 0
    variants = args.variants.split(',')
    assert set(variants) <= set(VARIANTS) and 'baseline' in variants
    experts = list(range(256)) if args.experts == 'all' else [int(x) for x in args.experts.split(',')]
    exp = Experiment(args.library)
    exp.self_test()
    totals = {name: dict(original_bytes=0, payload_bytes=0, blocks=0, encode_ns=0,
                        restore_ns=0, index_bytes=0, aligned512=0, aligned4096=0,
                        exact_blocks=0) for name in variants}
    report = dict(state='running', experts=experts, variants=totals, widths={},
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  zstd_version=exp.zstd.ZSTD_versionString().decode(),
                  notes='Offline native helpers, warm reusable buffers; not inference P99. '
                        'Decode timing includes ctypes dispatch and native restoration, excludes equality comparison. '
                        'Index budgets: baseline24/candidate64 per logical block, rounded once at the end.')
    source_digest = hashlib.sha256()
    matrix = c.create_string_buffer(args.d_model * args.d_ff * 4)
    started = time.monotonic()
    for expert in experts:
        for stage in ('gate', 'up', 'down'):
            rows, cols = (args.d_ff, args.d_model) if stage == 'down' else (args.d_model, args.d_ff)
            path = args.weights / f'W_{stage}_{expert}_{args.d_model}x{args.d_ff}.bin'
            assert path.stat().st_size == len(matrix), path
            with path.open('rb') as stream:
                assert stream.readinto(matrix) == len(matrix), path
            for first in range(0, rows * cols, MIB // 4):
                words = min(MIB // 4, rows * cols - first)
                exp.lib.research_pack(matrix, exp.packed, rows, cols, first, words)
                data = exp.packed.raw[:words * 4]
                source_digest.update(data)
                b = Block()
                t = time.perf_counter_ns()
                assert exp.lib.moe_codec_encode(exp.packed, words, exp.lattice,
                                                len(exp.lattice), c.byref(b)) == 0
                baseline_encode_ns = time.perf_counter_ns() - t
                key = str(b.bits)
                report['widths'][key] = report['widths'].get(key, 0) + 1
                for name in variants:
                    row = totals[name]
                    t = time.perf_counter_ns()
                    payload, meta = exp.encode(name, words, b)
                    encode_ns = time.perf_counter_ns() - t
                    if name == 'baseline' or name.startswith(('lattice_', 'integer_')):
                        encode_ns += baseline_encode_ns
                    encoded = c.create_string_buffer(payload, len(payload) + 16)
                    t = time.perf_counter_ns()
                    exp.decode(encoded, len(payload), meta)
                    row['restore_ns'] += time.perf_counter_ns() - t
                    assert exp.output.raw[:len(data)] == data, (expert, stage, first, name)
                    row['exact_blocks'] += 1
                    row['encode_ns'] += encode_ns
                    row['original_bytes'] += len(data)
                    row['payload_bytes'] += len(payload)
                    row['blocks'] += 1
                    row['index_bytes'] += meta['index_bytes']
                    for alignment in (512, 4096):
                        row['aligned' + str(alignment)] += max(alignment, (
                            len(payload) + alignment - 1) // alignment * alignment)
        report.update(completed_expert=expert, elapsed_seconds=time.monotonic() - started)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('EXPERT', expert, 'seconds', round(report['elapsed_seconds'], 2), flush=True)
    for row in totals.values():
        row['restore_gbps'] = row['original_bytes'] / row['restore_ns']
        row['payload_ratio'] = row['payload_bytes'] / row['original_bytes']
        for alignment in (512, 4096):
            index = (row['index_bytes'] + alignment - 1) // alignment * alignment
            row['disk_ratio' + str(alignment)] = (row['aligned' + str(alignment)] + index) / row['original_bytes']
    report.update(state='complete', source_sha256=source_digest.hexdigest(), bitwise_equal=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
