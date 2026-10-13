#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause

"""Read-only compression research on packed weights; never an inference benchmark."""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import zstandard as zstd


MIB = 1024 * 1024


def transform(data, mode, inverse=False):
    if mode == 'none':
        return data
    if len(data) % 4:
        raise ValueError('FP32 transform requires whole words')
    a = np.frombuffer(data, dtype=np.uint8)
    if mode == 'byte':
        return (a.reshape(4, -1).T if inverse else a.reshape(-1, 4).T).copy().tobytes()
    if mode != 'bit' or len(data) % 32:
        raise ValueError('bit shuffle requires multiples of eight FP32 words')
    bits = np.unpackbits(a, bitorder='little')
    bits = bits.reshape(32, -1).T if inverse else bits.reshape(-1, 32).T
    return np.packbits(bits.reshape(-1), bitorder='little').tobytes()


class Codec:
    def __init__(self, name, dictionary=None):
        self.name = name
        if name == 'zstd':
            self.encoder = zstd.ZstdCompressor(level=1, dict_data=dictionary,
                                              write_checksum=True)
            self.decoder = zstd.ZstdDecompressor(dict_data=dictionary)
        else:
            self.lib = ctypes.CDLL(ctypes.util.find_library('lz4'))
            self.lib.LZ4_compressBound.argtypes = [ctypes.c_int]
            self.lib.LZ4_compress_default.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_int, ctypes.c_int]
            self.lib.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.c_int, ctypes.c_int]

    def compress(self, data):
        if self.name == 'zstd':
            return self.encoder.compress(data)
        bound = self.lib.LZ4_compressBound(len(data))
        out = ctypes.create_string_buffer(bound)
        size = self.lib.LZ4_compress_default(data, out, len(data), bound)
        if size <= 0:
            raise ValueError('LZ4 compression failed')
        return out.raw[:size]

    def decompress(self, data, size):
        if self.name == 'zstd':
            out = self.decoder.decompress(data, max_output_size=size)
        else:
            buffer = ctypes.create_string_buffer(size)
            result = self.lib.LZ4_decompress_safe(data, buffer, len(data), size)
            if result != size:
                raise ValueError('LZ4 length/corruption error')
            out = buffer.raw
        if len(out) != size:
            raise ValueError('decompressed length mismatch')
        return out


class Image:
    def __init__(self, path):
        self.path = path
        with path.open('rb') as stream:
            header = stream.read(88)
        f = struct.unpack('<Q6I7Q', header)
        if f[0:3] != (0x31454f4d4b445053, 1, 1):
            raise ValueError('requires complete version-1 packed image')
        self.experts = f[5]
        self.start, self.stride, self.gate, self.down, self.total = f[9:14]
        if path.stat().st_size < self.total:
            raise ValueError('truncated packed image')
        self.header_sha256 = hashlib.sha256(header).hexdigest()

    def matrices(self, experts):
        for expert in experts:
            if not 0 <= expert < self.experts:
                raise ValueError('expert outside image')
            for stage in range(3):
                yield expert, stage, self.start + expert * self.stride + stage * self.gate, (
                    self.down if stage == 2 else self.gate)


def dictionaries(image, held_out, mode, grouped):
    training = [e for e in range(image.experts) if e not in held_out][:16]
    if len(training) < 2:
        raise ValueError('dictionary training needs at least two held-in experts')
    samples = [[], [], []] if grouped else [[]]
    with image.path.open('rb') as stream:
        for _, stage, offset, size in image.matrices(training):
            for position in (0, size // 2 // 32 * 32, (size - 16384) // 32 * 32):
                stream.seek(offset + position)
                samples[stage if grouped else 0].append(transform(stream.read(16384), mode))
    return [zstd.train_dictionary(65536, group) for group in samples]


def measure(image, experts, codec_name, mode, block_mib, alignment, dictionary_mode='none'):
    dicts = dictionaries(image, experts, mode, dictionary_mode == 'stage') if (
        dictionary_mode != 'none') else []
    codecs = [Codec(codec_name, dictionary) for dictionary in dicts] or [Codec(codec_name)]
    original_bytes = disk_bytes = blocks = raw_blocks = 0
    encode_ns = restore_ns = 0
    digest = hashlib.sha256()
    with image.path.open('rb') as stream:
        for _, stage, offset, length in image.matrices(experts):
            stream.seek(offset)
            codec = codecs[stage] if len(codecs) == 3 else codecs[0]
            for position in range(0, length, block_mib * MIB):
                data = stream.read(min(block_mib * MIB, length - position))
                if len(data) != min(block_mib * MIB, length - position):
                    raise ValueError('short read')
                digest.update(data)
                started = time.perf_counter_ns()
                encoded = codec.compress(transform(data, mode))
                encode_ns += time.perf_counter_ns() - started
                compressed_disk = (len(encoded) + alignment - 1) // alignment * alignment
                raw_disk = (len(data) + alignment - 1) // alignment * alignment
                if compressed_disk < raw_disk:
                    started = time.perf_counter_ns()
                    restored = transform(codec.decompress(encoded, len(data)), mode, True)
                    restore_ns += time.perf_counter_ns() - started
                    disk_bytes += compressed_disk
                else:
                    restored = data
                    disk_bytes += raw_disk
                    raw_blocks += 1
                if restored != data:
                    raise AssertionError('not lossless')
                original_bytes += len(data)
                blocks += 1
    # A concrete 64-byte index entry budget, padded as a separate index region.
    index_bytes = (blocks * 64 + alignment - 1) // alignment * alignment
    dictionary_bytes = sum((len(d.as_bytes()) + alignment - 1) // alignment * alignment
                           for d in dicts)
    disk_bytes += index_bytes + dictionary_bytes + alignment
    return dict(codec=codec_name, transform=mode, block_mib=block_mib,
                dictionary=dictionary_mode, original_bytes=original_bytes, disk_bytes=disk_bytes,
                disk_ratio=disk_bytes / original_bytes, index_bytes=index_bytes,
                dictionary_bytes=dictionary_bytes, blocks=blocks, raw_blocks=raw_blocks,
                encode_seconds=encode_ns / 1e9, restore_seconds=restore_ns / 1e9,
                restore_gbps=original_bytes / restore_ns if restore_ns else None,
                source_sha256=digest.hexdigest(), bitwise_equal=True,
                serial_estimate_7gbps_ms=(disk_bytes / 7e9 + restore_ns / 1e9) * 1000,
                ideal_overlap_estimate_7gbps_ms=max(disk_bytes / 7e9, restore_ns / 1e9) * 1000)


def self_test():
    rng = np.random.default_rng(71)
    # Includes all FP32 bit patterns, NaNs, signed zero and non-multiple chunk tails.
    data = rng.integers(0, 256, 32768, dtype=np.uint8).tobytes()
    data = struct.pack('<8I', 0, 0x80000000, 0x7f800000, 0xff800000,
                       0x7fc00001, 0x7fa12345, 1, 0xffffffff) + data[32:]
    for mode in ('none', 'byte', 'bit'):
        shuffled = transform(data, mode)
        assert transform(shuffled, mode, True) == data
        for name in ('lz4', 'zstd'):
            codec = Codec(name)
            packed = codec.compress(shuffled)
            assert transform(codec.decompress(packed, len(data)), mode, True) == data
            try:
                codec.decompress(packed, len(data) + 32)
            except (ValueError, zstd.ZstdError):
                pass
            else:
                raise AssertionError('invalid length accepted')
            damaged = bytearray(packed)
            damaged[len(damaged) // 2] ^= 0x80
            try:
                result = transform(codec.decompress(bytes(damaged), len(data)), mode, True)
            except (ValueError, zstd.ZstdError):
                pass
            else:
                assert hashlib.sha256(result).digest() != hashlib.sha256(data).digest()
    samples = [rng.integers(0, 16, 4096, dtype=np.uint8).tobytes() for _ in range(64)]
    dictionary = zstd.train_dictionary(4096, samples)
    wrong = zstd.train_dictionary(4096, [s[::-1] for s in samples])
    encoded = Codec('zstd', dictionary).compress(samples[0])
    assert Codec('zstd', dictionary).decompress(encoded, len(samples[0])) == samples[0]
    try:
        Codec('zstd', wrong).decompress(encoded, len(samples[0]))
    except zstd.ZstdError:
        pass
    else:
        raise AssertionError('wrong dictionary accepted')
    print('PASS: bitwise transforms, codec round trips, corruption, wrong dictionary, lengths', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--experts', default='0,128,255')
    parser.add_argument('--blocks-mib', default='1,4,8,56')
    parser.add_argument('--alignment', type=int, default=4096)
    parser.add_argument('--verify-all', action='store_true')
    args = parser.parse_args()
    if args.alignment < 512 or args.alignment & (args.alignment - 1):
        parser.error('alignment must be a power of two >= 512')
    if any(b <= 0 for b in map(int, args.blocks_mib.split(','))):
        parser.error('block sizes must be positive')
    self_test()
    image = Image(args.image)
    experts = [int(e) for e in args.experts.split(',')]
    rows = []
    report = dict(image=str(args.image.resolve()), header_sha256=image.header_sha256,
                  experts=experts, alignment=args.alignment, zstd_version=zstd.__version__,
                  numpy_version=np.__version__, notes='CPU estimates, not measured IO or P99',
                  rows=rows)

    def record(row):
        rows.append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    for block in map(int, args.blocks_mib.split(',')):
        for mode in ('none', 'byte', 'bit'):
            for codec in ('lz4', 'zstd'):
                record(measure(image, experts, codec, mode, block, args.alignment))
    best_zstd = min((r for r in rows if r['codec'] == 'zstd'),
                    key=lambda r: r['serial_estimate_7gbps_ms'])
    for dictionary in ('global', 'stage'):
        record(measure(image, experts, 'zstd', best_zstd['transform'],
                       best_zstd['block_mib'], args.alignment, dictionary))
    best = min(rows, key=lambda r: r['serial_estimate_7gbps_ms'])
    report['best_estimated'] = best
    if args.verify_all:
        # No dictionary leakage from the verification population: reuse the non-dictionary winner.
        candidates = [r for r in rows if r['dictionary'] == 'none' and r['raw_blocks'] < r['blocks']]
        if not candidates:
            candidates = [r for r in rows if r['dictionary'] == 'none']
        best_plain = min(candidates, key=lambda r: r['serial_estimate_7gbps_ms'])
        report['full_weight_verification'] = measure(
            image, range(image.experts), best_plain['codec'], best_plain['transform'],
            best_plain['block_mib'], args.alignment)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
