#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause

"""Validate native reversible FP32 coding against every packed weight block."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import time


class Block(ctypes.Structure):
    _fields_ = [('words', ctypes.c_uint32), ('bytes', ctypes.c_uint32),
               ('scale', ctypes.c_int32), ('base', ctypes.c_int32),
               ('bits', ctypes.c_uint32), ('checksum', ctypes.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('library', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--decode-only', action='store_true')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.moe_codec_encode.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.POINTER(Block)]
    lib.moe_codec_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                    ctypes.POINTER(Block)]
    chunk = 1024 * 1024
    encoded = ctypes.create_string_buffer(chunk + 16)
    restored = ctypes.create_string_buffer(chunk)
    totals = dict(original_bytes=0, disk_bytes=0, blocks=0, raw_blocks=0,
                  encode_seconds=0., restore_seconds=0., widths={}, bitwise_equal=True)
    digest = hashlib.sha256()
    with args.image.open('rb') as stream:
        fields = struct.unpack('<Q6I7Q', stream.read(88))
        version = 2 if args.decode_only else 1
        if fields[:3] != (0x31454f4d4b445053, version, 1):
            raise ValueError(f'requires complete version-{version} image')
        experts, start, stride, gate, down = fields[5], *fields[9:13]
        gate_blocks = (gate + chunk - 1) // chunk
        expert_blocks = 2 * gate_blocks + (down + chunk - 1) // chunk
        index = stream.read(experts * expert_blocks * 24) if args.decode_only else None
        for expert in range(experts):
            for stage in range(3):
                length = down if stage == 2 else gate
                stream.seek(start + expert * stride + stage * gate)
                for position in range(0, length, chunk):
                    size = min(chunk, length - position)
                    if args.decode_only:
                        position_index = expert * expert_blocks + stage * gate_blocks + position // chunk
                        block = Block.from_buffer_copy(index[position_index * 24:(position_index + 1) * 24])
                        if block.words * 4 != size or block.bytes > size:
                            raise ValueError('invalid block index')
                        stream.seek(start + expert * stride + stage * gate + position)
                        payload = stream.read(block.bytes)
                        if len(payload) != block.bytes:
                            raise ValueError('short encoded input')
                        ctypes.memmove(encoded, payload, len(payload))
                        ctypes.memset(ctypes.addressof(encoded) + len(payload), 0, 16)
                    else:
                        data = stream.read(size)
                        if len(data) != size:
                            raise ValueError('short input')
                        digest.update(data)
                        block = Block()
                        t = time.perf_counter()
                        assert lib.moe_codec_encode(data, size // 4, encoded, len(encoded),
                                                    ctypes.byref(block)) == 0
                        totals['encode_seconds'] += time.perf_counter() - t
                    t = time.perf_counter()
                    assert lib.moe_codec_decode(encoded, len(encoded), restored,
                                                ctypes.byref(block)) == 0
                    totals['restore_seconds'] += time.perf_counter() - t
                    if args.decode_only:
                        digest.update(restored.raw[:size])
                    elif restored.raw[:size] != data:
                        raise AssertionError(f'not lossless: expert={expert} stage={stage} offset={position}')
                    totals['original_bytes'] += size
                    totals['disk_bytes'] += max(512, (block.bytes + 511) // 512 * 512)
                    totals['blocks'] += 1
                    totals['raw_blocks'] += block.bits == 32
                    key = str(block.bits)
                    totals['widths'][key] = totals['widths'].get(key, 0) + 1
            if expert % 16 == 0:
                print(f'expert={expert} blocks={totals["blocks"]}', flush=True)
    totals['disk_ratio'] = totals['disk_bytes'] / totals['original_bytes']
    totals['restore_gbps'] = totals['original_bytes'] / totals['restore_seconds'] / 1e9
    totals['source_sha256'] = digest.hexdigest()
    if args.expected_sha256 and totals['source_sha256'] != args.expected_sha256:
        raise AssertionError('restored image fingerprint differs from original weights')
    totals['library_sha256'] = hashlib.sha256(args.library.read_bytes()).hexdigest()
    totals['notes'] = 'All blocks verified; hot reused buffers, not streaming inference throughput'
    args.output.write_text(json.dumps(totals, indent=2) + '\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
