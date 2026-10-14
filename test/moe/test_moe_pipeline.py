#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause
import array
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import subprocess
import struct
import sys
import tempfile


def check_layout(image, weights, encoding='raw', expected_h=256):
    with image.open('rb') as stream:
        fields = struct.unpack('<Q6I7Q', stream.read(88))
        magic, version, complete, d, h, experts, block = fields[:7]
        router_offset, router_bytes, expert_offset, stride, gate_bytes, down_bytes, total = fields[7:]
        assert (magic, version, complete, d, h, experts, block) == (0x31454f4d4b445053, 2 if encoding == 'lattice' else 1, 1, 128, expected_h, 9, 64)
        assert total <= image.stat().st_size
        stream.seek(router_offset)
        assert stream.read(router_bytes) == (weights / 'W_router_128x9.bin').read_bytes()
        for expert in range(experts):
            offset = expert_offset + expert * stride
            for name, rows, cols, size in [('W_gate', d, h, gate_bytes), ('W_up', d, h, gate_bytes),
                                          ('W_down', h, d, down_bytes)]:
                original = (weights / f'{name}_{expert}_{d}x{h}.bin').read_bytes()
                expected = b''.join(original[(row * cols + col) * 4:(row * cols + col + 64) * 4]
                                    for col in range(0, cols, 64) for row in range(rows))
                stream.seek(offset)
                if encoding == 'raw':
                    actual = stream.read(size)
                else:
                    matrix = ['W_gate', 'W_up', 'W_down'].index(name)
                    blocks = (size + 1048575) // 1048576
                    actual = bytearray()
                    for piece in range(blocks):
                        stream.seek(88 + (expert * 3 * blocks + matrix * blocks + piece) * 24)
                        words, length, scale, base, bits, checksum = struct.unpack('<IIiiII', stream.read(24))
                        stream.seek(offset + piece * 1048576)
                        payload = stream.read(length)
                        if bits == 32:
                            actual.extend(payload)
                        else:
                            mask = (1 << bits) - 1
                            actual.extend(b''.join(struct.pack('<f', math.ldexp(float(base +
                                ((int.from_bytes(payload[(i * bits) // 8:(i * bits) // 8 + 8], 'little')
                                  >> ((i * bits) % 8)) & mask)), -scale)) for i in range(words)))
                assert actual == expected + b'\0' * (size - len(expected))
                offset += size


def check_matrix_trace(rows, require_overlap=False, tiled=False):
    early = overlap = False
    for row in rows:
        if row['status']:
            continue
        events = {(e['selected'], e['stage']): e for e in row['stages']}
        for (selected, stage), event in events.items():
            start, end = event['compute_start'], event['compute_end']
            if not start:
                continue
            assert event['read_start'] <= event['read_submit'] <= start <= end
            if not tiled:
                assert event['read_end'] <= start
            if stage and not tiled:
                assert events[selected, stage - 1]['compute_end'] <= start
            if stage < 2:
                assert events[selected, stage + 1]['read_start'] <= start
            if stage == 0:
                early |= start < events[selected, 2]['read_end']
            for other in events.values():
                if other['selected'] == selected:
                    overlap |= bool(other['read_submit']) and max(start, other['read_submit']) < min(end, other['read_end'])
    if require_overlap:
        assert early and overlap, 'expected actual intra-expert overlap under slow I/O'


def main():
    root = Path(__file__).resolve().parents[2]
    wide = bool(os.environ.get('MOE_TEST_WIDE'))
    h = 4096 if wide else 256
    binary = root / ('test/moe/tile_progress/moe_tile_progress_test' if wide else
                     'test/moe/pipeline/moe_pipeline_test')
    artifacts = Path(tempfile.mkdtemp(prefix='moe_final_pipeline_'))
    weights = artifacts / 'weights'
    weights.mkdir()
    rng = random.Random(42)
    for name, count in [('W_router_128x9.bin', 128 * 9)] + [
        (f'{kind}_{expert}_128x{h}.bin', 128 * h)
        for expert in range(9) for kind in ('W_gate', 'W_up', 'W_down')
    ]:
        data = array.array('f', (rng.randint(-1048576, 1048576) / 16777216
                                for _ in range(count)))
        if sys.byteorder != 'little':
            data.byteswap()
        (weights / name).write_bytes(data.tobytes())
    cases = ['', 'truncate', 'remove', 'slow', 'strict', 'shutdown', 'short_source',
             'warmup_failure', 'warmup_retry', 'warmup_alloc_retry',
             'reject_tile', 'reject_raw', 'reject_threads', 'reject_decode', 'reject_slots']
    selected = os.environ.get('MOE_TEST_CASE')
    if selected is not None:
        if selected not in cases:
            raise ValueError(f'unknown test case: {selected}')
        cases = [selected]
    print(f'artifacts={artifacts}', flush=True)
    for fault in cases:
        label = fault or 'normal'
        image = artifacts / f'{label}.bin'
        with image.open('xb') as stream:
            stream.truncate((56 if wide else 11) * 1024 * 1024)
        stats_path = artifacts / f'{label}_stats.jsonl'
        params = dict(name='moe_test', backend='aio', base_bdev='weight_aio',
                      weight_dir=str(weights), d_model=128, d_ff=h, num_experts=9,
                      top_k=8, cache_slots=8, pipeline='matrix_fifo', kernel='avx2',
                      weight_encoding='lattice', io_size=1048576, io_depth=16,
                      compute_threads=1, decode_threads=1, warmup_requests=16,
                      diagnostics=str(stats_path))
        if wide:
            params['compute_cpu'] = 5
        overrides = dict(reject_tile=dict(pipeline='tile_fifo'),
                         reject_raw=dict(weight_encoding='raw'),
                         reject_threads=dict(compute_threads=2),
                         reject_decode=dict(decode_threads=0),
                         reject_slots=dict(cache_slots=9))
        params.update(overrides.get(fault, {}))
        if fault == 'strict':
            params['backend'] = 'nvme'
        config = [
            dict(method='bdev_set_options', params=dict(bdev_io_pool_size=4096,
                                                       bdev_io_cache_size=128)),
            dict(method='bdev_aio_create', params=dict(name='weight_aio',
                                                      filename=str(image), block_size=512))]
        if fault in ('slow', 'shutdown'):
            config.append(dict(method='bdev_set_qos_limit',
                               params=dict(name='weight_aio', r_mbytes_per_sec=16)))
        config.append(dict(method='bdev_moe_create', params=params))
        config_path = artifacts / f'{label}.json'
        config_path.write_text(json.dumps(dict(subsystems=[dict(subsystem='bdev', config=config)])))
        originals = {}
        if fault == 'short_source':
            path = weights / f'W_gate_0_128x{h}.bin'
            originals[path] = path.read_bytes()
            path.write_bytes(b'\\0' * 4)
        if fault == 'warmup_failure':
            for expert in range(9):
                path = weights / f'W_gate_{expert}_128x{h}.bin'
                originals[path] = path.read_bytes()
                path.write_bytes(struct.pack('<f', math.nan) + originals[path][4:])
        environment = dict(os.environ, MOE_TEST_WEIGHTS=str(weights),
                           MOE_TEST_STARTUP_WARMUP='16')
        retry = fault in ('warmup_retry', 'warmup_alloc_retry')
        if retry:
            environment['MOE_TEST_STARTUP_READ_FAIL' if fault == 'warmup_retry' else
                        'MOE_TEST_STARTUP_ALLOC_FAIL'] = '1'
        if fault not in ('truncate', 'remove'):
            environment['MOE_TEST_REPEAT_INPUT'] = '1'
        if fault == 'shutdown':
            environment['MOE_TEST_EXPECT_SHUTDOWN'] = '1'
        elif fault == 'truncate':
            environment['MOE_TEST_TRUNCATE'] = str(image)
        elif fault == 'remove':
            environment['MOE_TEST_REMOVE'] = '1'
        elif fault == 'slow':
            environment['MOE_TEST_SLOW'] = '1'
        log = artifacts / f'{label}.log'
        command = [str(binary), '-c', str(config_path), '-m',
                   environment.get('MOE_TEST_REACTOR_MASK', '0x1'),
                   '-r', str(artifacts / 'rpc.sock')]
        try:
            with log.open('w') as output:
                if fault == 'shutdown':
                    process = subprocess.Popen(command, env=environment, stdout=output,
                                               stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + 60
                        while 'vendor_request_submitted=1' not in log.read_text():
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError('shutdown test startup failed')
                            time.sleep(0.05)
                        time.sleep(0.3)
                        process.send_signal(signal.SIGTERM)
                        rc = process.wait(timeout=30)
                        assert 'inflight_shutdown_drained=1' in log.read_text()
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                else:
                    rc = subprocess.run(command, env=environment, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=120).returncode
        finally:
            for path, original in originals.items():
                path.write_bytes(original)
        text = log.read_text()
        print(f'{label} exit={rc} log={log}', flush=True)
        rejected = fault in ('strict', 'short_source', 'warmup_failure') or fault.startswith('reject_')
        if bool(rc) != rejected:
            print(text[-10000:])
            return 1
        if rejected:
            assert 'moe_test_ready=1' not in text
            if fault == 'warmup_failure':
                assert 'MoE startup warmup failed:' in text
                assert text.count('MoE startup retry:') == 1
            if fault == 'short_source':
                with image.open('rb') as stream:
                    assert struct.unpack('<Q6I7Q', stream.read(88))[2] == 0
            continue
        assert text.index('MoE startup warmup completed: requests=16 ') < text.index('moe_test_ready=1')
        assert 'startup_output_reset=1' in text
        if retry:
            assert text.count('MoE startup retry:') == 1
            assert 'cache_slots=8->7 pipeline=matrix' in text
            marker = 'startup_read_failure_injected=1' if fault == 'warmup_retry' else 'startup_allocation_failure_injected=1'
            assert marker in text
        else:
            assert 'MoE startup retry:' not in text
        if fault not in ('truncate', 'remove'):
            check_layout(image, weights, 'lattice', h)
        rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
        check_matrix_trace(rows, require_overlap=fault == 'slow')
        assert all(row['pipeline'] == ('matrix' if retry else 'matrix_fifo') for row in rows)
        assert all(row['decode_threads'] == 1 and row['decode_pending'] == 0 and
                   row['codec_buffer_slots'] == 32 for row in rows)
        if fault == '':
            assert rows[1]['cache_hits_total'] - rows[0]['cache_hits_total'] == 8
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
