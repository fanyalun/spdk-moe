#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause
import array
import json
import os
from pathlib import Path
import random
import signal
import time
import subprocess
import struct
import sys
import tempfile


def check_layout(image, weights):
    with image.open('rb') as stream:
        fields = struct.unpack('<Q6I7Q', stream.read(88))
        magic, version, complete, d, h, experts, block = fields[:7]
        router_offset, router_bytes, expert_offset, stride, gate_bytes, down_bytes, total = fields[7:]
        assert (magic, version, complete, d, h, experts, block) == (0x31454f4d4b445053, 1, 1, 128, 256, 9, 64)
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
                assert stream.read(size) == expected + b'\0' * (size - len(expected))
                offset += size


def check_matrix_trace(rows, require_overlap=False):
    early = overlap = False
    for row in rows:
        if row['status']:
            continue
        events = {(e['selected'], e['stage']): e for e in row['stages']}
        for (selected, stage), event in events.items():
            start, end = event['compute_start'], event['compute_end']
            if not start:
                continue
            assert event['read_start'] <= event['read_submit'] <= event['read_end'] <= start <= end
            if stage:
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
    binary = root / 'test/moe/pipeline/moe_pipeline_test'
    artifacts = Path(tempfile.mkdtemp(prefix='moe_pipeline_'))
    weights = artifacts / 'weights'
    weights.mkdir()
    rng = random.Random(42)
    for name, count in [('W_router_128x9.bin', 128 * 9)] + [
        (f'{kind}_{expert}_128x256.bin', 128 * 256)
        for expert in range(9) for kind in ('W_gate', 'W_up', 'W_down')
    ]:
        data = array.array('f', (rng.uniform(-0.1, 0.1) for _ in range(count)))
        if sys.byteorder != 'little':
            data.byteswap()
        with (weights / name).open('wb') as stream:
            data.tofile(stream)
    print(f'artifacts={artifacts}', flush=True)
    cases = [('file', 1, ''), ('file', 7, ''), ('aio', 1, ''),
                                  ('aio', 7, ''), ('aio', 8, ''),
                                  ('aio', 7, 'threads2'), ('aio', 7, 'threads4'),
                                  ('aio', 1, 'threads4'),
                                  ('aio', 6, 'tune'), ('aio', 8, 'large_chunk'),
                                  ('aio', 7, 'truncate'), ('aio', 7, 'remove'),
                                  ('aio', 7, 'slow'), ('aio', 7, 'strict'),
                                  ('aio', 7, 'shutdown'), ('aio', 7, 'bad_gate'), ('aio', 7, 'bad_up'), ('aio', 7, 'bad_down'),
             ('file', 1, 'matrix_reject'), ('aio', 7, 'bad_pipeline'),
             ('aio', 7, 'short_source')]
    selected_case = os.environ.get('MOE_TEST_CASE')
    if selected_case is not None:
        cases = [case for case in cases if case[2] == selected_case]
        if not cases:
            raise ValueError(f'unknown test case: {selected_case}')
    for backend, slots, fault, pipeline in [
        (*case, mode) for case in cases
        for mode in (['matrix'] if case[2] == 'short_source' else
                     ['expert', 'matrix'] if case[0] == 'aio' else ['expert'])]:
        label = f'{backend}_{slots}_{fault or "normal"}_{pipeline}'
        image = artifacts / f'{label}.bin'
        if backend == 'aio':
            with image.open('xb') as stream:
                stream.truncate(11 * 1024 * 1024)
        params = dict(name='moe_test', backend=backend, weight_dir=str(weights),
                      d_model=128, d_ff=256, num_experts=9, top_k=8, cache_slots=slots,
                      diagnostics=str(artifacts / f'{label}_stats.jsonl'))
        params['pipeline'] = pipeline
        params['compute_threads'] = int(fault[-1]) if fault.startswith('threads') else 1
        if fault == 'tune':
            params.update(io_size=256 * 1024, io_depth=16, prefetch=4, compute_threads=4)
        elif fault == 'large_chunk':
            params.update(io_size=2 * 1024 * 1024, io_depth=1, prefetch=1)
        config = [dict(method='bdev_set_options', params=dict(
            bdev_io_pool_size=4096, bdev_io_cache_size=128))]
        if backend == 'aio':
            params['base_bdev'] = 'weight_aio'
            config.append(dict(method='bdev_aio_create', params=dict(
                name='weight_aio', filename=str(image), block_size=512)))
        if fault in ('slow', 'shutdown'):
            config.append(dict(method='bdev_set_qos_limit', params=dict(name='weight_aio', r_mbytes_per_sec=1)))
        if fault == 'strict':
            params['backend'] = 'nvme'
        if fault == 'matrix_reject':
            params['pipeline'] = 'matrix'
        if fault == 'bad_pipeline':
            params['pipeline'] = 'invalid'
        if fault == 'short_source':
            (weights / 'W_gate_0_128x256.bin').write_bytes(b'\0' * 4)
        config.append(dict(method='bdev_moe_create', params=params))
        config_path = artifacts / f'{label}.json'
        config_path.write_text(json.dumps(dict(subsystems=[dict(subsystem='bdev', config=config)])))
        log = artifacts / f'{label}.log'
        environment = dict(os.environ, MOE_TEST_WEIGHTS=str(weights))
        if slots == 8:
            environment['MOE_TEST_REPEAT_INPUT'] = '1'
        if fault in ('bad_gate', 'bad_up', 'bad_down'):
            environment['MOE_TEST_BAD_IMAGE'] = str(image)
            environment['MOE_TEST_BAD_MATRIX'] = str(['bad_gate', 'bad_up', 'bad_down'].index(fault))
        if fault == 'shutdown':
            environment['MOE_TEST_EXPECT_SHUTDOWN'] = '1'
        if fault == 'truncate':
            environment['MOE_TEST_TRUNCATE'] = str(image)
        elif fault == 'remove':
            environment['MOE_TEST_REMOVE'] = '1'
        elif fault == 'slow':
            environment['MOE_TEST_SLOW'] = '1'
        with log.open('w') as output:
            command = [str(binary), '-c', str(config_path), '-m',
                       environment.get('MOE_TEST_REACTOR_MASK', '0x1'),
                       '-r', str(artifacts / 'rpc.sock')]
            if fault == 'shutdown':
                process = subprocess.Popen(command, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    while 'vendor_request_submitted=1' not in log.read_text():
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('shutdown test startup failed')
                        time.sleep(0.05)
                    time.sleep(0.3)
                    process.send_signal(signal.SIGTERM)
                    rc = process.wait(timeout=30)
                    assert 'inflight_shutdown_drained=1' in log.read_text()
                    run = subprocess.CompletedProcess(command, rc)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            else:
                run = subprocess.run(command, env=environment, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=60)
        print(f'{label} exit={run.returncode} log={log}', flush=True)
        if bool(run.returncode) != (fault in ('strict', 'short_source', 'matrix_reject', 'bad_pipeline')):
            print(log.read_text()[-10000:])
            return 1
        if backend == 'aio' and fault not in ('truncate', 'strict', 'short_source', 'bad_pipeline'):
            check_layout(image, weights)
        if fault == 'short_source':
            with image.open('rb') as stream:
                header = struct.unpack('<Q6I7Q', stream.read(88))
            assert header[2] == 0, 'failed import must remain incomplete'
        if pipeline == 'matrix' and not run.returncode:
            rows = [json.loads(line) for line in (artifacts / f'{label}_stats.jsonl').read_text().splitlines()]
            check_matrix_trace(rows, require_overlap=fault == 'slow')
        if slots == 8 and not run.returncode:
            stats = [json.loads(line) for line in (artifacts / f'{label}_stats.jsonl').read_text().splitlines()]
            assert stats[1]['cache_hits_total'] == 8, 'repeated input must fully hit eight cached experts'
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
