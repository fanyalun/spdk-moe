#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause
"""Measure exact request P99 for representative AIO MoE configurations."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PARSER = argparse.ArgumentParser(description='50-request AIO comparison; reimports configured scratch image')
PARSER.add_argument('config', type=Path, help='Authorized existing AIO scratch-image configuration')
PARSER.add_argument('output', type=Path)
PARSER.add_argument('--encoding-comparison', action='store_true')
PARSER.add_argument('--tile-comparison', action='store_true')
PARSER.add_argument('--tile-cols', type=int, nargs='+', default=[64, 128, 256, 512],
                    help='Task channel caps for --tile-comparison (multiples of 64, at most 2048)')
ARGS = PARSER.parse_args()
if any(cols < 64 or cols > 2048 or cols % 64 for cols in ARGS.tile_cols):
    PARSER.error('--tile-cols values must be multiples of 64 between 64 and 2048')
if len(set(ARGS.tile_cols)) != len(ARGS.tile_cols):
    PARSER.error('--tile-cols values must be distinct')
OUT = ARGS.output.resolve()
BASE = ARGS.config.resolve()
TARGET = ROOT / 'build/examples/moe_tgt'
BENCH = ROOT / 'test/moe/fixed_bench/moe_fixed_bench'
ACCURACY = ROOT / 'test/moe/accuracy_initiator/moe_accuracy_initiator'
RPC = ROOT / 'scripts/rpc.py'
COUNT = 50
COMMON = dict(kernel='avx2', pipeline='matrix', cache_slots=7,
              compute_cpu=1, compute_threads=1, io_size=1048576,
              io_depth=4, prefetch=2)
CASES = [('baseline7', {}), ('baseline8', dict(cache_slots=8)),
         ('fifo8', dict(cache_slots=8, pipeline='matrix_fifo')),
         ('fifo8_repeat1', dict(cache_slots=8, pipeline='matrix_fifo')),
         ('baseline8_repeat1', dict(cache_slots=8)),
         ('baseline8_repeat2', dict(cache_slots=8)),
         ('fifo8_repeat2', dict(cache_slots=8, pipeline='matrix_fifo'))]
if ARGS.encoding_comparison:
    CASES = [(label, dict(cache_slots=8, pipeline='matrix_fifo', weight_encoding=encoding))
             for label, encoding in [('raw_fifo8', 'raw'), ('lattice8', 'lattice'),
                 ('lattice8_repeat1', 'lattice'), ('raw_fifo8_repeat1', 'raw'),
                 ('raw_fifo8_repeat2', 'raw'), ('lattice8_repeat2', 'lattice')]]
if ARGS.tile_comparison:
    COMMON.update(cache_slots=8, pipeline='matrix_fifo', weight_encoding='lattice')
    CASES = [('matrix_fifo', {})] + [(f'tile{cols}', dict(pipeline='tile_fifo', compute_tile_cols=cols))
                                    for cols in ARGS.tile_cols]


def cpu_stat():
    return {key: int(value) for key, value in
            (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}


def run_client(label, count, seed, output, target):
    log = OUT / f'{label}.log'
    with log.open('w') as stream:
        process = subprocess.Popen([str(BENCH), str(count), str(seed), '0', '0', str(output)],
                                   stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if target.poll() is not None:
                    raise RuntimeError(f'target exited during {label}')
                if time.monotonic() - started > count * 5 + 120:
                    process.terminate()
                    raise TimeoutError(label)
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if process.returncode:
        raise RuntimeError(f'{label} failed: {process.returncode}; see {log}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_file = OUT / 'results.json'
    results = json.loads(manifest_file.read_text()) if manifest_file.exists() else []
    reference_digest = next((item['input_digest'] for item in results), None)
    source = json.loads(BASE.read_text())
    for subsystem in source['subsystems']:
        for entry in subsystem['config']:
            if entry['method'] == 'bdev_moe_create' and entry['params']['backend'] != 'aio':
                raise ValueError('This local simulation runner requires AIO')
    (OUT / 'method.json').write_text(json.dumps(dict(count=COUNT, seed=42, warmup_count=16,
        warmup_seed=777, p99='nearest rank ceil(0.99*n)',
        binary_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(), target_cpu=10, initiator_cpu=0,
        source_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse',
        'HEAD'], text=True).strip(), source_patch_sha256=hashlib.sha256(
        subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD'])).hexdigest()), indent=2))

    for index, (label, changes) in enumerate(CASES, 1):
        if any(item['label'] == label for item in results):
            continue
        params = dict(COMMON, **changes)
        config = copy.deepcopy(source)
        for subsystem in config['subsystems']:
            for entry in subsystem['config']:
                if entry['method'] == 'bdev_moe_create':
                    entry['params'].update(params)
                    entry['params'].pop('diagnostics', None)
        config_path = OUT / f'{label}.json'
        config_path.write_text(json.dumps(config, indent=2))
        socket = OUT / 'rpc.sock'
        memory_mb = 1664 if params['cache_slots'] == 8 else 1536
        print(f'[{index}/{len(CASES)}] starting {label} params={params}', flush=True)
        stat_before = cpu_stat()
        started = time.monotonic()
        with (OUT / f'{label}_target.log').open('w') as target_log:
            target = subprocess.Popen([str(TARGET), '--no-huge', '--no-pci', '-s',
                str(memory_mb), '-m', '0x400', '-c', str(config_path), '-r', str(socket)],
                stdout=target_log, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, {1, 2, 3, 4, 10}))
            try:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if target.poll() is not None:
                        raise RuntimeError(f'{label} target exited during import')
                    query = subprocess.run(['python3', str(RPC), '-s', str(socket), '-t', '1',
                        'nvmf_get_subsystems'], stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL)
                    if query.returncode == 0 and any(x.get('listen_addresses') and
                        x.get('namespaces') for x in json.loads(query.stdout)):
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(f'{label} import')
                print(f'[{index}/{len(CASES)}] ready {label} import_s={time.monotonic()-started:.1f}',
                      flush=True)
                run_client(f'{label}_warmup', 16, 777, OUT / f'{label}_warmup.jsonl', target)
                output = OUT / f'{label}_requests.jsonl'
                run_client(f'{label}_requests', COUNT, 42, output, target)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                if len(rows) != COUNT:
                    raise RuntimeError(f'{label} incomplete: {len(rows)}/{COUNT}')
                digest = hashlib.sha256(''.join(row['input_hash'] for row in rows).encode()).hexdigest()
                if reference_digest is None:
                    reference_digest = digest
                if digest != reference_digest:
                    raise RuntimeError(f'{label} input sequence differs')
                latencies = sorted(row['latency_ns'] / 1000000 for row in rows)
                env = dict(os.environ, MOE_INITIATOR_CPU_MASK='0x1')
                with (OUT / f'{label}_accuracy.log').open('w') as accuracy_log:
                    subprocess.run([str(ACCURACY), '50' if label in ('fifo8', 'lattice8', 'tile64') else '1', '4242'], stdout=accuracy_log,
                        stderr=subprocess.STDOUT, env=env, timeout=1200, check=True)
                stat_after = cpu_stat()
                item = dict(label=label, parameters=params, request_count=COUNT,
                    p50_ms=latencies[math.ceil(.50 * COUNT)-1],
                    p95_ms=latencies[math.ceil(.95 * COUNT)-1],
                    p99_ms=latencies[math.ceil(.99 * COUNT)-1],
                    max_ms=latencies[-1], mean_ms=statistics.mean(latencies),
                    input_digest=digest, accuracy_passed=True,
                    throttled_delta=stat_after['nr_throttled']-stat_before['nr_throttled'],
                    elapsed_s=time.monotonic()-started,
                    target_memory={line.split(':')[0]: line.split(':')[1].strip()
                        for line in Path(f'/proc/{target.pid}/status').read_text().splitlines()
                        if line.startswith(('VmRSS:', 'VmHWM:', 'HugetlbPages:'))})
                results.append(item)
                manifest_file.write_text(json.dumps(results, indent=2))
                print(f'[{index}/{len(CASES)}] complete {label} P99={item["p99_ms"]:.3f}ms '
                    f'P50={item["p50_ms"]:.3f}ms max={item["max_ms"]:.3f}ms '
                    f'throttle={item["throttled_delta"]} elapsed={item["elapsed_s"]:.1f}s',
                    flush=True)
            finally:
                if target.poll() is None:
                    target.send_signal(signal.SIGTERM)
                try:
                    target.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait()
    print('completed_all=true', flush=True)


if __name__ == '__main__':
    main()
