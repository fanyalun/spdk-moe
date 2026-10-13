#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Capture 16 warmup + 50 reversible MoE requests')
    parser.add_argument('--nsys', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accuracy-count', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.accuracy_count <= 50:
        parser.error('--accuracy-count must be between 1 and 50')
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for subsystem in config['subsystems']:
        for entry in subsystem['config']:
            if entry['method'] == 'bdev_moe_create':
                if entry['params']['backend'] != 'aio':
                    raise ValueError('Local capture requires authorized AIO scratch configuration')
                pipeline = 'tile_fifo' if entry['params'].get('pipeline') == 'tile_fifo' else 'matrix_fifo'
                entry['params'].update(pipeline=pipeline, weight_encoding='lattice',
                    cache_slots=8, io_size=1048576, diagnostics=str(output / 'lattice.jsonl'))
    config_path = output / 'lattice.json'
    config_path.write_text(json.dumps(config, indent=2))
    sock = str(output / 'rpc.sock')

    def rpc(*arguments, timeout=5):
        return subprocess.run(['python3', str(root / 'scripts/rpc.py'), '-s', sock,
            '-t', str(timeout), *arguments], capture_output=True, text=True, timeout=timeout + 5)

    target = root / 'build/examples/moe_tgt'
    command = [args.nsys, 'profile', '--trace=nvtx', '--sample=none',
        '--capture-range=nvtx', '--nvtx-capture=PROFILE_WINDOW', '--capture-range-end=stop',
        '--cpuctxsw=none', '--force-overwrite=false', '-o', str(output / 'moe_cpu_experts_io_decode'),
        str(target), '--no-huge', '--no-pci', '-c', str(config_path), '-m', '0x400',
        '-s', '1664', '-r', sock]
    manifest = dict(command=command, warmup_count=16, warmup_seed=777, count=50, seed=42,
        accuracy_count=args.accuracy_count,
        nsys_version=subprocess.check_output([args.nsys, '--version'], text=True).strip(),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    manifest['capture_window'] = dict(first_request=17, last_request=66, includes_warmup=False)
    sources = ['lib/moe_ffn/moe_trace.h', 'lib/moe_ffn/moe_workspace.c',
               'module/bdev/moe/bdev_moe.c', 'module/bdev/moe/moe_store.c']
    manifest['source_sha256'] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                                for path in sources}
    (output / 'moe_trace_source.h').write_bytes((root / sources[0]).read_bytes())
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root))
    print(f'Importing weights; capture output={output}', flush=True)
    with (output / 'lattice_target.log').open('w') as log:
        environment = dict(os.environ, MOE_PROFILE_FIRST='17', MOE_PROFILE_COUNT='50')
        profiler = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 900
            while True:
                if profiler.poll() is not None:
                    raise RuntimeError('target exited during import')
                query = rpc('nvmf_get_subsystems', timeout=1)
                if query.returncode == 0 and any(item.get('listen_addresses') and
                    item.get('namespaces') for item in json.loads(query.stdout)):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('weight import')
                time.sleep(1)
            query = rpc('bdev_get_bdevs')
            query.check_returncode()
            (output / 'lattice_bdevs.json').write_text(query.stdout)
            manifest['cpu_before'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
            for label, count, seed in [('warmup', 16, 777), ('requests', 50, 42)]:
                print(f'{label}: count={count} seed={seed}', flush=True)
                with (output / f'lattice_{label}.log').open('w') as bench_log:
                    subprocess.run([str(root / 'test/moe/fixed_bench/moe_fixed_bench'),
                        str(count), str(seed), '0', '0', str(output / f'lattice_{label}.jsonl')],
                        stdout=bench_log, stderr=subprocess.STDOUT, check=True, timeout=300)
            manifest['cpu_after'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
            print('Independent TCP accuracy check', flush=True)
            with (output / 'lattice_accuracy.log').open('w') as accuracy_log:
                subprocess.run([str(root / 'test/moe/accuracy_initiator/moe_accuracy_initiator'),
                    str(args.accuracy_count), '4242'], stdout=accuracy_log, stderr=subprocess.STDOUT,
                    check=True, timeout=1200)
            manifest['accuracy_passed'] = True
        finally:
            if profiler.poll() is None:
                manifest['shutdown_rpc_returncode'] = rpc('spdk_kill_instance', 'SIGTERM').returncode
                try:
                    profiler.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    profiler.terminate()
                    profiler.wait(timeout=30)
            manifest['profiler_returncode'] = profiler.returncode
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        if profiler.returncode or not (output / 'moe_cpu_experts_io_decode.nsys-rep').exists():
            raise RuntimeError('Nsight capture failed')
    subprocess.run([args.nsys, 'export', '--type', 'sqlite', '--output',
        str(output / 'lattice.sqlite'), str(output / 'moe_cpu_experts_io_decode.nsys-rep')], check=True)
    print(f'Capture complete: {output}', flush=True)


if __name__ == '__main__':
    main()
