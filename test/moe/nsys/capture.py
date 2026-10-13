#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Profile the 256-expert AIO target; reimports its image')
    parser.add_argument('--nsys', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(nsys_version=subprocess.check_output([args.nsys, '--version'], text=True),
                    source_config=str(args.config.resolve()),
                    revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                     text=True).strip(), cases=[])
    (output / 'source.patch').write_bytes(subprocess.check_output(
        ['git', '-C', str(root), 'diff', 'HEAD']))
    sock = str(output / 'rpc.sock')

    def rpc(*arguments, timeout=5):
        return subprocess.run(['python3', str(root / 'scripts/rpc.py'), '-s', sock,
                               '-t', str(timeout), *arguments], capture_output=True, text=True,
                              timeout=timeout + 5)

    for pipeline in ('expert', 'matrix'):
        current = copy.deepcopy(config)
        for subsystem in current['subsystems']:
            for entry in subsystem['config']:
                if entry['method'] == 'bdev_moe_create':
                    entry['params'].update(pipeline=pipeline, weight_encoding='raw', d_model=2048, d_ff=7168,
                                           num_experts=256, top_k=8)
                    entry['params']['diagnostics'] = str(output / f'{pipeline}.jsonl')
        config_path = output / f'{pipeline}.json'
        config_path.write_text(json.dumps(current, indent=2))
        slots = next(entry['params'].get('cache_slots', 8) for subsystem in current['subsystems']
                     for entry in subsystem['config'] if entry['method'] == 'bdev_moe_create')
        memory = '1664' if slots == 8 else '1536'
        command = [args.nsys, 'profile', '--trace=nvtx,osrt', '--sample=none',
                   '--cpuctxsw=none', '--force-overwrite=false', '-o', str(output / pipeline),
                   str(root / 'build/examples/moe_tgt'), '--no-huge', '--no-pci',
                   '-c', str(config_path), '-m', '0x400', '-s', memory, '-r', sock]
        print(f'Starting {pipeline}: 256 experts, Top-8, existing image import', flush=True)
        case = dict(label=pipeline, command=command)
        with (output / f'{pipeline}_target.log').open('w') as log:
            profiler = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            ready = False
            try:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if profiler.poll() is not None:
                        raise RuntimeError(f'target exited during initialization: {pipeline}')
                    query = rpc('nvmf_get_subsystems', timeout=1)
                    if query.returncode == 0 and any(
                            item.get('listen_addresses') and item.get('namespaces')
                            for item in json.loads(query.stdout)):
                        ready = True
                        break
                    time.sleep(1)
                if not ready:
                    raise TimeoutError('target import')
                query = rpc('bdev_get_bdevs')
                query.check_returncode()
                (output / f'{pipeline}_bdevs.json').write_text(query.stdout)
                case['cpu_before'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
                bench = [str(root / 'build/examples/moe_bench'), '--cpu=0', '--seed=42',
                         '--warmup=0', f'--runtime={args.runtime}',
                         f'--json-output={output / (pipeline + "_bench.json")}']
                case['benchmark_command'] = bench
                print(f'Benchmark {pipeline}: {args.runtime}s', flush=True)
                with (output / f'{pipeline}_bench.log').open('w') as bench_log:
                    subprocess.run(bench, stdout=bench_log, stderr=subprocess.STDOUT,
                                   check=True, timeout=args.runtime + 120)
                case['cpu_after'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
                print(f'Independent accuracy check {pipeline}', flush=True)
                with (output / f'{pipeline}_accuracy.log').open('w') as accuracy_log:
                    subprocess.run([str(root / 'test/moe/accuracy_initiator/moe_accuracy_initiator')],
                                   stdout=accuracy_log, stderr=subprocess.STDOUT,
                                   check=True, timeout=300)
                case['accuracy_passed'] = True
            finally:
                if profiler.poll() is None:
                    stopped = rpc('spdk_kill_instance', 'SIGTERM')
                    case['shutdown_rpc_returncode'] = stopped.returncode
                    try:
                        profiler.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        profiler.terminate()
                        profiler.wait(timeout=30)
                case['profiler_returncode'] = profiler.returncode
                manifest['cases'].append(case)
                (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            if profiler.returncode or not (output / f'{pipeline}.nsys-rep').exists():
                raise RuntimeError(f'capture failed: {pipeline}')
        print(f'Finished {pipeline}', flush=True)
    print(f'Reports: {output}', flush=True)


if __name__ == '__main__':
    main()
