#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Alternate expert/matrix targets using one explicitly supplied, reimported image."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='Existing AIO target JSON; reimports its image')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--runtime', type=int, default=60)
    parser.add_argument('--cpu', type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1 or args.runtime < 1:
        parser.error('rounds and runtime must be positive')
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    entries = [item for subsystem in config['subsystems'] for item in subsystem['config']]
    moe = [item for item in entries if item['method'] == 'bdev_moe_create']
    if len(moe) != 1 or moe[0]['params']['backend'] != 'aio':
        parser.error('requires exactly one AIO MoE bdev')
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    used = int(Path('/sys/fs/cgroup/memory.current').read_text())
    estimate = (168 * moe[0]['params'].get('cache_slots', 7) + 161) * 1024 ** 2
    if limit != 'max' and used + estimate > int(limit):
        parser.error('insufficient shared cgroup headroom for target; use an exclusive environment')
    artifacts = Path(tempfile.mkdtemp(prefix='moe_pipeline_bench_'))
    print(f'artifacts={artifacts}', flush=True)
    manifest = dict(command=os.sys.argv, revision=subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(), results=[])
    (artifacts / 'working.patch').write_bytes(subprocess.check_output(
        ['git', '-C', str(root), 'diff', 'HEAD']))
    for round_number in range(args.rounds):
        for mode in ('expert', 'matrix'):
            label = f'{round_number + 1}_{mode}'
            current = copy.deepcopy(config)
            for subsystem in current['subsystems']:
                for item in subsystem['config']:
                    if item['method'] == 'bdev_moe_create':
                        item['params']['pipeline'] = mode
                        item['params'].pop('diagnostics', None)
            path = artifacts / f'{label}.json'
            path.write_text(json.dumps(current, indent=2))
            sock = str(artifacts / 'rpc.sock')
            stop = threading.Event()
            with (artifacts / f'{label}_target.log').open('w') as log:
                target = subprocess.Popen([str(root / 'build/examples/moe_tgt'), '--no-huge',
                                           '--no-pci', '-s', '1536', '-m', '0x1',
                                           '-c', str(path), '-r', sock], stdout=log,
                                          stderr=subprocess.STDOUT)

                def sample_memory():
                    with (artifacts / f'{label}_memory.jsonl').open('w') as output:
                        while not stop.is_set():
                            try:
                                status = Path(f'/proc/{target.pid}/status').read_text()
                            except FileNotFoundError:
                                break
                            values = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                                      for line in status.splitlines()
                                      if line.startswith(('VmRSS:', 'VmHWM:', 'HugetlbPages:'))}
                            output.write(json.dumps(dict(time=time.monotonic(), **values)) + '\n')
                            stop.wait(0.25)

                sampler = threading.Thread(target=sample_memory)
                sampler.start()
                try:
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        if target.poll() is not None:
                            raise RuntimeError(f'target failed: {label}')
                        query = subprocess.run(['python3', str(root / 'scripts/rpc.py'),
                                                '-s', sock, '-t', '1', 'nvmf_get_subsystems'],
                                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                        if query.returncode == 0 and any(
                                x.get('listen_addresses') and x.get('namespaces')
                                for x in json.loads(query.stdout)):
                            break
                        time.sleep(1)
                    else:
                        raise TimeoutError(f'target initialization: {label}')
                    output_json = artifacts / f'{label}_bench.json'
                    command = [str(root / 'build/examples/moe_bench'), f'--cpu={args.cpu}',
                               '--seed=42', '--warmup=0', f'--runtime={args.runtime}',
                               f'--json-output={output_json}']
                    cpu_before = Path('/sys/fs/cgroup/cpu.stat').read_text()
                    with (artifacts / f'{label}_bench.log').open('w') as output:
                        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                       check=True, timeout=args.runtime + 120)
                    result = dict(label=label, command=command,
                                  cpu_before=cpu_before,
                                  cpu_after=Path('/sys/fs/cgroup/cpu.stat').read_text(),
                                  benchmark=json.loads(output_json.read_text()))
                    manifest['results'].append(result)
                    (artifacts / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                    print(f'completed={label}', flush=True)
                finally:
                    if target.poll() is None:
                        target.send_signal(signal.SIGTERM)
                    try:
                        target.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        target.kill()
                        target.wait()
                        raise
                    finally:
                        stop.set()
                        sampler.join()
                if target.returncode:
                    raise RuntimeError(f'target shutdown failed: {label}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
