#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Screen AIO parameters with identical request sequences; reimports the supplied image."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--cpu', type=int, default=10)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.count < 2 or args.cpu < 0:
        parser.error('count >= 2 and cpu >= 0 required')
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    entries = [e for s in config['subsystems'] for e in s['config']]
    opts = [e['params'] for e in entries if e['method'] == 'bdev_moe_create']
    if len(opts) != 1 or opts[0]['backend'] != 'aio':
        parser.error('requires one AIO MoE target with an overwriteable test image')
    artifacts = args.resume or Path(tempfile.mkdtemp(prefix='moe_tuning_'))
    print(f'artifacts={artifacts}', flush=True)
    manifest = dict(command=os.sys.argv, revision=subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(), results=[], choices=[])
    if args.resume:
        manifest = json.loads((artifacts / 'manifest.json').read_text())
    else:
        (artifacts / 'source.patch').write_bytes(subprocess.check_output(
            ['git', '-C', str(root), 'diff', 'HEAD']))
    hashes = {}
    for result in manifest['results']:
        for batch in result['batches']:
            key = (batch['seed'], batch['repeat'], batch['count'])
            if hashes.setdefault(key, batch['input_digest']) != batch['input_digest']:
                raise RuntimeError('saved input sequences differ')

    def save():
        (artifacts / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    def run(label, parameters, count):
        previous = [x for x in manifest['results'] if x['label'] == label]
        if previous:
            if previous[0]['parameters'] != parameters or previous[0]['batches'][0]['count'] != count:
                raise RuntimeError('resume parameters differ')
            return previous[0]
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        if limit != 'max' and used + 2200 * 1024**2 > int(limit):
            raise RuntimeError('insufficient cgroup headroom')
        current = copy.deepcopy(config)
        for subsystem in current['subsystems']:
            for entry in subsystem['config']:
                if entry['method'] == 'bdev_moe_create':
                    entry['params'].update({k: v for k, v in parameters.items() if k != 'memory_mb'})
                    entry['params'].pop('diagnostics', None)
        path = artifacts / f'{label}.json'
        path.write_text(json.dumps(current, indent=2))
        sock = str(artifacts / 'rpc.sock')
        stop = threading.Event()
        rss = []
        print(f'start={label} params={parameters}', flush=True)
        with (artifacts / f'{label}_target.log').open('w') as log:
            target = subprocess.Popen([str(root / 'build/examples/moe_tgt'), '--no-huge',
                                       '--no-pci', '-s', str(parameters.get('memory_mb', 1536)), '-m', '0x1',
                                       '-c', str(path), '-r', sock], stdout=log,
                                      stderr=subprocess.STDOUT)

            def sample():
                while not stop.is_set():
                    try:
                        text = Path(f'/proc/{target.pid}/status').read_text()
                        rss.append(int(next(x for x in text.splitlines()
                                            if x.startswith('VmRSS:')).split()[1]))
                    except (FileNotFoundError, StopIteration):
                        break
                    stop.wait(0.25)

            sampler = threading.Thread(target=sample)
            sampler.start()
            try:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if target.poll() is not None:
                        raise RuntimeError(f'target exited during import: {label}')
                    query = subprocess.run(['python3', str(root / 'scripts/rpc.py'), '-s', sock,
                                            '-t', '1', 'nvmf_get_subsystems'],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    if query.returncode == 0 and any(x.get('listen_addresses') and x.get('namespaces')
                                                    for x in json.loads(query.stdout)):
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(label)
                topology = subprocess.check_output(['python3', str(root / 'scripts/rpc.py'),
                                                    '-s', sock, 'bdev_get_bdevs'], text=True)
                (artifacts / f'{label}_bdevs.json').write_text(topology)
                result = dict(label=label, parameters=parameters.copy(), batches=[],
                              cpu_before=Path('/sys/fs/cgroup/cpu.stat').read_text())
                for seed, repeat, n in [(42, 0, count), (43, 0, count), (44, 0, count), (42, 1, 8)]:
                    name = f'{label}_{seed}_{repeat}'
                    output = artifacts / f'{name}.jsonl'
                    with (artifacts / f'{name}.log').open('w') as client_log:
                        subprocess.run([str(root / 'test/moe/fixed_bench/moe_fixed_bench'),
                                        str(n), str(seed), str(args.cpu), str(repeat), str(output)],
                                       stdout=client_log, stderr=subprocess.STDOUT,
                                       check=True, timeout=n * 15 + 60)
                    data = [json.loads(x) for x in output.read_text().splitlines()]
                    if len(data) != n:
                        raise RuntimeError('incomplete fixed-count run')
                    digest = hashlib.sha256(''.join(x['input_hash'] for x in data).encode()).hexdigest()
                    key = (seed, repeat, n)
                    if hashes.setdefault(key, digest) != digest:
                        raise RuntimeError('input sequences differ')
                    latencies = [x['latency_ns'] / 1e6 for x in data]
                    result['batches'].append(dict(seed=seed, repeat=repeat, count=n,
                                                 mean_ms=statistics.mean(latencies),
                                                 max_ms=max(latencies),
                                                 steady_mean_ms=statistics.mean(latencies[1:]),
                                                 input_digest=digest))
                result['cpu_after'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
                env = dict(os.environ, MOE_INITIATOR_CPU_MASK=hex(1 << args.cpu))
                with (artifacts / f'{label}_accuracy.log').open('w') as accuracy:
                    subprocess.run([str(root / 'test/moe/accuracy_initiator/moe_accuracy_initiator')],
                                   env=env, stdout=accuracy, stderr=subprocess.STDOUT,
                                   check=True, timeout=120)
                result['accuracy_passed'] = True
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
        result['rss_peak_kib'] = max(rss)
        result['mean_ms'] = statistics.mean(b['mean_ms'] for b in result['batches'][:3])
        result['max_ms'] = max(b['max_ms'] for b in result['batches'][:3])
        manifest['results'].append(result)
        save()
        print(f'completed={label} mean_ms={result["mean_ms"]:.3f} '
              f'max_ms={result["max_ms"]:.3f} rss_kib={max(rss)}', flush=True)
        return result

    best = dict(kernel='avx2', pipeline='matrix', cache_slots=7, compute_threads=1,
                compute_cpu=1, io_size=1048576, io_depth=4, prefetch=2)
    baseline = run('kernel_avx2', best, args.count)
    current = baseline
    for stage, key, values in [('kernel', 'kernel', ['avx512']),
                               ('slots', 'cache_slots', [8]),
                               ('size', 'io_size', [262144, 2097152]),
                               ('depth', 'io_depth', [1, 8]),
                               ('threads', 'compute_threads', [2, 4])]:
        candidates = [current]
        anchor = best.copy()
        for value in values:
            candidate = dict(anchor, **{key: value})
            label = f'{stage}_{value}'
            if key == 'cache_slots' and value == 8:
                candidate['memory_mb'] = 1664
                label += '_pool1664'
            candidates.append(run(label, candidate, args.count))
        eligible = [x for x in candidates if x['rss_peak_kib'] < 1800 * 1024
                    and x['max_ms'] <= current['max_ms'] * 1.02]
        winner = min(eligible, key=lambda x: x['mean_ms']) if eligible else current
        if winner['mean_ms'] < current['mean_ms'] * 0.98:
            current = winner
            best = winner['parameters'].copy()
        manifest['choices'].append(dict(stage=stage, selected=current['label'], parameters=best.copy()))
        save()
        print(f'selected={stage}:{current["label"]}', flush=True)
    # Recheck the reference configuration late, then the selected configuration.
    run('confirm_baseline', baseline['parameters'], 64)
    run('confirm_selected', best, 64)
    manifest['selected_parameters'] = best
    save()
    print('completed_all=true', flush=True)


if __name__ == '__main__':
    main()
