#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check real target defaults and worker affinity on the current Linux topology."""
import array
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


def core(cpu):
    base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return tuple((base / name).read_text().strip() for name in ('physical_package_id', 'core_id'))


def main():
    root = Path(__file__).resolve().parents[2]
    output = Path(tempfile.mkdtemp(prefix='moe_affinity_'))
    print(f'artifacts={output}', flush=True)
    cpus = []
    for cpu in sorted(os.sched_getaffinity(0)):
        if core(cpu) == core(0) or any(core(cpu) == core(c) for c in cpus):
            continue
        if cpus and (core(cpu)[0] != core(cpus[0])[0] or
                     [x.name for x in Path(f'/sys/devices/system/cpu/cpu{cpu}').glob('node*')] !=
                     [x.name for x in Path(f'/sys/devices/system/cpu/cpu{cpus[0]}').glob('node*')]):
            continue
        cpus.append(cpu)
        if len(cpus) == 5:
            break
    if len(cpus) < 5:
        raise RuntimeError('test requires five allowed physical cores on one NUMA node')
    weights = output / 'weights'
    weights.mkdir()
    for name, n in [('W_router_128x9.bin', 128 * 9)] + [
        (f'W_{kind}_{expert}_128x256.bin', 128 * 256)
        for expert in range(9) for kind in ('gate', 'up', 'down')]:
        (weights / name).write_bytes(array.array('f', [0.001] * n).tobytes())
    cases = [('default', cpus, 4, [], None, True),
             ('restricted', cpus[1:3], 1, [], None, True),
             ('reserved_cpu', cpus, 1, [], None, True),
             ('explicit', cpus, 1, ['-m', hex(1 << cpus[-1])], cpus[0], True),
             ('one_core', cpus[:1], 1, [], None, False),
             ('reactor_outside', cpus[1:3], 1, ['-m', hex(1 << cpus[0])], None, False),
             ('outside_allowed', cpus[1:3], 1, [], cpus[0], False)]
    for label, allowed, threads, options, worker, success in cases:
        reserved = cpus[0] if label == 'reserved_cpu' else 0
        params = dict(name='moe_test', backend='file', weight_dir=str(weights),
                      d_model=128, d_ff=256, num_experts=9, top_k=8, cache_slots=1,
                      compute_threads=threads)
        # Multiple workers require the AIO backend.
        image = output / f'{label}.bin'
        with image.open('wb') as stream:
            stream.truncate(11 * 1024**2)
        params.update(backend='aio', base_bdev='weights')
        if worker is not None:
            params['compute_cpu'] = worker
        config = dict(subsystems=[dict(subsystem='bdev', config=[
            dict(method='bdev_set_options', params=dict(bdev_io_pool_size=4096, bdev_io_cache_size=128)),
            dict(method='bdev_aio_create', params=dict(name='weights', filename=str(image), block_size=512)),
            dict(method='bdev_moe_create', params=params)])])
        path = output / f'{label}.json'
        path.write_text(json.dumps(config))
        sock = str(output / f'{label}.sock')
        with (output / f'{label}.log').open('w') as log:
            process = subprocess.Popen(['taskset', '-c', ','.join(map(str, allowed)),
                                        str(root / 'build/examples/moe_tgt'), '--no-huge', '--no-pci',
                                        '-s', '512', '-c', str(path), '-r', sock, *options],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, MOE_INITIATOR_CPU=str(reserved)))
            try:
                ready = False
                deadline = time.monotonic() + 30
                while process.poll() is None and time.monotonic() < deadline:
                    reply = subprocess.run(['python3', str(root / 'scripts/rpc.py'), '-s', sock,
                                            '-t', '1', 'bdev_get_bdevs'], stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL)
                    if reply.returncode == 0 and any(x['name'] == 'moe_test' for x in json.loads(reply.stdout)):
                        ready = True
                        break
                    time.sleep(.1)
                assert ready == success, label
                if success:
                    text = (output / f'{label}.log').read_text()
                    matches = re.findall(r'worker=(\d+) compute_cpu=(\d+) reactor_cpu=(\d+) '
                                         r'worker_node=(\d+) reactor_node=(\d+)', text)
                    assert len(matches) == threads, (label, text)
                    chosen = [int(m[1]) for m in matches]
                    reactor = int(matches[0][2])
                    assert all(c in allowed and core(c) != core(reserved) for c in chosen + [reactor])
                    assert len({core(c) for c in chosen + [reactor]}) == threads + 1
                    assert all(m[3] == m[4] for m in matches)
                    if options:
                        assert reactor == cpus[-1] and chosen == [worker]
                    assert os.sched_getaffinity(process.pid) == {reactor}
                    affinities = [os.sched_getaffinity(int(t.name))
                                  for t in Path(f'/proc/{process.pid}/task').iterdir()]
                    assert all({c} in affinities for c in chosen)
                else:
                    assert process.poll() is not None and process.returncode != 0
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                process.wait(timeout=30)
            if success:
                assert process.returncode == 0
        print(f'passed={label}', flush=True)


if __name__ == '__main__':
    main()
