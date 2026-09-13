#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause
import argparse
import struct
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--pipeline', choices=['expert', 'matrix'], default='expert')
    parser.add_argument('--cache-slots', type=int, choices=range(1, 9), default=7)
    parser.add_argument('--mem-size', type=int, default=1536)
    parser.add_argument('--reuse-image', type=Path,
                        help='Explicitly authorize reimport into this existing MoE image')
    args = parser.parse_args()
    weights = args.weights.resolve()
    destination = args.destination.resolve()
    mib = 1024 * 1024
    required = (3 + 256 * 168) * mib
    available = min(shutil.disk_usage(args.reuse_image or destination).free,
                    int(subprocess.check_output(['df', '-B1', '--output=avail',
                                                 str(args.reuse_image or destination)],
                                                text=True).splitlines()[-1]))
    limit_text = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    estimated = (168 * args.cache_slots + 129 + 168 + 32) * mib
    if limit_text != 'max' and current + estimated > int(limit_text):
        raise RuntimeError(f'shared cgroup lacks headroom: current={current}, '
                           f'estimated_test={estimated}, limit={limit_text}')
    needed = 0 if args.reuse_image else required
    if available < needed + 10 * 1024 ** 3:
        raise RuntimeError(f'need {needed} bytes plus 10GiB reserve; available={available}')
    artifacts = Path(tempfile.mkdtemp(prefix='moe_full_', dir=destination))
    if args.reuse_image:
        image = args.reuse_image.resolve()
        with image.open('rb') as stream:
            header = struct.unpack('<Q6I7Q', stream.read(88))
        if image.stat().st_size != required or header[0] != 0x31454f4d4b445053 or header[1] != 1:
            raise RuntimeError('reuse requires an existing full-size version-1 MoE image')
    else:
        image = artifacts / 'packed.bin'
        with image.open('xb') as stream:
            stream.truncate(required)
    config = dict(subsystems=[dict(subsystem='bdev', config=[
        dict(method='bdev_set_options', params=dict(bdev_io_pool_size=4096, bdev_io_cache_size=128)),
        dict(method='bdev_aio_create', params=dict(name='weight_aio', filename=str(image), block_size=512)),
        dict(method='bdev_moe_create', params=dict(name='moe_test', backend='aio', base_bdev='weight_aio',
             weight_dir=str(weights), cache_slots=args.cache_slots, compute_cpu=1, pipeline=args.pipeline,
             diagnostics=str(artifacts / 'diagnostics.jsonl')))
    ])])
    config_path = artifacts / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    root = Path(__file__).resolve().parents[2]
    binary = root / 'test/moe/full_pipeline/moe_full_pipeline_test'
    print(f'artifacts={artifacts} image_bytes={required} free_before={available}', flush=True)
    started = time.monotonic()
    with (artifacts / 'test.log').open('w') as log, (artifacts / 'memory.jsonl').open('w') as memory:
        process = subprocess.Popen([str(binary), '-c', str(config_path), '-m', '0x1', '-s', str(args.mem_size),
                                    '-r', str(artifacts / 'rpc.sock')],
                                   env=dict(os.environ, MOE_TEST_WEIGHTS=str(weights)),
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
            except FileNotFoundError:
                break
            sample = {line.split(':')[0]: line.split(':')[1].strip()
                      for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:'))}
            sample['elapsed_seconds'] = time.monotonic() - started
            memory.write(json.dumps(sample) + '\n')
            memory.flush()
            time.sleep(0.25)
        rc = process.wait()
    print(f'exit={rc} elapsed_seconds={time.monotonic() - started:.3f}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
