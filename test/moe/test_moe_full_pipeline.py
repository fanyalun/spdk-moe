#!/usr/bin/env python3
#  SPDX-License-Identifier: BSD-3-Clause
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    if len(sys.argv) != 3:
        print('usage: test_moe_full_pipeline.py WEIGHT_DIRECTORY OUTPUT_FILESYSTEM_DIRECTORY')
        return 2
    weights = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    mib = 1024 * 1024
    required = (3 + 256 * 168) * mib
    available = shutil.disk_usage(destination).free
    if available < required + 10 * 1024 ** 3:
        raise RuntimeError(f'need {required} bytes plus 10GiB reserve; available={available}')
    artifacts = Path(tempfile.mkdtemp(prefix='moe_full_', dir=destination))
    image = artifacts / 'packed.bin'
    with image.open('xb') as stream:
        stream.truncate(required)
    config = dict(subsystems=[dict(subsystem='bdev', config=[
        dict(method='bdev_set_options', params=dict(bdev_io_pool_size=4096, bdev_io_cache_size=128)),
        dict(method='bdev_aio_create', params=dict(name='weight_aio', filename=str(image), block_size=512)),
        dict(method='bdev_moe_create', params=dict(name='moe_test', backend='aio', base_bdev='weight_aio',
             weight_dir=str(weights), cache_slots=7, compute_cpu=1,
             diagnostics=str(artifacts / 'diagnostics.jsonl')))
    ])])
    config_path = artifacts / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    root = Path(__file__).resolve().parents[2]
    binary = root / 'test/moe/full_pipeline/moe_full_pipeline_test'
    print(f'artifacts={artifacts} image_bytes={required} free_before={available}', flush=True)
    started = time.monotonic()
    with (artifacts / 'test.log').open('w') as log, (artifacts / 'memory.jsonl').open('w') as memory:
        process = subprocess.Popen([str(binary), '-c', str(config_path), '-m', '0x1',
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
