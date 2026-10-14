#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check submission defaults and readiness ordering without accessing an NVMe device."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('run_target', ROOT / 'run_target.py')
TARGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGET)


class FakeRpc:
    def __init__(self, completed=16, slots=8, encoding='lattice'):
        self.calls = []
        self.info = dict(warmup_completed=completed, cache_slots=slots,
                         cache_memory='cpu', warmup_retries=int(slots == 7),
                         compute_threads=1, decode_threads=1,
                         weight_encoding=encoding, pipeline='matrix_fifo' if slots == 8 else 'matrix')

    def call(self, method, **params):
        self.calls.append((method, params))
        if method == 'bdev_nvme_attach_controller':
            return ['ScoringNvmen1']
        if method == 'bdev_get_bdevs':
            if params['name'] == 'moe_bdev_0':
                return [dict(driver_specific=dict(moe=self.info))]
            return [dict(block_size=512, num_blocks=TARGET.LAYOUT_BYTES // 512,
                         supported_io_types=dict(unmap=False))]
        return True


class LauncherTest(unittest.TestCase):
    def test_defaults_and_command(self):
        options = TARGET.MOE_OPTIONS
        self.assertEqual(options['pipeline'], 'matrix_fifo')
        self.assertEqual(options['weight_encoding'], 'lattice')
        self.assertEqual(options['warmup_requests'], 16)
        self.assertEqual(options['io_depth'], 16)
        self.assertEqual(options['compute_threads'], 1)
        self.assertEqual(options['decode_threads'], 1)
        self.assertNotIn('prefetch', options)
        command = TARGET.target_command('moe_tgt', '/tmp/rpc', '0000:01:00.0', '/mnt/huge')
        self.assertIn('-A', command)
        self.assertNotIn('--no-pci', command)
        self.assertNotIn('--no-huge', command)
        self.assertEqual(command[-2:], ['-s', '256'])
        self.assertEqual(TARGET.HUGE_PAGES, 128)

    def test_publish_after_warmup_and_fallback(self):
        for actual_slots in (8, 7):
            rpc = FakeRpc(slots=actual_slots)
            info = TARGET.publish(rpc, '0000:01:00.0', None, Path('/tmp/weights'))
            self.assertEqual(info['cache_slots'], actual_slots)
            methods = [method for method, _ in rpc.calls]
            self.assertLess(methods.index('iobuf_set_options'), methods.index('framework_start_init'))
            self.assertLess(methods.index('bdev_moe_create'), methods.index('nvmf_create_transport'))
            self.assertEqual(methods[-1], 'nvmf_subsystem_add_listener')

    def test_invalid_warmup_or_encoding_never_listens(self):
        wrong_memory = FakeRpc()
        wrong_memory.info['cache_memory'] = 'dma'
        for rpc in (FakeRpc(completed=15), FakeRpc(encoding='raw'), wrong_memory):
            with self.assertRaises(RuntimeError):
                TARGET.publish(rpc, '0000:01:00.0', None, Path('/tmp/weights'))
            self.assertFalse(any(method.startswith('nvmf_') for method, _ in rpc.calls))

    def test_unexpected_fallback_never_listens(self):
        for field, value in [('cache_slots', 6), ('warmup_retries', 0),
                             ('pipeline', 'matrix_fifo'), ('compute_threads', 2),
                             ('decode_threads', 0)]:
            rpc = FakeRpc(slots=7)
            rpc.info[field] = value
            with self.assertRaises(RuntimeError):
                TARGET.publish(rpc, '0000:01:00.0', None, Path('/tmp/weights'))
            self.assertFalse(any(method.startswith('nvmf_') for method, _ in rpc.calls))

    def test_prepare_reserves_same_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(TARGET.os, 'geteuid', return_value=0), \
                    patch.object(Path, 'is_dir', return_value=True), \
                    patch.object(TARGET.os.path, 'ismount', return_value=True), \
                    patch.object(TARGET, 'check_huge_directory'), \
                    patch.object(TARGET, 'check_device'), \
                    patch.object(TARGET, 'check_huge_pages'), \
                    patch.object(TARGET.subprocess, 'run') as run:
                TARGET.prepare('0000:01:00.0', 0, 'vfio-pci', Path(directory))
            env = run.call_args.kwargs['env']
            self.assertEqual(env['HUGEMEM'], '256')
            self.assertEqual(env['HUGEPGSZ'], '2048')
            self.assertEqual(env['PCI_ALLOWED'], '0000:01:00.0')
            self.assertNotIn('NRHUGE', env)


if __name__ == '__main__':
    unittest.main()
