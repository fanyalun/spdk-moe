#!/usr/bin/env python3
"""Foreground scoring entry point. Only a directly attached SPDK NVMe is accepted."""
import argparse
import json
import mmap
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SPDK = ROOT
NQN = 'nqn.2024-07.io.spdk:moe_target'
LAYOUT_BYTES = 45100302336
PCI_ROOT = Path('/sys/bus/pci/devices')
MOE_OPTIONS = dict(name='moe_bdev_0', backend='nvme', kernel='avx2', pipeline='matrix',
                   d_model=2048, d_ff=7168, num_experts=256, top_k=8, cache_slots=7,
                   io_size=1048576, io_depth=4, prefetch=2, compute_threads=1)


def target_command(binary, rpc_path, bdf, huge_directory):
    return [str(binary), '--wait-for-rpc', '-r', str(rpc_path),
            '-A', bdf, '--huge-dir', str(huge_directory)]


def target_environment():
    return dict(os.environ)


def check_weights(directory):
    files = [('W_router_2048x256.bin', 2048 * 256 * 4)]
    files += [(f'{kind}_{expert}_2048x7168.bin', 2048 * 7168 * 4)
              for expert in range(256) for kind in ('W_gate', 'W_up', 'W_down')]
    for name, size in files:
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f'Invalid weight file: {path}; expected {size} bytes')
        # Require aligned direct reads for every source, before any NVMe writes.
        # This prevents entering the legacy buffered-import fallback in scoring.
        with mmap.mmap(-1, 4096) as buffer:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                if os.readv(fd, [buffer]) != 4096:
                    raise RuntimeError(f'Direct source read failed: {path}')
            finally:
                os.close(fd)


def discover_device(explicit=None):
    if explicit:
        return explicit.lower()
    candidates = []
    for device in sorted(PCI_ROOT.iterdir()):
        if int((device / 'class').read_text().strip(), 16) != 0x010802:
            continue
        driver = (device / 'driver').resolve().name if (device / 'driver').exists() else ''
        if driver in ('', 'vfio-pci', 'uio_pci_generic'):
            candidates.append(device.name)
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one released NVMe, found {candidates}; specify --nvme BDF if ambiguous. '
                           'The organizer must unbind the selected disk from the kernel first.')
    return candidates[0]


def choose_driver(bdf, requested):
    if requested != 'auto':
        return requested
    device = PCI_ROOT / bdf
    if (device / 'driver').exists():
        current = (device / 'driver').resolve().name
        if current in ('vfio-pci', 'uio_pci_generic'):
            return current
    return 'vfio-pci' if (device / 'iommu_group').exists() else 'uio_pci_generic'


def check_device(bdf, preparing=False, driver_name='vfio-pci'):
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf):
        raise RuntimeError('Use a full PCI address, e.g. 0000:01:00.0')
    device = PCI_ROOT / bdf
    if not device.exists() or int((device / 'class').read_text().strip(), 16) != 0x010802:
        raise RuntimeError(f'{bdf} is not an NVMe PCI controller')
    if driver_name == 'vfio-pci':
        if not (device / 'iommu_group').exists():
            raise RuntimeError('vfio-pci requires IOMMU; enable IOMMU or explicitly select --driver uio_pci_generic')
        group = sorted(p.name for p in (device / 'iommu_group/devices').iterdir())
        if group != [bdf]:
            raise RuntimeError(f'Refusing a shared IOMMU group: {group}')
    driver = (device / 'driver').resolve().name if (device / 'driver').exists() else ''
    if preparing:
        mounted = {line.split()[2] for line in Path('/proc/self/mountinfo').read_text().splitlines()}
        swaps = set()
        for line in Path('/proc/swaps').read_text().splitlines()[1:]:
            path = Path(line.split()[0])
            if path.exists():
                dev = path.stat().st_rdev
                swaps.add(f'{os.major(dev)}:{os.minor(dev)}')
        for block in Path('/sys/class/block').iterdir():
            if device.resolve() not in block.resolve().parents:
                continue
            number = (block / 'dev').read_text().strip()
            if (number in mounted or number in swaps or (block / 'partition').exists()
                    or any((block / 'holders').iterdir())):
                raise RuntimeError(f'Refusing in-use or partitioned device: {block.name}')
        if driver not in ('', 'vfio-pci', 'uio_pci_generic'):
            raise RuntimeError(f'Unsupported existing driver: {driver}')
    elif driver != driver_name:
        raise RuntimeError(f'{bdf} uses {driver or "no driver"}; run --prepare on the dedicated disk first')
    return device


def select_node(device):
    node = int((device / 'numa_node').read_text())
    allowed = os.sched_getaffinity(0)
    reserved = int(os.environ.get('MOE_INITIATOR_CPU', '0'))

    def core(cpu):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return ((topology / 'physical_package_id').read_text().strip(),
                (topology / 'core_id').read_text().strip())

    reserved_core = core(reserved) if Path(f'/sys/devices/system/cpu/cpu{reserved}').exists() else None
    local = {cpu for cpu in allowed if Path(f'/sys/devices/system/node/node{node}/cpu{cpu}').exists()}
    if node >= 0 and len({core(cpu) for cpu in local if core(cpu) != reserved_core}) >= 2:
        os.sched_setaffinity(0, local)
        print(f'AFFINITY: prefer NVMe NUMA node {node}; allowed CPUs={sorted(local)}', flush=True)
        return node
    print('AFFINITY: using assigned cpuset; automatic reactor/worker physical-core separation', flush=True)
    return None


def check_huge_directory(path):
    resolved = str(path.resolve())
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        mounted = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), before.split()[4])
        if mounted == resolved and after.split()[0] == 'hugetlbfs':
            if os.statvfs(path).f_bsize == 2 * 1024 * 1024:
                return
            raise RuntimeError(f'{path} must use 2MiB hugepages, not 1GiB pages')
    raise RuntimeError(f'{path} must be a 2MiB hugetlbfs mount; run --prepare first')


class Rpc:
    def __init__(self, path, timeout):
        self.path, self.timeout, self.sequence = str(path), timeout, 0

    def call(self, method, **params):
        self.sequence += 1
        request = dict(jsonrpc='2.0', id=self.sequence, method=method)
        if params:
            request['params'] = params
        timeout = self.timeout if method in ('bdev_moe_create', 'bdev_moe_trim') else min(60, self.timeout)
        deadline = time.monotonic() + timeout
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(timeout)
            connection.connect(self.path)
            connection.sendall((json.dumps(request) + '\n').encode())
            data = b''
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'RPC timeout: {method}')
                connection.settimeout(remaining)
                chunk = connection.recv(65536)
                if not chunk:
                    raise RuntimeError(f'RPC connection closed: {method}')
                data += chunk
                if len(data) > 16 * 1024 * 1024:
                    raise RuntimeError('RPC response exceeds limit')
                try:
                    response = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if response.get('id') != self.sequence or 'error' in response:
                    raise RuntimeError(f'{method}: {response}')
                return response['result']


def publish(rpc, bdf, nsid, weights, cache_slots=None):
    rpc.call('bdev_set_options', bdev_io_pool_size=4096, bdev_io_cache_size=128)
    rpc.call('framework_start_init')
    namespaces = rpc.call('bdev_nvme_attach_controller', name='ScoringNvme', trtype='PCIe', traddr=bdf)
    if nsid is None and len(namespaces) != 1:
        raise RuntimeError(f'Expected one namespace, found {namespaces}; specify --nsid')
    base = namespaces[0] if nsid is None else f'ScoringNvmen{nsid}'
    if base not in namespaces:
        raise RuntimeError(f'NVMe namespace {nsid} missing; attached bdevs={namespaces}')
    bdevs = rpc.call('bdev_get_bdevs', name=base)
    if len(bdevs) != 1:
        raise RuntimeError('Expected exactly one selected namespace')
    bdev = bdevs[0]
    if bdev.get('claimed') or bdev['block_size'] * bdev['num_blocks'] < LAYOUT_BYTES:
        raise RuntimeError(f'NVMe namespace is claimed or smaller than {LAYOUT_BYTES} bytes')
    if bdev.get('md_size', 0) or bdev['block_size'] not in (512, 4096):
        raise RuntimeError('Use a namespace formatted with 512/4096 byte data blocks and no metadata')
    if bdev.get('supported_io_types', {}).get('unmap', False):
        print(f'TRIM: selected namespace {base}, {bdev["block_size"] * bdev["num_blocks"]} bytes', flush=True)
        rpc.call('bdev_moe_trim', name=base)
    else:
        print(f'TRIM: {base} does not support UNMAP; continuing without the recommended TRIM', flush=True)
    print(f'IMPORTING: {base}, {LAYOUT_BYTES} bytes; listener remains closed', flush=True)
    options = dict(MOE_OPTIONS)
    if cache_slots is not None:
        options['cache_slots'] = cache_slots
    rpc.call('bdev_moe_create', **options, base_bdev=base, weight_dir=str(weights))
    rpc.call('nvmf_create_transport', trtype='TCP', io_unit_size=16384, max_io_size=131072,
             max_io_qpairs_per_ctrlr=128, in_capsule_data_size=16384, max_aq_depth=32)
    rpc.call('nvmf_create_subsystem', nqn=NQN, allow_any_host=True,
             serial_number='MOE00000000000001', model_number='MoE FFN Target', max_namespaces=1)
    rpc.call('nvmf_subsystem_add_ns', nqn=NQN, namespace=dict(nsid=1, bdev_name='moe_bdev_0'))
    rpc.call('nvmf_subsystem_add_listener', nqn=NQN,
             listen_address=dict(trtype='TCP', adrfam='IPv4', traddr='127.0.0.1', trsvcid='4420'))


def prepare(bdf, node, driver, huge_directory):
    if os.geteuid() != 0:
        raise RuntimeError('--prepare requires root')
    if not (SPDK / 'scripts/setup.sh').is_file():
        raise RuntimeError('Build SPDK before device preparation')
    if not Path('/sys/kernel/mm/hugepages/hugepages-2048kB').is_dir():
        raise RuntimeError('This release requires 2MiB hugepages')
    huge_directory.mkdir(parents=True, exist_ok=True)
    if os.path.ismount(huge_directory):
        check_huge_directory(huge_directory)
    else:
        if any(huge_directory.iterdir()):
            raise RuntimeError(f'Refusing to mount over a nonempty directory: {huge_directory}')
        subprocess.run(['mount', '-t', 'hugetlbfs', '-o', 'pagesize=2M', 'nodev', str(huge_directory)], check=True)
        check_huge_directory(huge_directory)
    env = dict(os.environ)
    for key in ('PCI_BLOCKED', 'NVME_ALLOWED', 'NVME_BLOCKED', 'HUGEMEM', 'NRHUGE', 'HUGENODE',
                'CLEAR_HUGE', 'SHRINK_HUGE', 'SKIP_HUGE', 'SKIP_PCI', 'TARGET_USER',
                'UNBIND_ENTIRE_IOMMU_GROUP'):
        env.pop(key, None)
    env.update(PCI_ALLOWED=bdf, DRIVER_OVERRIDE=driver, HUGEPGSZ='2048')
    # A node-specific sysfs path selects 2MiB even when the host default is 1GiB.
    env['HUGENODE'] = str(node if node is not None else 0)
    subprocess.run(['bash', str(SPDK / 'scripts/setup.sh')], env=env, check=True)
    check_device(bdf, driver_name=driver)
    print(f'PREPARED: {driver} and hugepages; start target inside the official scoring cgroup', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvme', help='Full PCI BDF; default: discover the unique released NVMe')
    parser.add_argument('--driver', choices=('auto', 'vfio-pci', 'uio_pci_generic'), default='auto',
                        help='Explicit SPDK PCI driver; UIO is available for a host without IOMMU')
    parser.add_argument('--nsid', type=int, help='Physical namespace ID; default: unique namespace (exported NSID is always 1)')
    parser.add_argument('--weights', type=Path, default=Path('/tmp/moe_weights'))
    parser.add_argument('--cache-slots', type=int, choices=range(1, 8), default=MOE_OPTIONS['cache_slots'],
                        help='Resident expert slots; default: 7')
    parser.add_argument('--huge-dir', type=Path, default=Path('/mnt/moe_huge'), help='Dedicated 2MiB hugetlbfs mount')
    parser.add_argument('--allow-overwrite', action='store_true', help='Required: authorize replacing data on this namespace')
    parser.add_argument('--prepare', action='store_true', help='Bind only the selected controller and prepare 2MiB hugepages; then exit')
    parser.add_argument('--dry-run', action='store_true', help='Print scoring configuration without touching devices')
    parser.add_argument('--startup-timeout', type=int, default=7200, help='Import/RPC timeout in seconds')
    args = parser.parse_args()
    if (args.nsid is not None and args.nsid < 1) or args.startup_timeout < 1:
        parser.error('nsid and startup-timeout must be positive')
    if args.dry_run:
        print(json.dumps(dict(nvme=args.nvme, driver=args.driver, nsid=args.nsid, weights=str(args.weights),
                              huge_dir=str(args.huge_dir), memory_mode='dynamic',
                              layout_bytes=LAYOUT_BYTES, moe=dict(MOE_OPTIONS, cache_slots=args.cache_slots),
                              nqn=NQN, address='127.0.0.1:4420'), indent=2))
        return 0
    if not args.allow_overwrite:
        parser.error('--allow-overwrite is required for the explicitly selected scratch NVMe')
    args.nvme = discover_device(args.nvme)
    args.driver = choose_driver(args.nvme, args.driver)
    device = check_device(args.nvme, preparing=True, driver_name=args.driver)
    print(f'DEVICE: {args.nvme}, driver={args.driver}', flush=True)
    node = select_node(device)
    if not args.prepare:
        check_weights(args.weights)
        if not (SPDK / 'build/examples/moe_tgt').is_file():
            raise RuntimeError('Build examples/moe/tgt first')
    if args.prepare:
        prepare(args.nvme, node, args.driver, args.huge_dir.resolve())
        return 0
    try:
        check_device(args.nvme, driver_name=args.driver)
        check_huge_directory(args.huge_dir)
    except RuntimeError:
        prepare(args.nvme, node, args.driver, args.huge_dir.resolve())
    binary = SPDK / 'build/examples/moe_tgt'
    if not binary.is_file():
        raise RuntimeError('Build examples/moe/tgt first')
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 4420))
    # A unique RPC directory avoids removing another running target's socket.
    with tempfile.TemporaryDirectory(prefix='moe_scoring_') as directory:
        rpc_path = Path(directory) / 'rpc.sock'
        command = target_command(binary, rpc_path, args.nvme, args.huge_dir.resolve())
        process = subprocess.Popen(command, env=target_environment())

        def stop(signum, frame):
            raise InterruptedError(f'Received signal {signum}')

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            deadline = time.monotonic() + 60
            while not rpc_path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f'Target exited during startup: {process.returncode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Target RPC socket startup timed out')
                time.sleep(0.1)
            publish(Rpc(rpc_path, args.startup_timeout), args.nvme, args.nsid, args.weights.resolve(),
                    args.cache_slots)
            print(f'READY: {NQN} 127.0.0.1:4420 nsid=1 target_pid={process.pid}', flush=True)
            return process.wait()
        finally:
            # Drain the target on failure/signal; never leave a background scoring service.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            if process.poll() is None:
                process.terminate()
                process.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except InterruptedError as error:
        print(f'STOPPED: {error}', file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f'FAILED: {error}', file=sys.stderr)
        sys.exit(1)
