#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
target_exe=$(readlink -f "$repo_dir/build/examples/moe_tgt")
found=0

for proc_dir in /proc/[0-9]*; do
	resolved_exe=$(readlink "$proc_dir/exe" 2>/dev/null || true)
	if [[ "$resolved_exe" != "$target_exe" && "$resolved_exe" != "$target_exe (deleted)" ]]; then
		continue
	fi
	found=1
	printf '\nTarget PID=%s executable=%s\n' "${proc_dir##*/}" "$resolved_exe"
	if [[ ! -r "$proc_dir/status" ]]; then
		printf 'Process exited during inspection\n'
		continue
	fi
	grep -E '^(Name|State|Pid|VmPeak|VmSize|VmRSS|VmHWM|RssAnon|RssFile|RssShmem):' "$proc_dir/status" || true
	printf 'Cgroup membership:\n'
	cat "$proc_dir/cgroup" 2>/dev/null || true
done

if [[ "$found" == 0 ]]; then
	printf 'No visible process matches %s; check container, permissions and target terminal.\n' "$target_exe" >&2
	exit 1
fi

printf '\nContainer memory counters (not target-only):\n'
for counter in /sys/fs/cgroup/memory/memory.limit_in_bytes \
	/sys/fs/cgroup/memory/memory.usage_in_bytes \
	/sys/fs/cgroup/memory/memory.max_usage_in_bytes \
	/sys/fs/cgroup/memory/memory.failcnt \
	/sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current \
	/sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events; do
	if [[ -r "$counter" ]]; then
		printf '%s:\n' "$counter"
		cat "$counter"
	fi
done
