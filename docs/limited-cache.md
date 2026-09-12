# Limited Expert Cache Prototype

The target keeps the router weights resident and loads expert weights from
`MOE_WEIGHT_DIR` into a fixed-size LRU cache. The prototype cache is exposed
through `module/bdev/moe/moe_cache.[ch]` and is integrated into target requests.

Build from the Linux SPDK checkout with:

```bash
make -j1
make -C examples/moe -j1
```

## Integrated target

The target now loads only the router at initialization. It routes each request,
loads selected experts sequentially, computes each FFN, and accumulates in the
original Top-K order. Each expert can be evicted after its computation; one slot
therefore supports Top-8. The original reference calculation remains unchanged.

Run on Linux after stopping the old target, keeping both configuration headers
at 8 experts/Top-2 and `/tmp/moe_weights` pointing to the demo weights:

```bash
MOE_CACHE_SLOTS=2 build/examples/moe_tgt examples/moe/moe_tgt.json
```

The default is two slots; valid values are 1 through the configured expert count.
Each expert uses about 168 MiB; two slots use about 336 MiB of weight memory,
in addition to DPDK, router and working memory. Allocation is lazy; eviction
frees the previous expert before loading its replacement. File size, missing
file and allocation failures fail the request and invalidate its cached output.

In another Linux terminal run the initiator twice, saving separate logs:

```bash
set -o pipefail
timeout 120s build/examples/moe_initiator 2>&1 | tee /root/moe-run-logs/cache-check-1.log
printf 'exit=%s\n' "${PIPESTATUS[0]}"
timeout 120s build/examples/moe_initiator 2>&1 | tee /root/moe-run-logs/cache-check-2.log
printf 'exit=%s\n' "${PIPESTATUS[0]}"
```

Both must PASS and exit zero; the second request should hit both experts.
Restart with `MOE_CACHE_SLOTS=1` and repeat to exercise eviction with capacity
below Top-K. Record target VmRSS/VmHWM and cgroup memory before and after requests.
Artifacts are target hit/miss/eviction logs, correctness logs and memory records;
do not commit these generated logs.

## Standalone regression

Run on Mac or Linux without SPDK dependencies or production weights:

```bash
cc -g -fsanitize=address,undefined -Ilib/moe_ffn -Imodule/bdev/moe \
  module/bdev/moe/test_cache.c module/bdev/moe/moe_cache.c \
  lib/moe_ffn/{swiglu_moe,swiglu_ffn,matvec,topk,softmax,silu}.c \
  -lm -o /tmp/moe-cache-regression
/tmp/moe-cache-regression
```

Expected: `cache regression PASS`. Temporary small weights exercise 1/2/3 slots,
A-B-A inputs, hits, eviction and recovery after missing files, compared against
the original MoE calculation. Linux end-to-end validation remains necessary.

## Limits

### Linux process and memory inspection

SPDK renames its reactor thread to `reactor_0`. Consequently, `pgrep -x moe_tgt`
and `ps -C moe_tgt` can return no matches even while the target is alive.
Use executable-path matching instead, without restarting a working target:

```bash
set -o pipefail
bash examples/moe/inspect-target.sh | tee /root/moe-run-logs/cache-memory-before.log
timeout 600s build/examples/moe_initiator 2>&1 | tee /root/moe-run-logs/cache-memory-check.log
printf 'initiator exit=%s\n' "${PIPESTATUS[0]}"
bash examples/moe/inspect-target.sh | tee /root/moe-run-logs/cache-memory-after.log
```

The script reports every matching target in the current PID namespace, including
its renamed command, RSS/HWM (kB), and cgroup membership. It also reports visible
container counters (bytes). A missing match can reflect permissions or a different
container; it is not proof of an OOM kill. Container historical peaks include the
reference initiator, file cache and earlier runs, and cannot establish target-only
memory compliance. Retain the before/after logs and the initiator report.

This prototype assumes a single reactor and serialized requests. Synchronous
file reads block the reactor; per-expert logging affects timings. File page
cache counts toward cgroup usage, so reduced RSS is not proof of compliance.
This is not SPDK NVMe offload or a validated formal 2 GB baseline.
For 256 experts, both headers and weight paths must match; the reference
initiator still loads approximately 42 GiB of weights.
