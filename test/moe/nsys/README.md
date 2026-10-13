# MoE Nsight Systems capture

This profiles the existing 2048 x 7168, 256-expert, Top-8 target through the original
NVMe-oF TCP benchmark. It reimports the packed image named by the supplied AIO
configuration. Use the existing disposable MoE test image. The source weights are
read only. Each mode starts with an empty expert cache.

NVTX is opt-in at build time. Supply the directory containing `nvtx3/nvToolsExt.h`:

```sh
make -C lib/moe_ffn -j4 -W moe_workspace.c \
  MOE_NVTX_INCLUDE=/usr/local/cuda-12.8/targets/x86_64-linux/include
make -C module/bdev/moe -j4 -W bdev_moe.c -W moe_store.c \
  MOE_NVTX_INCLUDE=/usr/local/cuda-12.8/targets/x86_64-linux/include
make -C examples/moe/tgt -j4
make -C test/moe/accuracy_initiator -j4
python3 test/moe/nsys/capture.py \
  --nsys /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys \
  --config /root/moe-run-logs/parameter_tuning_20260914/recommended_aio_target.json \
  --output /root/moe-run-logs/nsys_capture --runtime 30
```

The capture uses reactor CPU 10, benchmark CPU 0 and the configuration's compute
CPU. Verify this topology on another machine. Both runs preserve the supplied
cache, I/O and kernel settings and change only `pipeline`. The target is shut down
through RPC after each benchmark and independent accuracy check.

Open `matrix.nsys-rep` or `expert.nsys-rep` with Nsight Systems 2025.1.1 or newer.
Expand the target process's NVTX tracks and find `REQUEST`, `COMPUTE` and
`WEIGHT_IO`. The range labels include the actual expert ID and matrix stage:
0 = Gate, 1 = Up, 2 = Down, -1 = whole expert. Whole-expert fast paths and cache
hits are included in COMPUTE. WEIGHT_IO records each successful bdev chunk
submission through entry to its completion callback; its payload is the chunk byte count.
The range ends before decoding. `DECODE` covers validation and restoration into the FP32 cache.
An expert-mode chunk can span a matrix boundary; its stage label describes its
starting offset. The request range covers target processing, not the complete
TCP round trip.

The profiler records NVTX and OS runtime calls. CPU sampling/context-switch
collection is disabled because this container disallows perf events. SPDK AIO
uses direct I/O on an NVMe-backed filesystem. WEIGHT_IO measures software-observed
in-flight I/O, including device/queue/completion delays, not PCIe wire occupancy.

Export SQLite and calculate overlap:

```sh
nsys export --type sqlite --output /path/to/capture/expert.sqlite /path/to/capture/expert.nsys-rep
nsys export --type sqlite --output /path/to/capture/matrix.sqlite /path/to/capture/matrix.nsys-rep
python3 test/moe/nsys/analyze.py /path/to/capture
```

Analysis uses unions of intervals, so overlapping I/O chunks are not counted
multiple times. It verifies completed I/O bytes against SPDK diagnostics and
request count against benchmark output. The original benchmark performs one
unmeasured request even with `--warmup=0`; analysis excludes this first request
and the final independent accuracy request. `compute_overlap_percent` divides
overlap by compute time, not request or I/O time. Profiled timings are diagnostic
samples, not an uninstrumented performance qualification.

To return to a build without NVTX, first rebuild `lib/moe_ffn` with `-W moe_workspace.c`
without `MOE_NVTX_INCLUDE`, then rebuild `module/bdev/moe` with
`-W bdev_moe.c -W moe_store.c` and without `MOE_NVTX_INCLUDE`, then relink target
and tests. Make does not detect changes to command-line compilation flags automatically.

## Reversible encoding profile

```sh
python3 test/moe/nsys/capture_reversible.py \
  --nsys /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys \
  --config /path/to/authorized_aio_target.json --output /path/to/new_capture
python3 test/moe/nsys/analyze_reversible.py /path/to/new_capture
python3 test/moe/nsys/validate_reversible.py /path/to/new_capture
```

This capture enables eight slots, matrix FIFO, 1 MiB reversible coding and
diagnostics. It preserves the compute kernel and thread settings in the source
configuration. It reimports its authorized scratch image, warms up 16 requests
with seed 777, captures exactly 50 requests with seed 42, then checks accuracy
independently with seed 4242. The GUI report contains only the 50 timed requests:
`PROFILE_WINDOW` starts at request 17 and ends at request 66. Import and warmup
are outside the report. Sampling/context switches and OS runtime tracing are disabled; no new
dependencies are installed. SQLite export runs automatically after target shutdown.

The analyzer validates I/O and decode bytes against diagnostics, separates warmup
and accuracy requests, and measures the union of decode and compute intervals.
It reports their mutual overlap and coverage by outstanding I/O separately.
`lattice_timeline.png` and `.svg` show median and P99 requests; plotting uses
an existing Matplotlib installation. Open `moe_cpu_experts_io_decode.nsys-rep` for individual I/O,
expert, matrix and decoder ranges. An I/O range represents software observed
submission to callback, and may include completion handling delays. It does not
prove SSD DMA was physically active throughout that interval.

## GUI reading guide

The current workload executes on CPUs. It has no CUDA kernels, H2D copies or
GPU waiting intervals. `Normalize` here is routing-weight Softmax; the workload
does not contain input RMSNorm or LayerNorm.

1. Expand the target process and the thread named
   **CPU Compute / Router + Experts + Aggregate**, then expand its **NVTX** row.
   Actual CPU stages use thread-local push/pop ranges and nested leaf stages.
2. Expand **CPU IO + Decode / SPDK owner**, then its **NVTX** row. Orange ranges
   are `Expert ID / Gate|Up|Down / CPU Decode + checksum + FP32 writeback`.
   Dark blue ranges are the CPU's `SPDK read` submission, not the device transfer.
3. Expand the process-level NVTX/Annotations row for async **Request**,
   blue **I/O outstanding** and red **Pipeline wait** ranges. Their durations
   span submission/completion or coordinator dependency, not CPU core occupancy.
4. Select **Request 17 / CPU MoE Forward** and zoom to that request. Router,
   Top-K and Normalize are short; zoom into the beginning of the request to read them.
   Further zoom into a few milliseconds to inspect 1 MiB block decoders.
5. Right-click a row and use **Show in Events View** to read full labels and
   durations. The searchable names include the real expert ID and matrix name.

Green expert leaf stages are Gate, Up, SiLU(Gate)*Up and Down; routing is purple;
aggregation is pink. Red **CPU worker condvar wait** is the actual condition-variable
wait for owner dispatch, which can include matrix readiness and dispatch delay.
Red **Pipeline wait** specifically records the coordinator observing no active
compute while matrix reads/decodes remain pending. Neither is a GPU interval.
The report preserves all block events; four in-flight I/O chunks may overlap.
Collection start/stop can delay the first/last TCP responses. This capture is a
pipeline diagnosis; its TCP P99 must not replace the uninstrumented benchmark.

Nsight's `Show in Events View` operation is documented in the
[official GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#timeline-events-correlation).
