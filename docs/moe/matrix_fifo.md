# Matrix FIFO and lossless compression research

`pipeline=matrix_fifo` reserves every selected cache miss before starting reads. It
requires `cache_slots >= top_k` (eight slots for the official Top-8 workload).
Packed backends with enough slots now default to FIFO; see
[reversible_weights.md](reversible_weights.md) for encoding defaults. `prefetch`
does not limit FIFO admission; `bdev_get_bdevs` exposes `prefetch_ignored=true`.

## Submission and ownership

The owner thread constructs up to 24 matrix descriptors in selected Top-K order:
Gate, Up, Down for each miss. Each descriptor points directly into its reserved
expert cache entry. The store fills all free global I/O operation slots from the
first descriptor with unassigned chunks. Once every chunk has an operation slot,
it advances to the next matrix without waiting for completion of the current one.
It does not distribute chunks round-robin among experts. NVMe/bdev may split these
logical reads further according to the device's transfer limit.

Per-matrix completion bits advance a contiguous ready prefix. This allows Up or
Down to complete first without making Gate ready. Computation and final Top-K
aggregation retain their existing order and arithmetic. A slot becomes a valid
cache entry only after all three matrices are read successfully. A read error
stops the entire batch's new submissions. Already submitted DMA and ENOMEM wait
callbacks drain before request buffers are released. Partially loaded entries are
invalidated. There is no per-request weight allocation or full-matrix copy.

The eight FP32 expert buffers occupy 1344 MiB at the official dimensions. Pool
reservation, workspaces, stacks, dictionaries and future decompression buffers
must also fit the official 2 GB total, including hugepages. Local no-huge tests use
a 1664 MiB SPDK reservation; RSS/HWM alone does not certify the official cgroup.

## Diagnostics and validation

Optional diagnostics preserve per-matrix enqueue, first successful submission,
read completion and compute timestamps. New cumulative counters expose submitted
I/O count, resource waits, time with logical bdev I/O outstanding, the integral of
logical I/O depth, and time with pending reads but no submitted I/O. These measure
the software path, not PCIe transactions or physical NAND activity. Subtract
successive cumulative values for per-request results. Disable diagnostics when
measuring P99.

```sh
make -C module/bdev/moe -j4
make -C examples/moe/tgt -j4
make -C test/moe/pipeline -j4
make -C test/moe/accuracy_initiator -j4
cc -D_GNU_SOURCE -O1 -g -ffunction-sections -fdata-sections -Iinclude -Ilib \
  test/moe/test_moe_store_wait.c module/bdev/moe/moe_codec.c \
  -Wl,--gc-sections -lm -lpthread -o /tmp/moe_store_wait_test
/tmp/moe_store_wait_test
MOE_TEST_FIFO=1 MOE_TEST_ROUNDS=50 python3 test/moe/test_moe_pipeline.py
MOE_TEST_FIFO=1 MOE_TEST_CASE= MOE_TEST_ROUNDS=50 MOE_TEST_NO_REPEAT=1 MOE_TEST_WEIGHT_SEED=43 \
  python3 test/moe/test_moe_pipeline.py
MOE_TEST_FIFO=1 MOE_TEST_CASE= MOE_TEST_ROUNDS=50 MOE_TEST_NO_REPEAT=1 MOE_TEST_WEIGHT_SEED=44 \
  python3 test/moe/test_moe_pipeline.py
```

The store test forces reverse completion order, crosses matrix boundaries before
completion, checks submission addresses, and cancels a batch during resource
starvation. Integration tests cover cache hits, misses, eviction, multiple compute
threads, truncation, removal, shutdown, and poisoned Gate/Up/Down weights followed
by recovery. Three small weight seeds exercise different data; they are not a
substitute for three independent full-size official weight sets.

Against a running full-size target, the independent TCP initiator accepts an
optional count and input seed (defaults: 1 and 42):

```sh
MOE_INITIATOR_CPU_MASK=0x1 test/moe/accuracy_initiator/moe_accuracy_initiator 50 4242
```

Every request uses the vendor command followed by READ and a separately linked
reference computation. Every output element must be finite and the maximum
absolute error must be strictly below `1e-5f`. Reports print nine significant
digits and the worst element. Test seeds are confined to the test harness.

## Local P99 comparison

```sh
python3 test/moe/test_moe_fifo_bench.py /path/to/authorized_aio_config.json /path/to/results
```

This command reimports into the existing scratch image named by the configuration.
It requires an explicitly disposable/authorized image. It runs old seven-slot,
old eight-slot and FIFO eight-slot configurations, then two further alternating
eight-slot comparisons. Each run uses 16 warmup and 50 measured requests; nearest
rank P99 is the maximum of those 50 samples. It checks input sequence equality,
records CPU throttling and target RSS/HWM, and runs independent accuracy checks
separately from timing (50 inputs for the first FIFO run, one for other runs).
It does not modify the official benchmark or certify official performance.

## Compression experiments

```sh
python3 test/moe/research_lossless.py /path/to/packed.bin /path/to/compression.json --verify-all
```

The script reads a complete version-1 packed image without modifying it. It uses
NumPy, Python zstandard and the system LZ4 shared library; install missing
dependencies only with approval. Default held-out experts are 0, 128 and 255.
Candidates are independent 1/4/8 MiB blocks or whole 56 MiB matrices, optional byte
or bit transpose, and LZ4 or Zstd level 1. The best Zstd candidate also tests a
shared dictionary and three stage-specific dictionaries trained on other experts.
Dictionary sampling derives only from the supplied image, never a generator seed.

Every tested block must restore byte-for-byte. Non-shrinking blocks fall back to
raw data. Accounting includes LBA alignment, a 64-byte/block index budget, a header
and dictionaries. SHA-256 identifies the tested source byte sequence. `--verify-all`
round-trips every expert with the best genuinely compressing non-dictionary
candidate. Timings include Python allocation and NumPy inverse transforms; they
are research prototype timings, not native decoder ceilings or measured SSD P99.
The reported serialized/ideal-overlap estimates use nominal 7 GB/s only as a
screening model. A compressed inference backend is gated on measured benefit,
exact restoration, official output tolerance and the 2 GB total memory limit.
