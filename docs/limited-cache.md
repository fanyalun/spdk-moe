# Limited Expert Cache Prototype

The target keeps the router weights resident and loads expert weights from
`MOE_WEIGHT_DIR` into a fixed-size LRU cache. The prototype cache is exposed
through `module/bdev/moe/moe_cache.[ch]`; it is currently an isolated loading
component and is not yet wired into request execution.

Build from the Linux SPDK checkout with:

```bash
make -j1
make -C examples/moe -j1
```

The next integration step is to replace the fixed expert arrays in
`moe_subsystem.c` and `bdev_moe.c` with a cache handle, then split routing from
expert execution so only selected experts are fetched. Do not use this
prototype as a formal 2 GiB baseline until that integration and correctness
testing are complete.
