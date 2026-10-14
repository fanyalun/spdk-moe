# 可逆权重编码

## 默认配置

`bdev_moe_create` 默认缓存槽位为 8。AIO/NVMe 后端未指定 `pipeline`
且槽位不少于 Top-K 时，使用 `matrix_fifo`；此模式在 `io_size=1048576`
且未指定 `weight_encoding` 时默认启用 `lattice` 可逆编码。
文件调试后端、显式 `expert`/`matrix` 模式及其他分块大小默认使用 `raw`。
显式启用 `lattice` 支持 `matrix`、`matrix_fifo`、`tile_fifo`，要求 1 MiB 分块。
FIFO 模式要求槽位不少于 Top-K；降槽位重试可保留编码并切换为 `matrix`。

建议正式配置写明参数，避免旧配置中的七槽位或旧流水线覆盖默认值：

```json
{
  "method": "bdev_moe_create",
  "params": {
    "name": "moe_bdev_0",
    "backend": "nvme",
    "base_bdev": "weights",
    "weight_dir": "/tmp/moe_weights",
    "cache_slots": 8,
    "pipeline": "matrix_fifo",
    "weight_encoding": "lattice",
    "io_size": 1048576,
    "io_depth": 4,
    "compute_threads": 1,
    "warmup_requests": 16
  }
}
```

`base_bdev` 必须改为实际权重设备名称。本地模拟使用 `backend=aio`。
CLI 对应 `--weight-encoding lattice`；设置 `weight_encoding=raw` 可以关闭编码，
保留 FIFO 和八槽位。`bdev_get_bdevs` 和保存配置均显示实际编码与流水线。
修改配置后重启会重新导入权重，原始权重文件不变。

## 启动预热与就绪

`bdev_moe_create` 默认在导入完成后内部执行 16 次完整推理，再注册 MoE
bdev 并返回创建成功。正常 JSON 配置依次创建 bdev、namespace、listener，
因此 MoE 服务对外就绪和 `MoE target started` 日志都晚于预热完成。
预热复用配置指定的计算线程、Router/Top-K、专家读取、解码、计算和聚合路径。
输入由独立的固定 PRNG 状态生成，不依赖评测输入或权重生成种子。

缓存槽位在预热后保留实际加载的 FP32 权重。预热输出清零，不能被客户端
当作当前请求结果读取；请求序号、命中/读取/解码统计从第一个客户端请求
开始，诊断 JSONL 不包含内部预热。Nsight 的正式请求采集窗口也不受预热
序号影响。没有新增专家缓冲区，仍受原内存预算限制。

启动缓存分配遇到 `ENOMEM`，或完整预热失败，会减少一个缓存槽位并重试
一次。当前八槽位配置会降到七；此时 Top-8 全 miss 无法一次预留全部槽位，
因此切换为已有的 `matrix` 流水线，计算完成后回收槽位再加载其他专家。
盘上的无损编码不变，不重新导入或编码。预热中途失败时先排空 I/O 和计算，
再回收槽位、从头执行预热。第二次失败仍返回创建错误，不发布就绪。
设备移除、关闭或仅剩一个槽位时直接返回错误。

`warmup_requests` 支持 0–64，0 可用于冷启动与故障实验；CLI 为
`--warmup-requests 16`。`bdev_get_bdevs` 显示实际 `cache_slots`、
`warmup_requests`、`warmup_completed`、`warmup_retries` 和流水线；
保存配置记录重试后的实际槽位和模式。减少缓存槽位不改变 SPDK 的预留池大小。
外部 benchmark 的预热步骤仍按其原逻辑执行；此前性能报告仅使用外部预热，
不能将新增启动预热解释为已经测得的 P99 收益。

验证启动顺序和一次重试：

```sh
make -C module/bdev/moe -j4
make -C examples/moe/tgt -j4
make -C test/moe/pipeline -j4
MOE_TEST_STARTUP_WARMUP=16 MOE_TEST_TILE=1 MOE_TEST_TILE_COLS=512 \
  MOE_TEST_ENCODING=lattice MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE= \
  MOE_TEST_ROUNDS=50 python3 test/moe/test_moe_pipeline.py
MOE_TEST_TILE=1 MOE_TEST_TILE_COLS=512 MOE_TEST_ENCODING=lattice \
  MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE=warmup_retry MOE_TEST_ROUNDS=50 \
  python3 test/moe/test_moe_pipeline.py
MOE_TEST_TILE=1 MOE_TEST_TILE_COLS=512 MOE_TEST_ENCODING=lattice \
  MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE=warmup_alloc_retry MOE_TEST_ROUNDS=50 \
  python3 test/moe/test_moe_pipeline.py
MOE_TEST_TILE=1 MOE_TEST_TILE_COLS=512 MOE_TEST_ENCODING=lattice \
  MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE=warmup_failure \
  python3 test/moe/test_moe_pipeline.py
```

故障注入仅链接进 `moe_pipeline_test`，不进入生产 target。
测试确认预热完成日志先于 ready、首次客户端 READ 不包含预热输出、内部
预热不占客户端请求序号，临时 I/O/分配失败可以降槽位恢复，坏权重只重试
一次且阻止就绪。上述正常与恢复后的 50 个小尺寸独立参考比对误差均为 0。

正式尺寸 2048/7168、256 专家、Top-8 的本机 AIO 验证记录位于
`/root/moe-run-logs/startup_warmup_20261014/`。默认内部预热 16 次，
从计算 worker 启动到预热完成约 5.44 秒；之后才出现 TCP 监听日志。
预热读取编码数据 16,928,014,336 字节，恢复 FP32 数据 21,667,774,464 字节。
随后 seed 4242 的 50 个独立 TCP 参考输入误差均为 0，诊断仅包含客户端
请求 1–50，`bdev_get_bdevs` 确认完成 16 次、八槽位且没有重试。
这是本地 CPU/AIO 正确性与就绪顺序验证，不是官方 NVMe/2 GB cgroup 验收。
最终诊断文件在 target 关闭并刷新后验证，详见该目录的 `validation.json`。

当前版本八槽位的内存补测记录位于
`/root/moe-run-logs/eight_slots_memory_20261014/`。同样完成 16 次内部预热和
50 个正式尺寸独立 TCP 参考比对，误差均为 0，始终保留八槽位且没有重试。
以 100 ms 间隔记录进程 RSS，并读取内核 `VmHWM` 捕获采样间隔中的峰值：

| 范围 | 峰值 VmHWM / KiB | 折合 GiB |
| --- | ---: | ---: |
| 初始化、预热和推理，关闭前 | 1,498,312 | 1.429 |
| 完整生命周期，包含退出 | 1,695,388 | 1.617 |

最终峰值为 1,736,077,312 字节，低于十进制 2 GB，也低于 2 GiB。
较高峰值出现在关闭前快照之后；不把它解释为推理阶段峰值。
`memory.jsonl` 的 `inference` 标签覆盖预热后至进程退出，包含关闭过程。
最大片段采样 RSS 为 1,660,184 KiB；完整生命周期数值采用更高的内核 HWM。
完整原始采样与汇总分别为 `memory.jsonl`、`memory_summary.json`。

该测量使用 `--no-huge`，`HugetlbPages` 全程为 0。本机 THP 为 `never`，
预留的 2 MiB、1 GiB HugeTLB 页均为 0；未完成大页性能对照。
大页环境中的内存占用和性能不能直接由这批普通页结果推出。无需在提交代码中
另加 cgroup 限制，按官方环境提供的硬限制运行并监测实际峰值。

## SPDK 2 MiB 大页与预分配

`moe_tgt` 默认使用 2 MiB HugeTLB，默认 SPDK 内存池为 256 MiB（128 页）。
优先验证 `--huge-dir` 指定的挂载点，否则自动寻找可读写的 2 MiB hugetlbfs。
不会选择 1 GiB 页或静默退回普通页。未配置大页时启动失败并给出说明；
本地普通页调试仍可显式传入 `--no-huge --no-pci -s 256`。

启动接入 SPDK 的 `spdk_app_opts`、`spdk_env_init` 和现有 DMA 分配接口。
启用 DPDK `--legacy-mem` 以及仓库已有的 `MOE_FIXED_HUGEPAGES` 限额逻辑，
在初始映射之前把 2 MiB 页数限制为指定池大小，避免先映射宿主机全部大页。
固定池在推理时不再扩张。DPDK 初始化逐页预触碰；编码暂存区由 SPDK
分配，解码专家缓存使用预触碰的 CPU 内存并建议 2 MiB THP。随后执行
默认 16 次完整推理预热，完成后才发布就绪。
预分配失败发生在 EAL 初始化阶段，不能靠之后的降缓存槽位重试解决。

256 MiB SPDK 池用于对象池、DMA 暂存区、导入缓冲区等；八槽位的
1344 MiB 解码专家缓存位于 CPU 内存，两者合计 1600 MiB，不额外复制专家。
普通堆、线程栈、计算工作区和元数据还需另算。按十进制 2 GB 总上限，
其余余量约 307 MiB；是否足够须在真实大页环境记录完整生命周期峰值。
可用 `-s` 显式调整池大小，要求正偶数 MiB；不要仅为增加余量而盲目扩大池。
未编码 AIO/NVMe 的专家缓存仍使用 DMA 内存，完整尺寸 raw 调试须显式
增大 SPDK 池。编码路径的详细分配关系见 `cpu_expert_cache.md`。

在目标宿主机上由管理员预留页并挂载；以下以目标 reactor 所在 node0、
独占实验机且当前预留不足为例，已有更大预留池时无需减少它：

```sh
# 先查看目标 NUMA 节点的 2 MiB 空闲页数；128 页应留给本次进程。
cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages
# 仅当当前预留少于 128 且无其他用户时执行；写入后检查实际页数与空闲页数。
echo 128 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
mkdir -p /mnt/moe_hugepages_2m
mount -t hugetlbfs -o pagesize=2M none /mnt/moe_hugepages_2m
build/examples/moe_tgt --huge-dir /mnt/moe_hugepages_2m -c /path/to/nvme_config.json
```

管理员预留与挂载在运行 target 之前完成，应用不改宿主机预留页数、不绑定
磁盘。NVMe 配置需要启用 PCI，不能加 `--no-pci`；本地 AIO 可加该选项。
HugeTLB 占用不一定包含在进程 RSS/VmHWM 中，需同时检查
`/proc/PID/status` 的 `HugetlbPages`、`smaps_rollup` 的 `Private_Hugetlb` /
`Shared_Hugetlb` 及官方 cgroup 的内存和 hugetlb 计数，不重复相加同一份页。

1 MiB 是逻辑读取/编码块大小，2 MiB 是内存映射粒度，二者无需一致。
解码输出按原有逻辑写入 FP32 缓存，计算仍由矩阵或 tile 的就绪边界触发，
不等待整张 2 MiB 页填满。压缩后的实际读取长度仍按设备 LBA 对齐；
大页不改变 SSD 读取粒度或传输字节数。

本机未预留 HugeTLB 且 `/sys` 只读，当前只能验证挂载选择、参数检查、
固定池页数上限和显式普通页回归；实际大页分配、内存峰值与 P99 尚未实测。

```sh
cc -D_GNU_SOURCE -Iinclude test/moe/test_moe_hugepages.c -o /tmp/moe_hugepages_test
/tmp/moe_hugepages_test
cc test/moe/test_fixed_pool.c -o /tmp/moe_fixed_pool_test
/tmp/moe_fixed_pool_test
```

## 编码和读取

每个矩阵先按现有 64 列布局打包，再独立编码每个最多 1 MiB 的块。
编码器从实际 FP32 位模式寻找精确二进制缩放因子，将权重表示为整数，
减去块内最小值后按所需位宽打包。它逐值验证恢复结果的 FP32 位模式。
非有限值、负零、超出缩放或整数范围、无法精确恢复、无压缩收益的块
自动保存原始字节。没有训练字典、权重生成种子依赖或近似量化。

盘上格式版本为 2，每块索引 24 字节，包含元素数、载荷字节数、缩放指数、
整数基值、位宽及校验值。索引位于第一个 1 MiB 区域的布局头之后；
Router 仍保存原始 FP32。载荷使用四路 CRC32C 流校验，解码前验证。
版本 1 原始读取路径仍可显式选择。

为减少布局改动，专家间距和逻辑块偏移保持原始布局，编码载荷之后留空隙。
读取长度按实际载荷向设备 LBA 对齐。**节省读取字节，不缩短镜像文件大小**。
FIFO 仍按 Top-K 顺序提交每个 miss 专家的 Gate、Up、Down 矩阵。

每个 I/O 操作预分配一个 1 MiB 加 4096 字节的 DMA 暂存区。
读取完成后在 owner 线程中校验、解码到最终 FP32 缓存位置，随后通知矩阵就绪。
位宽不超过 25 时使用 AVX2 解码，其余支持宽度使用标量路径。
计算和聚合顺序保持原实现。关闭诊断后不采集解码计时；诊断开启时可读
`decoded_bytes_total`、`decode_ticks_total`。

## 验证命令

```sh
make -C module/bdev/moe -j4
make -C examples/moe/tgt -j4
make -C test/moe/pipeline -j4
cc -D_GNU_SOURCE -O2 -fno-fast-math -g -fsanitize=address,undefined -Iinclude \
  test/moe/test_moe_codec.c -lm -o /tmp/moe_codec_test
/tmp/moe_codec_test
MOE_TEST_FIFO=1 MOE_TEST_ENCODING=lattice MOE_TEST_LATTICE_WEIGHTS=1 \
  MOE_TEST_DEFAULTS=1 MOE_TEST_CASE= MOE_TEST_ROUNDS=50 \
  python3 test/moe/test_moe_pipeline.py
MOE_TEST_FIFO=1 MOE_TEST_ENCODING=lattice MOE_TEST_LATTICE_WEIGHTS=1 \
  MOE_TEST_ROUNDS=50 python3 test/moe/test_moe_pipeline.py
```

独立 TCP 正确性验证使用赛题输出阈值：每个输出有限，最大绝对误差严格小于
`1e-5f`。输入校验不计入性能测试。

```sh
MOE_INITIATOR_CPU_MASK=0x1 test/moe/accuracy_initiator/moe_accuracy_initiator 50 4242
python3 test/moe/test_moe_fifo_bench.py /path/to/authorized_aio_config.json \
  /path/to/results --encoding-comparison
```

比较脚本会重新导入配置指定的镜像，只能指定已授权覆盖的实验镜像。
每轮 16 次预热、50 次计时，并检查输入序列摘要一致。

全量离线验证无新增依赖：

```sh
cc -O3 -fno-fast-math -D_GNU_SOURCE -fPIC -shared -Iinclude \
  module/bdev/moe/moe_codec.c -lm -o /tmp/libmoe_codec.so
python3 test/moe/research_reversible.py /path/to/raw_version1.bin \
  /tmp/libmoe_codec.so /tmp/roundtrip.json
python3 test/moe/research_reversible.py /path/to/encoded_version2.bin \
  /tmp/libmoe_codec.so /tmp/stored.json --decode-only \
  --expected-sha256 ORIGINAL_WEIGHT_PAYLOAD_SHA256
```

## 部署条件

导入编码会增加启动时间；本地完整导入约 288–291 秒，原始格式约 64–65 秒。
导入不计入稳态请求 P99。编码路径仍缓存完整 FP32 权重，八槽位占 1344 MiB，
不会降低专家缓存占用。局部暂存区和索引也需计入赛题 2 GB 总内存。

已根据用户决定正式启用。官方原生 NVMe、设备带宽、cgroup 包含 hugepage 的
内存验收仍需要在对应环境实测。owner 同步解码会占用 I/O 提交线程，
本地暖缓冲解码速度不能直接当作官方流水线速度。
完整实测见 [reversible_weights_results.md](reversible_weights_results.md)。
