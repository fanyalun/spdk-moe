# 默认绑核与 NUMA 分配偏好

target 现在在 SPDK 初始化前保存进程允许的 CPU 集合。不传 `-m`、`--cpumask` 或 `--lcores` 时，自动选择一个 reactor 核心，并留出同 NUMA 节点、同 socket 的另一物理核心用于计算。默认避开 initiator 的逻辑 CPU 0 及其 SMT 兄弟；本机通常为 reactor CPU 2、计算 CPU 1，不写死 CPU 10。

显式 reactor 参数保留优先级；仅指定 `--main-core` 时也采用该核心。MoE JSON 的 `compute_cpu` 继续指定第一个计算线程，其余工作线程自动选择独立物理核心，避开 reactor、已选工作线程和预留 initiator 核心。自动计算线程与 reactor 保持同节点；显式 compute_cpu 可跨节点，需使用者判断其影响。默认选择、-m 掩码和计算线程检查启动时允许的 CPU 集合。高级 --lcores 映射交由 SPDK 处理，当前自动物理拓扑策略未验证此类逻辑/物理核心重映射，建议使用 -m。

```sh
# AIO 开发环境：无需再传 reactor 绑核参数。
build/examples/moe_tgt --no-huge --no-pci -c /path/to/aio_target.json

# 如需复现旧的 reactor CPU 10 设置，仍可显式覆盖。
build/examples/moe_tgt --no-huge --no-pci -m 0x400 -c /path/to/aio_target.json
```

建议从 JSON 中移除旧的固定 compute_cpu，使用自动选择；若保留，则该核心必须在启动时允许的 CPU 集合内，且不能与 reactor 共用物理核心。显式 compute_cpu 可覆盖 initiator 预留规则，因此不应无意指定到 initiator 核心。

如果官方 initiator 使用其他 CPU，可在启动 target 前设置 `MOE_INITIATOR_CPU=<逻辑CPU编号>`。这只告诉 target 避开哪个物理核心，不会修改或重新绑定官方 initiator。默认 0 是发布版参考 initiator 的常见设置，不能据此假定官方隐藏程序固定使用 CPU 0。

如果不能识别足够的允许核心/拓扑，默认选择明确失败，不擅自逃出 CPU 限制或静默与 initiator 同核。默认自动方案需要至少两个可用 target 物理核心；增加计算线程需要更多独立物理核心。绑定是线程亲和性，不会让这些核心获得系统独占权。

## NUMA 的使用

多 NUMA 节点机器值得保持“reactor、计算线程、权重内存”接近，以减少跨节点访存。单 NUMA 节点没有跨节点选址问题，但仍应避开物理核心竞争。不能仅凭 CPU 型号假定机器只有一个 NUMA 节点。

本轮默认增加专家 DMA 缓冲的 reactor 节点分配偏好，使用 SPDK 的 socket 分配接口。现有 SPDK 实现在该节点分配失败时允许回退到其他节点；总池大小不因此扩大。它不保证全部内存严格驻留同节点，也不改变整个系统的内存策略。普通工作区在初始化时分配/触页，其实际分布仍取决于系统内存策略。

不要默认强制 `--enforce-numa`：这会禁止指定节点的 SPDK 分配失败后回退，即使其他节点还有内存也可能启动失败。应在确认节点内 DMA 池容量和官方内存记账后再验证。该选项也不等同于整个进程的严格 membind。

在正式 NVMe 环境中，先读取设备的 NUMA 节点，再在官方允许的核心范围内选择靠近设备的 reactor 和计算核心：

```sh
lscpu -e=CPU,CORE,SOCKET,NODE
cat /sys/bus/pci/devices/<NVMe的PCI地址>/numa_node
```

节点值为 -1 表示未提供明确关联。当前默认选核尚不自动跟随 NVMe PCI 设备节点，需通过 -m 与 compute_cpu 或进程启动亲和性配置调整。若环境已安装 numactl，可用节点 0 举例（需替换为实际节点）：

```sh
numactl --cpunodebind=0 --preferred=0 build/examples/moe_tgt -c /path/to/nvme_target.json
```

这是进程级 CPU/内存偏好示例，不修改官方脚本或系统频率。preferred 允许内存回退；不要在未验证容量前直接改为严格 membind。无需为使用本轮内建默认策略安装 numactl。

启动日志输出 default reactor、每个计算线程 CPU 及双方 NUMA 节点。实际 CPU 与内存分布可核查：

```sh
grep Cpus_allowed_list /proc/<target_pid>/task/<thread_tid>/status
cat /proc/<target_pid>/numa_maps
```

## 构建与验证

```sh
make -C lib/moe_ffn -j4
make -C module/bdev/moe -j4
make -C examples/moe/tgt -j4
make -C test/moe/pipeline -j4
python3 test/moe/test_moe_affinity.py
python3 test/moe/test_moe_pipeline.py
```

亲和性测试使用实际 target 和小规模 AIO 权重，在同节点至少五个允许物理核心上验证默认四线程、缩小 CPU 集合、预留 initiator 核心变更、显式覆盖，以及单核心/非法计算核心的失败路径。流水线回归验证计算及 I/O 故障行为。正式尺寸仍需独立精度、内存和官方硬件验证；本轮不宣称新的 NUMA 性能加速倍数。
