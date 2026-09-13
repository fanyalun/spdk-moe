# 固定输入参数筛选

`test/moe/fixed_bench` 是开发测试入口，复用原 benchmark 的连接、vendor/READ 请求和单调时钟函数；官方 benchmark 源码及可执行文件不修改。新入口按请求数量停止，不执行隐含预热，保存每次请求的输入 FNV-1a 摘要及纳秒延迟，并拒绝非有限输出。摘要只用于核对输入一致性，不能替代独立准确性校验。

```sh
make -C test/moe/fixed_bench -j4
test/moe/fixed_bench/moe_fixed_bench 16 42 10 0 /tmp/random.jsonl
test/moe/fixed_bench/moe_fixed_bench 8 42 10 1 /tmp/repeat.jsonl
```

五个位置参数依次为请求数、输入 seed、initiator CPU、是否重复相同输入（0/1）、输出文件。必须先启动对应 target。输入在计时前生成，计时覆盖 vendor 和 READ；输出有限性检查及文件写入不计入延迟。

入口显式设置 SPDK core_mask，并检查初始化后的实际 CPU。原 benchmark 的 `--cpu` 在初始化 SPDK 前调用 sched_setaffinity，但默认 SPDK core_mask 为 0x1，会把线程再次绑定到 CPU 0；仅看报告中的 CPU 参数不足以证明实际绑定。运行未修改的官方 benchmark 时，应将 target reactor 放在其他物理核心，并核对实际线程亲和性。另外，原 run_phase 是先执行再判断截止时间，因此 warmup=0 仍执行一次不计时请求。

## 顺序筛选

```sh
make -C examples/moe/tgt -j4
make -C test/moe/accuracy_initiator -j4
python3 test/moe/test_moe_tuning.py /path/to/aio_target.json --count 16 --cpu 10
```

该脚本重新导入配置中的 AIO 镜像，必须使用已授权覆盖的 MoE 测试文件。每个配置单独启动 target，使用 no-huge、1536MiB SPDK 内存、reactor CPU 0、首个计算线程 CPU 1。默认 initiator CPU 10 避开本机 1/2/4 线程所用核心；在其他拓扑上必须核对实际工作线程日志再运行，脚本不保证任意机器上都无核心冲突。

初始配置为 AVX2、matrix、7 槽位、1MiB、深度 4、预取 2、1 线程。依次比较 AVX-512、8 槽位、256KiB/2MiB、深度 1/8、2/4 线程，每一步固定其他参数。每个候选运行 seed 42/43/44 各 count 个随机输入，再运行 seed 42 的 8 个重复输入，最后调用独立参考 initiator。候选间检查相同批次输入摘要序列完全一致。

本机 8 槽位在 1536MiB 池中实际分配失败，因此 8 槽位候选显式使用 1664MiB 池（参数记录中的 memory_mb），其余候选沿用当前选中配置的池大小。这一对照同时改变了槽位和池容量，不能描述为相同池大小实验；未改变 target 的默认池或失败时禁止隐式扩池的行为。`--resume /path/to/artifacts` 可在同一环境恢复已中断筛选，已成功完成的相同配置和请求数不重复运行；失败日志保留。切换环境或输入权重后应新建实验，不使用 resume。

同一配置内三批随机请求连续沿用缓存，只有第一批第一个请求从空专家缓存开始；重复输入批次也沿用此前缓存，不将其第一个请求称为必然全冷。重复输入收益用去掉第一请求后的平均值单列，不混入随机输入平均值。

自动筛选是保守的开发规则：RSS 采样峰值小于 1800MiB，随机请求平均延迟改善超过 2%，样本最大值不比当前候选增加超过 2%，才替换当前配置。该规则不是 P99/P99.99 排名规则；短样本最大值波动会影响选择，最终需人工检查分轮结果、节流、内存和准确性。筛选完对初始与选中配置分别增加三批 64 请求确认。

脚本保存原始逐请求数据、配置、target/initiator 日志、bdev 查询、独立精度结果、采样 RSS 峰值和计时区间 CPU 节流快照。无论成功或异常都排空并关闭自身 target。不是系统独占性或官方内存准入的替代品。该逐参数筛选不覆盖参数相互作用，也不证明全局最优；官方长样本尾延迟与真实 NVMe 验收仍需单独进行。

## 单专家计算对照

在已构建独立参考和 SPDK 候选计算库的环境中：

```sh
cc -O2 -Wall -Wextra -Ilib/moe_ffn -I../MoE-compute/test \
  test/moe/test_moe_ffn_bench.c \
  ../MoE-compute/build_accuracy/libmoe_spdk_candidate.a \
  ../MoE-compute/build_accuracy/libmoe_reference.a -lm -o /tmp/moe_ffn_bench
taskset -c 1 /tmp/moe_ffn_bench
```

该测试使用确定性合成 FP32 权重与正式矩阵尺寸，打包后的 168MiB 单专家驻留内存；导入、打包和独立参考计算不计时。AVX2/AVX-512 交替先后顺序，各运行三轮、每轮十次；每轮计时前验证非有限值及严格 `<1e-5f`。输出不能当作 SSD 或完整 MoE 延迟。
