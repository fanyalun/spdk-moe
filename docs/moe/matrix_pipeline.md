# 矩阵级加载与计算流水线

`bdev_moe_create` 新增 `pipeline`，取值为 `expert`（默认）或 `matrix`。矩阵级模式仅适用于 AIO/NVMe，支持现有 1/2/4 个计算线程。文件调试后端指定 matrix 时拒绝创建；未知模式同样失败。`bdev_get_bdevs` 的 `driver_specific.moe.pipeline` 显示实际模式。

## 配置与行为

在现有创建参数中增加：

```json
{"pipeline":"matrix","compute_threads":1}
```

CLI 对应 `scripts/rpc.py bdev_moe_create ... --pipeline matrix`。修改启动配置后重启将按现有规则重新导入；不复用上次初始化结果。盘上布局版本、权重文件、64 列打包、vendor/READ 协议均不变。

权重按 Gate → Up → Down 读取。一个矩阵的全部分块完成后，reactor 提交下一个矩阵读取，再调度可执行计算；不会等待整个专家加载完成才开始 Gate。阶段计算依次为 Gate 点积、Up 点积及 SiLU(Gate)×Up、Down 点积。下一阶段读取和上一阶段计算可重叠。

同一专家最多一个计算任务运行，不同专家可由不同线程执行。调度优先级为可执行 Down、可执行 Up、完整权重且尚未计算的专家、可执行 Gate；同级保持 Top-K 顺序。完整权重专家沿用整专家计算快路径。最终输出仍按原 Top-K 顺序以 FMA 汇总。

每个选中专家单独保存 Gate/Up 中间结果，允许线程在阶段之间切换专家。启动时预分配、预触页；2048×7168、Top-8 增加 448KiB 普通内存。专家权重仍占一个完整缓存槽，不能据此减少 168MiB 的槽位预算。

## 资源与错误

`prefetch` 仍限制未完整加载的专家数；每个专家最多一个矩阵读取任务活动，整个 store 共享 `io_depth`。矩阵读取跳过专家区域末尾填充，实际成功读取字节数可能比整专家模式少；正式维度无这项差异。

所有命中在加载 miss 前被引用保护。部分权重槽保持 LOADING，不能命中或被淘汰；计算任务依据单独的矩阵就绪状态执行。只有输出已保存且对应 I/O 全部完成，才释放槽位引用。1 槽位可继续处理 Top-8。

失败或设备移除后停止提交新阶段，排空活动读取与工作线程后完成请求一次。部分权重槽作废；失败结果不能通过 READ 返回。常规 SIGTERM 保留 SPDK 的退出流程：撤销服务并等待已接收请求排空，不强制中断计算线程。ENOMEM 沿用 SPDK 等待回调，不在 reactor 忙等。没有引入每请求大块分配。

## 诊断与验证

现有可关闭 `diagnostics` JSONL 增加 `pipeline`、`first_compute_wait_ticks` 及最多 24 条 `stages` 记录。每条包含 Top-K 位置、阶段号、读取入队 `read_start`、首个分块成功提交 bdev 的 `read_submit`、读取完成以及计算开始/结束 tick。事件在固定请求结构中记录，完成请求时统一输出；正式计时关闭诊断。

读取区间包括分块排队和等待，不能将它当作 SSD 总线持续传输时间。重叠测试使用 `read_submit` 到读取完成的区间，避免仅凭入队就判定 I/O 已提交；ENOMEM 等待期间不记录成功提交时间。计算时间戳在工作线程实际执行阶段时记录。整专家快路径不拆分三个阶段时间，`first_compute_wait_ticks` 则覆盖两种模式，表示接收请求到首次专家计算的时间。

```sh
cmake --build ../MoE-compute/build_accuracy -j4 --target test_accuracy test_spdk_accuracy
../MoE-compute/build_accuracy/test_accuracy
../MoE-compute/build_accuracy/test_spdk_accuracy
make -C lib/moe_ffn -j4
make -C module/bdev/moe -j4
make -C test/moe/pipeline -j4
python3 test/moe/test_moe_pipeline.py
```

计算测试按阶段跨专家交错执行并与冻结参考比较，覆盖三种内核、两种布局及尾部维度。集成测试覆盖两模式、少槽位、多线程、全命中、淘汰、慢 I/O、读取失败和移除。慢 I/O 用例要求事件证明同一专家内部读算区间重叠，并检查矩阵完成及计算依赖。

SIGTERM 用例等待 vendor 请求已提交的标志，再在慢 I/O 期间发送信号，要求在途请求排空并正常退出；可用 `MOE_TEST_CASE=shutdown python3 test/moe/test_moe_pipeline.py` 单独复核。错误权重用例分别注入 Gate/Up/Down NaN，随后修复镜像并重试，验证失败槽位不会保留为有效缓存。

正式尺寸复用已有 MoE 打包文件，以下命令会重新写入指定文件；仅用于明确授权的测试镜像：

```sh
make -C test/moe/full_pipeline -j4
python3 test/moe/test_moe_full_pipeline.py /tmp/moe_weights /path/to/output \
  --pipeline matrix --reuse-image /path/to/existing/packed.bin
```

低内存功能测试可增加 `--cache-slots 1 --mem-size 512`。脚本先检查共享 cgroup 的估计余量，拒绝明显不足的配置；此估计不是正式内存验收，也无法防止其他进程临时增加占用。原有空闲空间检查保留至少 10GiB，并结合 df 与 statvfs 的较小值。

独占环境性能对照入口：

```sh
make -C examples/moe/tgt -j4
python3 test/moe/test_moe_pipeline_bench.py /path/to/aio_target.json \
  --rounds 3 --runtime 60 --cpu 2
```

该脚本仅接受单个 AIO MoE 的现有 target 配置，交替启动两模式，每轮重新导入同一文件、关闭诊断，调用原 benchmark。固定 seed 42、warmup 0 秒；target 初始缓存为空，但原 run_phase 仍执行一次不计时请求，因此首个计时请求不是严格全冷。基于时间停止导致实际样本数量可能不同。保存配置、日志、CPU 节流、RSS 和原始 JSON。全命中与其他受控缓存分布的性能仍需单独测量，不能由随机输入结果推断。参考 initiator 的实际 CPU 绑定及 reactor-cpu 参数见 [参数筛选说明](tuning.md)。

`--warmup` 的单位是秒，不是请求数。正式 P99/P99.99、NVMe 与官方内存验收继续遵循原发布门槛。
