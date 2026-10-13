# 在 Nsight GUI 中查看 CPU MoE 流水线

新报告：
`/root/moe-run-logs/reversible_nsys_gui_20261013_v2/moe_cpu_experts_io_decode.nsys-rep`。
旧的 `reversible_nsys_20261013/lattice.nsys-rep` 已按用户要求删除，原始 SQLite
和日志保留供追溯。本报告来自实际 CPU AVX2 程序，没有 CUDA/GPU 执行。

## 打开与展开

1. 展开 target 进程及其线程，在 **CPU Compute / Router + Experts + Aggregate**
   下展开 **NVTX**。绿色区间包含真实专家 ID 和 Gate、Up、Down；展开嵌套叶阶段
   可见 SiLU(Gate)*Up。紫色为 Router、Top-K、Softmax，粉色为聚合。
2. 展开 **CPU IO + Decode / SPDK owner → NVTX**。橙色为 Decode、校验和 FP32
   写回，深蓝色为 CPU 提交 SPDK read。每个区间标有专家 ID 和矩阵名称。
3. 展开进程级 NVTX/Annotations。蓝色 **I/O outstanding** 是提交成功到回调
   入口的异步范围；灰色 **Request** 是一次完整 target forward。红色
   **Pipeline wait** 是 owner 观察到计算空闲但矩阵 I/O/解码仍待完成的范围。
4. 选择 **Request 17 / CPU MoE Forward**，放大到该请求。Router、Top-K、
   Softmax 很短，需要进一步放大请求开头；查看 Decode 时放大到几毫秒窗口。
5. 右键某行，选择 **Show in Events View**，查看完整标签和起止时间。
   该操作参见 [NVIDIA 官方说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#timeline-events-correlation)。

计算线程上的红色 **CPU worker condvar wait** 包围实际条件变量等待，可能包含
矩阵就绪、owner 调度与唤醒延迟。蓝色异步 I/O 不能当作 CPU 运行时间或 SSD
持续传输时间。此程序的 Normalize 是 Top-K 路由权重的 Softmax，没有 RMSNorm。

## 实际验证

采集为 16 次预热后的 50 个请求，输入与此前六轮实验一致。报告只包含请求
17–66，权重导入和预热不在时间线中。

| 事件 | 数量 |
| --- | ---: |
| Request、Router、Top-K、Normalize、Aggregate | 各 50 |
| Gate、Up、SiLU Multiply、Down | 各 400 |
| IO Submit、I/O Outstanding、Decode | 各 66024 |
| Pipeline Wait | 1171 |
| Worker Wait | 1286 |

每个请求都核对 Gate、Up、Down 的八个真实专家 ID 一致。读取和解码字节与
store 诊断计数一致。CPU 阶段、提交、Decode 均为原生线程 push/pop 范围；
Request 和未完成 I/O 保留异步范围。计算与 Decode 的线程不同，导出的线程
名称已核实。独立 TCP 准确性校验最大绝对误差为 0。

采集开关影响了第一个和最后一个 TCP 请求：末次请求的 TCP 时间约 4.50 秒，
而该 target forward 约 0.35 秒。报告用于流水线诊断，不能用这批 TCP P99
替代关闭 profiler 的性能测试，也没有删除首尾样本以宣称性能提升。

本机只有 Nsight Systems CLI，未安装对应 GUI；已验证原生报告和 SQLite
事件，没有声称完成本机 GUI 展开或截图验证。`validation.json` 明确记录
`gui_verified=false`。普通构建已恢复关闭 NVTX。

完整采集命令、文件摘要和环境位于报告目录下的 `manifest.json`。
源文件摘要及共享标记头另保存在 `source_fingerprints.json`、`moe_trace_source.h`。
采集、分析和验收脚本见 [README](../../test/moe/nsys/README.md)。
