# 独立解码线程

生产入口默认保持整矩阵流水线，使用I/O深度16、2个解码线程、1个计算线程和8个专家槽位。RPC增加 `decode_threads`，范围0至4；省略时为0，保留同步回调解码兼容路径。独立解码仅用于lattice编码。

reactor在读取回调中释放I/O并将编码块交给解码队列，随后可继续提交读取。解码线程执行原有校验和无损解码，写入各自块对应的专家缓存区域，再通过SPDK消息通知owner；只有owner更新完成计数、就绪前缀和矩阵完成状态。计算仍等待整矩阵解码完成。

I/O在途及资源等待最多为 `io_depth`，异步路径总共预分配 `2 × io_depth` 个编码缓冲槽位，覆盖读取、待解码、解码中和等待owner确认的块。缓冲区在owner处理解码完成前不会复用；池耗尽时停止新增读取，队列不会扩容。深度16时为32个 `1MiB+4KiB` DMA缓冲区，来自既有256MiB池，不增加专家缓存或扩大SPDK池。

解码线程与计算线程均绑定在reactor所在NUMA节点的不同物理核心上，避开reactor及彼此的SMT同胞。线程在启动预热前创建，关闭时先等待读/解码全部完成，再停止并join线程、释放内存。取消或解码失败仍保留所有未完成块的引用，直到其完成通知被owner消费。

```json
{"method":"bdev_moe_create","params":{"name":"moe_bdev_0","backend":"nvme","base_bdev":"ScoringNvmen1","weight_dir":"/tmp/moe_weights","pipeline":"matrix_fifo","weight_encoding":"lattice","cache_slots":8,"io_depth":16,"io_size":1048576,"decode_threads":2,"compute_threads":1}}
```

构建仍使用现有SPDK流程，不增加依赖。小规模验证可复用pipeline测试：

```sh
make -C module/bdev/moe -j2
make -C examples/moe/tgt -j2
make -C test/moe/tile_progress -j2
MOE_TEST_WIDE=1 MOE_TEST_FIFO=1 MOE_TEST_ENCODING=lattice \
  MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE= MOE_TEST_IO_DEPTH=16 \
  MOE_TEST_DECODE_THREADS=2 MOE_TEST_STARTUP_WARMUP=16 \
  python3 test/moe/test_moe_pipeline.py
```

测试需要reactor、计算、解码各自可用的不同物理核心。诊断日志增加 `decode_threads`、`decode_peak`、`decode_pending` 和 `codec_buffer_slots`。`decode_peak`包含排队、执行及等待owner确认的块；解码累计ticks在并发模式下是各工作线程耗时之和，不能直接解释为端到端解码窗口。

原生NVMe性能验证固定matrix_fifo、深度16、AVX2、1计算线程、8槽位和相同权重，比较decode_threads=0/1/2。同构建0线程为匹配对照。每组独立精度校验50次、内部预热16次，每轮外部预热16次，计时50次，测2轮取较低P99；监测目标RSS高水位、去重HugeTLB及实际入口进程峰值，不额外施加cgroup限制。

独立构建已通过，目标二进制SHA256为 `a414c836fb71f604af5b54f170349bad42fc2e139a5c020b396cd35f25d14874`。启动器5项测试和I/O资源等待/取消回归均通过。D=128、H=4096、9专家、Top-8的小规模测试中，深度16、双解码线程的冷启动与预热16次两组各32个正常请求最大绝对误差均为0，每个请求结束后decode_pending为0。默认深度4的损坏Down矩阵用例通过拒绝及恢复后32次准确推理，执行中关闭用例通过资源排空检查。

故障用例最初因测试脚本不允许对非正常用例覆盖深度而在启动前被拒绝；改用默认深度4后通过，未因此修改候选源码。格式门禁仍返回1，日志与此前既有格式问题完全一致，未修改无关文件。正式尺寸原生NVMe的0/1/2线程对照正在运行，本文件暂不宣称性能提升或正式规模内存通过。
