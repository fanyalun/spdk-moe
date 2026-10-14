# 推理解码移除载荷checksum

按用户要求，`moe_codec_decode()`不再扫描编码载荷计算checksum，也不再与块索引中的checksum字段比较。同步owner路径及独立解码线程共用此函数，均立即生效，无需配置开关。解码结果仍直接写入专家FP32缓存，没有增加矩阵复制。

保留输入指针、缓冲区长度、元素数、位宽、缩放范围、载荷长度一致性及标量恢复整数范围检查。编码算法、浮点恢复公式、矩阵布局、DMA缓冲区和线程配置不变。盘上版本仍为2、块索引仍为24字节；导入时继续生成checksum字段，但推理解码忽略它，因此已有镜像不必重新编码。

取消运行时载荷校验后，任意载荷位翻转不再保证被拒绝。原始权重NaN故障用例保留在raw模式；lattice模式不再运行依赖载荷校验拒绝损坏数据的bad_gate/up/down用例。实际输出仍须通过独立参考精度验证，不改变官方逐元素最大绝对误差小于1e-5的要求。

nsys新采集的阶段标签为 `CPU Decode + FP32 writeback`，历史报告保留原标签和计时。旧的P99和解码计时均包含checksum，不能当作本次修改后的测量结果。

## 最小验证

```sh
cc -D_GNU_SOURCE -O2 -fno-fast-math -g -fsanitize=undefined \
  -fno-sanitize-recover=all -Iinclude test/moe/test_moe_codec.c -lm \
  -o /tmp/moe_codec_no_checksum_ubsan_test
/tmp/moe_codec_no_checksum_ubsan_test
make -C module/bdev/moe -j2
make -C examples/moe/tgt -j2
make -C test/moe/tile_progress -j2
MOE_TEST_WIDE=1 MOE_TEST_FIFO=1 MOE_TEST_ENCODING=lattice \
  MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE= MOE_TEST_IO_DEPTH=16 \
  MOE_TEST_DECODE_THREADS=1 MOE_TEST_STARTUP_WARMUP=16 \
  MOE_TEST_REACTOR_MASK=0x4 python3 test/moe/test_moe_pipeline.py
```

pipeline命令需要reactor CPU2、测试计算CPU5及同NUMA的独立解码物理核均可用。测试使用临时AIO镜像，不操作原生NVMe实验盘。codec回归覆盖位宽、尾部、原始回退及NaN/Inf/负零的逐位往返；额外篡改checksum字段后仍须逐位恢复正确，并确认非法长度和编码参数仍被拒绝。

## 验证结果

裸金属隔离目录增量构建MoE模块、target及tile_progress测试均通过。UBSan codec回归通过，位宽、紧边界、尾部、raw回退、NaN/Inf/负零均逐位恢复正确；checksum字段被篡改后仍逐位一致，非法长度、位宽及缩放参数仍被拒绝。

独立冻结参考流水线使用D=128、H=4096、9专家、Top-8、matrix_fifo、深度16、单解码及单计算线程，内部预热16次，随后32次正常推理最大绝对误差为0。盘上所有矩阵的解码结果与打包后的原权重逐位一致；请求结束后decode_pending均为0，预分配编码槽位仍为32。

目标二进制SHA256为 `7bec513ddd9f5a721b7555c31a7df9d391f1b2ed698b6166eb7b6b15b2d37dce`。本次未恢复性能参数搜索，未运行正式尺寸原生NVMe的准确性initiator或P99复测，不能据此宣称端到端加速幅度。

首次ASan+UBSan测试启动时被系统预加载的 `/$LIB/libonion.so` 阻止：ASan提示其runtime不是加载列表首项，测试主体未执行。已先报告，未改系统配置或候选源码，改用上述UBSan命令完成回归；不能声称ASan通过。

`git diff --check`通过。格式门禁返回1，日志与前轮既有格式失败逐字一致，包含既有注释/头文件问题及缺少astyle/markdownlint等检查工具；未修改无关文件或安装依赖，不能声称格式门禁通过。
