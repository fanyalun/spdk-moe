# 可逆权重编码

## 默认配置

`bdev_moe_create` 默认缓存槽位为 8。AIO/NVMe 后端未指定 `pipeline`
且槽位不少于 Top-K 时，使用 `matrix_fifo`；此模式在 `io_size=1048576`
且未指定 `weight_encoding` 时默认启用 `lattice` 可逆编码。
文件调试后端、显式 `expert`/`matrix` 模式及其他分块大小默认使用 `raw`。
显式启用 `lattice` 要求 FIFO、槽位不少于 Top-K、1 MiB 分块，否则创建失败。

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
    "compute_threads": 1
  }
}
```

`base_bdev` 必须改为实际权重设备名称。本地模拟使用 `backend=aio`。
CLI 对应 `--weight-encoding lattice`；设置 `weight_encoding=raw` 可以关闭编码，
保留 FIFO 和八槽位。`bdev_get_bdevs` 和保存配置均显示实际编码与流水线。
修改配置后重启会重新导入权重，原始权重文件不变。

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
