# 按输出通道块推进的 MoE 流水线

## 64 通道的含义

现有 `MOE_PACK_COLS=64` 是固定内核设计，并非搜索得到的最优参数。
AVX2 内核使用八组向量累加器，每组八个 FP32；AVX512 使用四组、每组十六个。
多组独立累加器复用广播的输入，每个输出保持输入维度递增的 FMA 顺序。
历史测试验证了内核正确性及不同指令集，未证明 32/64/128 存储宽度谁最优。

本实现保留 64 通道存储布局，新增 `pipeline=tile_fifo` 和
`compute_tile_cols`。后者是每次计算任务的输出通道数上限，默认 512，
支持 64 的倍数，范围 64–2048；缓存命中仍保留完整专家计算的快速路径。
这不是更换 SIMD 内核宽度，也不是更换 I/O 或编码单位。

```json
{
  "pipeline": "tile_fifo",
  "cache_slots": 8,
  "weight_encoding": "lattice",
  "io_size": 1048576,
  "io_depth": 4,
  "compute_tile_cols": 512,
  "compute_threads": 1
}
```

CLI 增加 `--pipeline tile_fifo --compute-tile-cols 512`。
模式要求 AIO/NVMe、槽位不少于 Top-K、1 MiB 逻辑 I/O 块。
当前默认仍为 `matrix_fifo`；实验模式需要显式配置。

## 权重就绪与计算依赖

每个矩阵 descriptor 增加已完成块的位图。读取成功且解码完成后标记对应
逻辑 1 MiB 块，只发布从零开始连续完成的字节范围。后面的块先完成不会
发布中间缺失区域。计算线程可读取发布范围，后续 DMA/解码只写其他区域。
每个矩阵最多 64 个块，不分配逐块堆对象。

已完成字节转换为完整 64 通道权重组的数量：

| 矩阵 | 一个 64 通道组的 FP32 权重 | 本次就绪条件 |
| --- | ---: | --- |
| Gate / Up | 512 KiB | 一个 1 MiB 块完成，可用 128 通道 |
| Down | 1.75 MiB | 覆盖整组的逻辑块全部完成 |

不同维度按实际输入行数计算组大小；表中为官方 2048/7168。
调度顺序为 Down、Up、命中专家、Gate。每个专家一次只执行一个任务：

- Gate：对应权重完整即可计算该输出范围。
- Up：对应 Up 权重完整，且同范围 Gate 已计算，随后执行 SiLU(Gate)*Up。
- Down：整个激活向量完成，且该输出组权重完整，才计算对应输出范围。

`moe_expert_slice` 对输出范围使用原来的 ordered SIMD 内核，不拆分一个
输出的输入维度累加链。每个输出的 FMA 顺序、SiLU 和最终 Top-K 聚合顺序
保持原实现。多个 worker 完成 Down 时只复制对应输出范围，避免覆盖其他块。
错误与关闭仍排空 I/O 和 worker 后释放缓存，部分失败专家不会保留为有效命中。

## 编码单位是否要调整

本版继续使用版本 2 镜像和 1 MiB 独立可逆编码。典型实际读取 800 KiB，
恢复为 1 MiB FP32 后再发布。每个编码块仍须完整读入才能校验和解码。

如果要求每次仅读取一个 64 通道组，就需要将编码边界改为 Gate/Up 的
512 KiB 和 Down 的 1.75 MiB，或者采用更小的独立编码片段。那会改变
索引、读取粒度和队列行为，不能假设更小一定更快。本次先保持编码、
存储布局和 I/O 深度一致，比较计算调度粒度。

## 验证与复现

```sh
make -C lib/moe_ffn -j4
make -C module/bdev/moe -j4
make -C examples/moe/tgt -j4
make -C test/moe/pipeline -j4
make -C test/moe/tile_progress -j4
cc -D_GNU_SOURCE -O2 -fno-fast-math -g -fsanitize=address,undefined -Iinclude -Ilib \
  test/moe/test_moe_slices.c lib/moe_ffn/moe_workspace.c lib/moe_ffn/matvec_packed.c \
  lib/moe_ffn/silu.c lib/moe_ffn/topk.c lib/moe_ffn/softmax.c -lm -o /tmp/moe_slices
/tmp/moe_slices
MOE_TEST_TILE=1 MOE_TEST_ENCODING=lattice MOE_TEST_LATTICE_WEIGHTS=1 \
  MOE_TEST_ROUNDS=50 python3 test/moe/test_moe_pipeline.py
MOE_TEST_WIDE=1 MOE_TEST_TILE=1 MOE_TEST_TILE_COLS=128 MOE_TEST_REACTOR_MASK=0x8 \
  MOE_TEST_ENCODING=lattice MOE_TEST_LATTICE_WEIGHTS=1 MOE_TEST_CASE=slow \
  MOE_TEST_ROUNDS=2 python3 test/moe/test_moe_pipeline.py
python3 test/moe/test_moe_fifo_bench.py /path/to/authorized_aio_config.json \
  /path/to/results --tile-comparison
```

`tile_progress` 使用独立对象文件，避免与普通测试的不同编译宏冲突。
它验证 128×4096 权重的多块读取，限制读速以明确观察首个计算早于完整
矩阵就绪；这只是依赖关系回归测试，不是正式性能条件。
调度诊断的首尾计算时间是整个阶段的 envelope，不能当作连续 CPU active 时间。

已验证所有本机支持的 scalar/AVX2/AVX512 内核、64/128/256/512 任务结果
与完整计算逐位一致；单元测试含 ASAN/UBSAN。I/O 回归测试强制乱序完成，
确认可提前发布连续前缀；集成测试覆盖多个计算线程、缓存命中、取消、
设备移除、shutdown、坏权重、编码及原始回退。
正式尺寸的独立 TCP 验证按 `<1e-5f` 标准，与计时分开进行。

性能脚本保持相同输入、16 次预热和 50 次计时；每个配置重新导入授权实验
镜像。P99 是这 50 个样本中的最大值。原始权重文件不变。
测试结果见 [tile_fifo_results.md](tile_fifo_results.md)。
