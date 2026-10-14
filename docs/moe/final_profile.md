# 最终生产配置与恢复路径

正式入口是 `run_target.py`。正常配置固定为 AVX2/FMA、64通道打包、matrix_fifo、
lattice编码、8槽、1MiB逻辑编码块、深度16、1计算线程、1独立解码线程、
16次内部预热，以及256MiB的2MiB HugeTLB DMA池。8槽CPU缓存共1344MiB，
2MiB对齐、清零预触页并请求透明大页。THP未生效时仍可正确运行。

## 必须保留的恢复行为

缓存分配或内部预热首次失败时，最多从8槽降至7槽，并重新执行16次预热。
Top-K始终为8；7槽通过matrix流水线复用已释放的槽，内部换入并发固定为2。
正常8槽使用matrix_fifo。再次失败或设备移除时不发布服务，固定池不会扩容。

保留I/O资源等待与重试、读写失败及取消后的在途排空、设备移除通知、计算与
解码完成通知、缓存引用保护和退出时的线程join。有限输出及编码长度、位宽、
缩放和标量恢复范围检查继续执行。

解码不扫描或比较载荷checksum。编码仍写入版本2的既有索引字段，盘上格式
不变。无法精确编码的块回退为原始FP32字节，位宽大于25的块和SIMD尾部仍
需要标量恢复。这些是无损正确性所需分支。

## 已移除的可选路径

生产bdev不再接受file后端、expert/tile流水线、raw权重格式、AVX-512、多计算
线程、同步解码或多解码线程、9槽、可变I/O深度、可变逻辑编码块和tile通道参数。
RPC省略固定参数时使用最终值；显式提供不同值会返回EINVAL。prefetch和
compute_tile_cols字段已移除。部署所需的设备、驱动、namespace、权重目录和
CPU亲和性参数保留。AIO仅保留用于小规模独立参考和故障回归。

`bdev_get_bdevs`回显实际槽位、流水线、预热次数及重试次数。配置导出始终描述
正常8槽入口，重放后重新进行相同的启动恢复；不把降级结果变成另一套可选配置。
启动器只允许(8槽,0次重试,matrix_fifo)或(7槽,1次重试,matrix)，并核对单计算、
单解码和CPU缓存，全部满足后才建立TCP监听。

## 验证

以下命令在已有SPDK依赖和冻结参考库的环境执行，不需要安装新依赖：

```sh
python3 test/moe/test_submission_launcher.py
make -C lib/moe_ffn -j2
make -C module/bdev/moe -j2
make -C examples/moe/tgt -j2
make -C test/moe/pipeline -j2
python3 test/moe/test_moe_pipeline.py
```

小规模回归覆盖正常推理、严格profile拒绝、重复输入命中、逐位编码布局、慢I/O、
截断/移除/执行中关闭、短权重、预热永久失败，以及读失败/分配失败触发一次降槽。
客户端结果与冻结参考独立比较。正式尺寸准确性、P99/P99.99及完整内存验收另行收尾。
