# MoE 异步存储与计算

此版本使用显式 `bdev_moe_create` RPC；模块初始化不再自动创建 MoE bdev。
默认 FP32、AVX2/FMA、7 槽位、1 个持久计算线程、1MiB 读取、4 个应用层 I/O、最多 2 个专家换入。
`moe_bench` 的协议、时钟、统计和输出未修改。

## 构建和小规模验证

保留当前 SPDK configure 参数，不需要新增依赖。先在相邻 `MoE-compute` 按 `docs/validation.md` 构建冻结参考库。

```sh
make -C lib/moe_ffn -j 4
make -C module/bdev/moe -j 4
make -C module/bdev/aio -j 4
make -C examples/moe/tgt -j 4
make -C test/moe/pipeline -j 4
python3 test/moe/test_moe_pipeline.py
```

测试创建独立临时目录，保留权重、AIO 文件、JSON 和日志，不触碰 `/tmp/moe_weights`。
测试程序只在验证时链接冻结参考库，生产 target 不链接它。

## 原始文件开发后端

```sh
build/examples/moe_tgt --no-huge --no-pci -s 512 -m 0x1 \
  -c examples/moe/moe_tgt.json -r /tmp/moe.sock
```

文件后端使用普通内存槽位，文件读取和计算在工作线程串行执行，仅用于计算调试。
不要将 `-s 1536` 的 DMA 池再叠加文件后端 1176MiB 普通权重内存用于正式 2GB 验收。

## AIO 与 NVMe 后端

在 JSON 的 bdev config 中先创建 base，再创建 MoE。AIO 的示例参数：

```json
{"method":"bdev_aio_create","params":{"name":"weights","filename":"/tmp/moe_packed.bin","block_size":512}}
{"method":"bdev_moe_create","params":{"name":"moe_bdev_0","backend":"aio","base_bdev":"weights","weight_dir":"/tmp/moe_weights","cache_slots":7,"io_size":1048576,"io_depth":4,"prefetch":2}}
```

以上两项是 config 数组中的相邻元素。随后按原 `moe_tgt.json` 创建 NVMe-oF namespace。
AIO 是文件模拟后端。启动日志与 `bdev_get_bdevs` 的 `driver_specific.aio.direct_io` 字段显示实际 O_DIRECT 模式；false 时不得报告直接 I/O 性能。

NVMe 先用 SPDK 的 `bdev_nvme_attach_controller` 创建**已获授权的独占设备**，再设置 `backend:"nvme"` 与其 namespace bdev 名称。严格模式检查模块名为 nvme，拒绝文件或 AIO；不会自动解绑、TRIM 或选择设备。

AIO/NVMe 权重槽使用 SPDK DMA 分配，必须包含在总池中；正式起点 `-s 1536`，普通内存目标不超过 192MiB，总量预警 1800MiB。
默认 target 使用 hugepages，文件模拟开发时显式 `--no-huge --no-pci`。内存不足会返回错误，不扩大池。
`bdev_set_options` 的池配置也应保留。真实 hugepage/cgroup 记账仍须在独占环境验收。

## 导入与布局

版本 1：首个 1MiB 为布局头，含维度、FP32 64 列块宽、各区域偏移、总长度和完成标记；Router 保持行主序，专家 Gate/Up/Down 使用打包布局。
区域按 1MiB 对齐，矩阵按 base 的逻辑块对齐。默认 2048×7168、256 专家需要 45100302336 字节。

每次启动重新导入当前目录；先写未完成标记并 flush，再逐矩阵导入，数据 flush 后写完成标记并再次 flush。RPC 只有全流程成功才响应成功。
一个普通对齐矩阵缓冲加一个 1MiB DMA 缓冲，总导入工作缓冲不超过 64MiB。原始文件短读、额外字节、维度不符或写入失败都会拒绝发布服务。
创建打包文件前检查剩余空间，分配后至少保留 10GiB；只保留一份正式打包文件。

## 并发与故障语义

Reactor 持有描述符、I/O channel、缓存状态及请求完成权；输入复制后异步执行。计算线程只处理当前工作区和已就绪权重。
缓存状态为 EMPTY/LOADING/READY/IN_USE，固定缓冲原位覆盖。先保护请求已有命中，再选择无引用的 LRU 槽位。
7 槽位不会等待 Top-8 同时驻留：先计算并释放槽，再换入剩余专家。输出按原 Top-K 顺序以 FMA 合成。

每个 MoE bdev 同时只接受一个推理请求；竞争请求失败。结果未就绪或上一请求失败时 READ 失败。初始化阶段的探测 READ 返回零。
缺少 bdev I/O 资源时注册等待回调，不忙等。读取失败后排空已提交任务，设备移除触发上层卸载，退出时释放 I/O 后再释放槽位。

计算线程默认优先选择与 reactor 同 NUMA 节点的另一物理核，避开 SPDK 配置的 reactor 及其 SMT 同胞；可通过 `compute_cpu` 显式指定独立物理 CPU，避免与 initiator 冲突。进程仅有一个可用物理核时启动失败。
AIO/NVMe 可设置 `compute_threads` 为 1、2、4；额外线程各有工作区，按专家分工，最后仍按原顺序汇总。文件调试后端只接受一个线程。多线程时参照启动日志选择未被占用的 initiator CPU，例如本机四工作线程用 CPU 1–4 时将 benchmark 放到 CPU 5。

## 诊断和发布门槛

可选 `diagnostics:"/path/stats.jsonl"` 输出独立诊断；默认关闭。记录路由、专家、汇总时间，累计命中/未命中、实际成功读取字节和 I/O 峰值。
文件后端读取时间与专家计算时间分开记录；AIO 后端计算与读取会重叠，不能简单把阶段时间相加当总时间。
`kernel` 可显式选 scalar、avx2、avx512；不支持的指令集启动失败。

准确性、资源故障、正式大小、内存和真实 NVMe 验证应分别记录。小规模或文件缓存测量不构成正式 P99/P99.99 结论；尚不承诺官方评分或加速倍数。
