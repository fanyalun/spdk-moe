# 更高压缩比的可逆编码离线实验

`test/moe/research_codec_variants.py`使用已有权重文件，调用原生C编码及重排辅助库，比较11种表示，另加split的AVX2合并实现。只读取原始权重，不运行推理target，不接管或写入原生NVMe，不修改生产编码格式、默认配置或正式submission补丁。

## 候选表示

- baseline：当前1MiB块lattice编码，解码不做checksum。
- lattice_zstd1/9：当前编码载荷再做Zstd级别1/9压缩。
- fp32_byte_zstd9：FP32字节重排后Zstd压缩。
- integer_byte_zstd9：当前精确缩放得到的整数减基值，按字节重排后Zstd压缩。
- integer_bit_zstd9、integer_bit_lz4：整数按位平面重排后压缩。
- split：按实际FP32符号位稳定分为两组，各组独立执行原lattice编码，保存符号位图及两份块描述；解码后按位图合并。没有丢弃数值或低位，不依赖权重种子。
- split_fast：与split相同编码字节，使用AVX2置换和选择合并两组结果；无AVX2时回退原合并。两份预分配恢复缓冲各有至少8个float的尾部空间供向量读取，工作集计入实验内存。
- split_zstd9：对split载荷追加Zstd压缩。
- sub1024/4096：在1MiB逻辑块内，每1024/4096个元素独立编码并保存描述。

所有候选均以恢复出的FP32原始字节与输入直接比较。单元自检覆盖符号零、NaN载荷、Inf、原始回退、非整块尾部、单符号组和64列打包映射；整数转换仅用于原lattice已确认可精确恢复的块，其他块回退baseline。

## 构建和运行

使用系统现有LZ4/Zstd共享库、C编译器及Python标准库，无需numpy或Python压缩扩展。增量构建只生成实验共享库。

```sh
cc -D_GNU_SOURCE -O3 -fno-fast-math -g -fPIC -shared -Iinclude \
  test/moe/research_codec_variants.c -lm -o /tmp/lib_codec_variants.so
python3 test/moe/research_codec_variants.py --weights /tmp/moe_weights \
  --library /tmp/lib_codec_variants.so --experts 0,128,255 \
  --output /tmp/codec_variants_sample.json
```

输出路径必须尚不存在。默认CPU24；使用 `--cpu`指定可用核心。正式尺寸默认为D=2048、H=7168，实际矩阵按现有64列布局打包，每个逻辑块最多1MiB。可用 `--experts all --variants baseline,split_fast`对胜出方案做256专家全量验证。

报告包含原始/编码字节、512B和4KiB对齐后的读取量估算、索引预算、编码及恢复时间、逐位一致的块数、原始打包载荷SHA256。baseline索引按24字节/块计，候选保守按64字节/块计；子块的描述和16字节尾部保护计入载荷。索引总量单独对齐，不假设每个索引占一个扇区。

仅统计专家矩阵载荷和索引，不含共同的Router或固定布局头，也不代表原镜像的逻辑大小减少。64字节索引是离线研究预算，生产接入需要新的格式和索引安排，不能直接套入现有24字节索引。

解码计时包括ctypes调度及原生解码、重排和结果写入，不包含逐字节比较；所有方案复用缓冲区，数据处于热缓存环境。这些吞吐量不能当作正式流水线中的持续解码速度或P99。位平面转换目前为原生标量实现，结果不能代表专门SIMD实现的最高吞吐。

## 运行状态

隔离目录编译及UBSan自检已通过。第一轮用专家0/128/255的Gate/Up/Down全部504个逻辑块比较11种方案，均逐位恢复一致。split的512B对齐占比76.6107%，优于baseline的78.1273%，热缓冲恢复速度约1.50GB/s；直接追加Zstd、FP32字节重排和子块方案没有更好压缩比，整数位平面Zstd仅有较小收益。

第二轮同一新共享库复测baseline、split和split_fast，同样504个块，压缩量与第一轮一致。热缓冲恢复速度分别约10.55、1.79和4.01GB/s。

## 256专家全量结果

全部43,008块、45,097,156,608字节原始矩阵中，两种方案每块恢复后的字节均与原始打包输入相同。原始载荷SHA256为 `e7cad018b70e48c959d962d1f82a64a3bd98a747968317afeb95e03c096c17ba`，与此前全量基线一致。当前方案位宽分布仍为43,005个25位块和3个26位块。

| 表示 | 512B对齐含索引 | 4KiB对齐含索引 | 热缓冲恢复速度 |
|---|---:|---:|---:|
| 当前lattice | 35,233,284,096 bytes，78.1275% | 35,233,284,096 bytes，78.1275% | 10.63GB/s |
| split_fast | 34,548,808,704 bytes，76.6097% | 34,678,169,600 bytes，76.8966% | 3.99GB/s |

512B对齐时比当前方案少684,475,392字节，减少1.9427%；4KiB对齐时少555,114,496字节，减少1.5755%。分组载荷本身约24.501位/权重，已经包含符号位图和两组尾部保护字节；逐块LBA对齐和索引会增加实际占比。

全量恢复速度仍比当前方案慢约2.66倍。它多出两组临时恢复及合并的内存访问，不能仅依据少读约2%便断言推理更快。完整实验约519.6秒，包含读取、打包、编码、解码和字节比较。全量共享库SHA256为 `d748e844b505f5f044a0d5f2bb071e59b2ba9b6bf533015d4d36aea9c4334435`，没有启用checksum或近似浮点恢复。

结果已独立复核块计数、输入摘要、对齐及索引算术。原始产物位于 `/root/moe-run-logs/baremetal_20261014/codec_variants_results.tar.gz`，独立复核为 `codec_variants_full_audit.json`。格式门禁返回1，日志与前轮既有问题逐字一致；`git diff --check`通过，未为此安装依赖或修无关文件。

本次确认可以实现更高压缩比，但未接入生产、未运行实际推理P99或正式尺寸的独立准确性initiator，默认仍使用原lattice编码。实验没有恢复原生NVMe参数搜索。
