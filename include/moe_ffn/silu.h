/*
 * silu.h — SiLU（Sigmoid Linear Unit）激活函数模块
 *
 * 提供标量 SiLU 激活函数：silu(z) = z * sigmoid(z) = z / (1 + e^{-z})
 * 本模块不依赖项目中其他任何模块。
 */

#ifndef SILU_H
#define SILU_H

/*
 * 计算单个标量的 SiLU 激活值
 *
 * 参数：
 *   z — 输入标量值
 *
 * 返回：
 *   silu(z) = z / (1 + exp(-z))
 *
 * 数学背景：
 *   SiLU 也被称为 Swish 激活函数，由 Google Brain 在 2017 年提出。
 *   其公式为 silu(x) = x * σ(x)，其中 σ 是 sigmoid 函数。
 *   相比 ReLU，SiLU 在原点附近是光滑的，且保留了负值区域的小幅非零输出，
 *   这在训练深层网络时有助于梯度流动。
 */
float silu(float z);

#endif /* SILU_H */
