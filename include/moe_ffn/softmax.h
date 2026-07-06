/*
 * softmax.h — Softmax 归一化模块
 *
 * 提供朴素的 softmax 归一化函数：softmax(x)_i = e^{x_i} / Σ_j e^{x_j}
 * 对输入数组原地执行 softmax 变换。
 * 本模块不依赖项目中其他任何模块。
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

/*
 * 对数组执行 softmax 归一化，结果原地写回
 *
 * 参数：
 *   x — 输入/输出数组，长度 n，只读写（原地修改）
 *   n — 数组长度
 *
 * 计算：
 *   softmax(x)_i = e^{x_i} / Σ_{j=0}^{n-1} e^{x_j}
 *
 * 实现说明：
 *   1. 先计算所有 expf(x[i])，存入临时数组
 *   2. 累加求所有 expf 结果的和
 *   3. 逐元素除以和，写回 x
 *
 * 朴素实现，不做数值稳定性优化（如减去 max 偏移），
 * 有意保留可优化空间。
 */
void softmax(float* x, int n);

#endif /* SOFTMAX_H */
