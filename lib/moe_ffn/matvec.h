/*
 * matvec.h — 矩阵-向量乘法模块
 *
 * 提供朴素的矩阵-向量乘法：y = x @ W
 * 其中 x 为输入向量，W 为行主序存储的权重矩阵。
 * 本模块独立于 SwiGLU FFN，可作为通用工具函数使用。
 */

#ifndef MATVEC_H
#define MATVEC_H

/*
 * 计算矩阵-向量乘法：y = x @ W
 *
 * 参数：
 *   x       — 输入向量，形状 (in_dim,)，只读
 *   W       — 权重矩阵，形状 (in_dim, out_dim)，行主序存储，只读
 *   in_dim  — 输入维度
 *   out_dim — 输出维度
 *   y       — 输出向量，形状 (out_dim,)，只写，由调用者分配
 *
 * 行主序存储说明（以 in_dim=2, out_dim=3 为例）：
 *   W 的逻辑布局：              W 的内存布局：
 *   [ W[0][0] W[0][1] W[0][2] ] → W[0], W[1], W[2], W[3], W[4], W[5]
 *   [ W[1][0] W[1][1] W[1][2] ]      第0行→      第1行→
 *
 *   索引映射：W[i][j] = W[i * out_dim + j]
 *
 * 计算方式（朴素双层循环，不进行任何访存优化）：
 *   对于每个输出维度 j：
 *     y[j] = Σ_{i=0}^{in_dim-1} x[i] * W[i * out_dim + j]
 */
void matvec_mul(const float* x, const float* W, int in_dim, int out_dim, float* y);

#endif /* MATVEC_H */
