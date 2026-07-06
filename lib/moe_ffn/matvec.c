/*
 * matvec.c — 矩阵-向量乘法实现
 *
 * 实现朴素的双层循环矩阵-向量乘法：y = x @ W
 * 不进行任何访存优化（如循环融合、缓存分块、SIMD 等）。
 */

#include "matvec.h"

void matvec_mul(const float* x, const float* W, int in_dim, int out_dim, float* y) {
    /*
     * 计算 y = x @ W
     *
     * 其中：
     *   x 形状 (in_dim,)
     *   W 形状 (in_dim, out_dim)，行主序
     *   y 形状 (out_dim,)
     *
     * 行主序索引映射：W[i][j] = W[i * out_dim + j]
     *
     * 外层循环遍历输出维度 j，内层循环遍历输入维度 i 做点积累加。
     * 这是最朴素的实现方式，有意保留所有可优化空间。
     */

    // 外层循环：遍历输出维度的每个元素
    for (int j = 0; j < out_dim; j++) {
        // 初始化为 0，准备累加
        float sum = 0.0f;

        // 内层循环：遍历输入维度，计算点积
        for (int i = 0; i < in_dim; i++) {
            // W[i * out_dim + j]：第 i 行第 j 列的元素
            sum += x[i] * W[i * out_dim + j];
        }

        // 存储计算结果
        y[j] = sum;
    }
}
