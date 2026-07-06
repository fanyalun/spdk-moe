/*
 * softmax.c — Softmax 归一化实现
 *
 * 实现朴素的 softmax 归一化：
 *   1. 计算所有 expf(x[i])，存入临时数组
 *   2. 累加求和
 *   3. 逐元素除以和，写回 x
 *
 * 不做数值稳定性优化（如减去 max 偏移），有意保留可优化空间。
 * 对于 [-1, 1) 均匀分布的 logits，数值范围在安全区域内。
 */

#include "softmax.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

void softmax(float* x, int n) {
    /*
     * softmax 数学定义：
     *   softmax(x)_i = e^{x_i} / Σ_{j=0}^{n-1} e^{x_j}
     *
     * 两步走：
     *   第 1 步：计算所有指数值并求和
     *   第 2 步：逐元素除以和
     */

    // 分配临时数组存放 expf(x[i])
    float* exp_vals = (float*)malloc((size_t)n * sizeof(float));
    assert(exp_vals != NULL);

    // 第 1 步：计算所有 expf(x[i]) 并累加求和
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        exp_vals[i] = expf(x[i]);
        sum += exp_vals[i];
    }

    // 第 2 步：逐元素除以和，结果写回 x
    for (int i = 0; i < n; i++) {
        x[i] = exp_vals[i] / sum;
    }

    // 释放临时数组
    free(exp_vals);
}
