/*
 * silu.c — SiLU（Sigmoid Linear Unit）激活函数实现
 *
 * 实现标量 SiLU 激活函数：silu(z) = z / (1 + e^{-z})
 */

#include "silu.h"
#include <math.h>

float silu(float z) {
    /*
     * SiLU 公式推导：
     *   silu(z) = z * σ(z)
     *           = z * (1 / (1 + e^{-z}))
     *           = z / (1 + e^{-z})
     *
     * 直接使用公式 z / (1 + exp(-z)) 避免先算 sigmoid 再乘 z。
     */
    return z / (1.0f + expf(-z));
}
