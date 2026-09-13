/* SPDX-License-Identifier: BSD-3-Clause */

#include "matvec_packed.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MOE_X86 1
#endif

int moe_kernel_supported(enum moe_kernel kernel)
{
    if (kernel == MOE_KERNEL_SCALAR) {
        return 1;
    }
#ifdef MOE_X86
    __builtin_cpu_init();
    if (kernel == MOE_KERNEL_AVX2) {
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    if (kernel == MOE_KERNEL_AVX512) {
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
    }
#endif
    return 0;
}

const char *moe_kernel_name(enum moe_kernel kernel)
{
    switch (kernel) {
    case MOE_KERNEL_SCALAR: return "scalar";
    case MOE_KERNEL_AVX2: return "avx2";
    case MOE_KERNEL_AVX512: return "avx512";
    default: return "invalid";
    }
}

int moe_kernel_parse(const char *name, enum moe_kernel *kernel)
{
    if (!kernel) {
        return -1;
    }
    if (!name) {
        *kernel = moe_kernel_supported(MOE_KERNEL_AVX2) ? MOE_KERNEL_AVX2 : MOE_KERNEL_SCALAR;
        return 0;
    }
    for (int k = MOE_KERNEL_SCALAR; k <= MOE_KERNEL_AVX512; k++) {
        if (strcmp(name, moe_kernel_name(k)) == 0 && moe_kernel_supported(k)) {
            *kernel = k;
            return 0;
        }
    }
    return -1;
}

size_t matvec_packed_elements(int in_dim, int out_dim)
{
    if (in_dim <= 0 || out_dim <= 0) {
        return 0;
    }
    size_t cols = ((size_t)out_dim + MOE_PACK_COLS - 1) / MOE_PACK_COLS * MOE_PACK_COLS;
    if (cols > SIZE_MAX / sizeof(float) / (size_t)in_dim) {
        return 0;
    }
    return cols * (size_t)in_dim;
}

int matvec_pack(const float *row, int in_dim, int out_dim, float *packed)
{
    size_t count = matvec_packed_elements(in_dim, out_dim);
    if (!row || !packed || !count) {
        return -1;
    }
    memset(packed, 0, count * sizeof(float));
    for (size_t j = 0; j < (size_t)out_dim; j += MOE_PACK_COLS) {
        size_t width = (size_t)out_dim - j;
        if (width > MOE_PACK_COLS) {
            width = MOE_PACK_COLS;
        }
        for (int i = 0; i < in_dim; i++) {
            memcpy(packed + j * (size_t)in_dim + (size_t)i * MOE_PACK_COLS,
                   row + (size_t)i * out_dim + j, width * sizeof(float));
        }
    }
    return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void scalar_block(const float *x, const float *w, int rows, size_t stride,
                         int width, float *y)
{
    for (int j = 0; j < width; j++) {
        y[j] = 0.0f;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < width; j++) {
            y[j] = fmaf(x[i], w[(size_t)i * stride + j], y[j]);
        }
    }
}

#ifdef MOE_X86
__attribute__((target("avx2,fma")))
static void avx2_block(const float *x, const float *w, int rows, size_t stride, float *y)
{
    __m256 accum[8];
#pragma GCC unroll 8
    for (int v = 0; v < 8; v++) {
        accum[v] = _mm256_setzero_ps();
    }
    for (int i = 0; i < rows; i++) {
        __m256 input = _mm256_set1_ps(x[i]);
#pragma GCC unroll 8
        for (int v = 0; v < 8; v++) {
            accum[v] = _mm256_fmadd_ps(input, _mm256_loadu_ps(w + (size_t)i * stride + v * 8),
                                       accum[v]);
        }
    }
#pragma GCC unroll 8
    for (int v = 0; v < 8; v++) {
        _mm256_storeu_ps(y + v * 8, accum[v]);
    }
}

__attribute__((target("avx512f,fma")))
static void avx512_block(const float *x, const float *w, int rows, size_t stride, float *y)
{
    __m512 accum[4];
#pragma GCC unroll 4
    for (int v = 0; v < 4; v++) {
        accum[v] = _mm512_setzero_ps();
    }
    for (int i = 0; i < rows; i++) {
        __m512 input = _mm512_set1_ps(x[i]);
#pragma GCC unroll 4
        for (int v = 0; v < 4; v++) {
            accum[v] = _mm512_fmadd_ps(input, _mm512_loadu_ps(w + (size_t)i * stride + v * 16),
                                       accum[v]);
        }
    }
#pragma GCC unroll 4
    for (int v = 0; v < 4; v++) {
        _mm512_storeu_ps(y + v * 16, accum[v]);
    }
}
#endif

void matvec_ordered(const float *x, const float *w, int in_dim, int out_dim,
                    float *y, enum moe_kernel kernel, int packed)
{
    if (in_dim < 0 || out_dim <= 0) {
        return;
    }
    for (size_t j = 0; j < (size_t)out_dim; j += MOE_PACK_COLS) {
        int width = (size_t)out_dim - j < MOE_PACK_COLS ? (int)((size_t)out_dim - j) : MOE_PACK_COLS;
        size_t stride = packed ? MOE_PACK_COLS : (size_t)out_dim;
        const float *block = w + (packed ? j * (size_t)in_dim : j);
        float tail[MOE_PACK_COLS];
        float *dest = width < MOE_PACK_COLS ? tail : y + j;
#ifdef MOE_X86
        if ((packed || width == MOE_PACK_COLS) && kernel == MOE_KERNEL_AVX2) {
            avx2_block(x, block, in_dim, stride, dest);
        } else if ((packed || width == MOE_PACK_COLS) && kernel == MOE_KERNEL_AVX512) {
            avx512_block(x, block, in_dim, stride, dest);
        } else
#endif
        {
            scalar_block(x, block, in_dim, stride, width, dest);
        }
        if (width < MOE_PACK_COLS) {
            memcpy(y + j, tail, (size_t)width * sizeof(float));
        }
    }
}
