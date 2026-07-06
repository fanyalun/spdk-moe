/*
 * test_utils.c — 测试工具模块实现
 *
 * 提供向量内容打印和统计摘要功能，用于测试程序的输出展示。
 */

#include "test_utils.h"
#include <stdio.h>
#include <float.h>

void print_vector(const float* vec, int len, int head, int tail, const char* name) {
    printf("%s [%d]: ", name, len);

    // 当向量足够短时，打印全部元素
    if (len <= head + tail) {
        printf("[");
        for (int i = 0; i < len; i++) {
            printf("%.6f", vec[i]);
            if (i < len - 1) {
                printf(", ");
            }
        }
        printf("]\n");
        return;
    }

    // 向量较长时，打印前 head 个，省略号，后 tail 个
    printf("[");

    // 前 head 个元素
    for (int i = 0; i < head; i++) {
        printf("%.6f", vec[i]);
        if (i < head - 1) {
            printf(", ");
        }
    }

    printf(", ... , ");

    // 后 tail 个元素
    for (int i = len - tail; i < len; i++) {
        printf("%.6f", vec[i]);
        if (i < len - 1) {
            printf(", ");
        }
    }

    printf("]\n");
}

void print_stats(const float* vec, int len, const char* name) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    double sum = 0.0;

    for (int i = 0; i < len; i++) {
        float val = vec[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += (double)val;
    }

    double mean = sum / (double)len;

    printf("%s 统计摘要:\n", name);
    printf("  min  = %.6f\n", min_val);
    printf("  max  = %.6f\n", max_val);
    printf("  mean = %.6f\n", (float)mean);
}
