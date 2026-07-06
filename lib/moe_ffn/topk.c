/*
 * topk.c — Top-k 选择实现
 *
 * 实现朴素的全排序 Top-k 选择：
 *   1. 构造 n 个 (value, index) 对
 *   2. 使用插入排序按 value 降序排列
 *   3. 取前 k 个的 index 和 value
 *
 * 使用插入排序（O(n²)），不使用 qsort()，
 * 有意保留性能优化空间。
 */

#include "topk.h"
#include <stdlib.h>
#include <assert.h>

/*
 * (value, index) 对，用于排序
 */
typedef struct {
    float value;
    int index;
} Pair;

void topk_select(const float* values, int n, int k, int* indices, float* out_values) {
    // 当 k >= n 时，返回全部 n 个元素
    int actual_k = (k < n) ? k : n;

    // 在堆上分配排序所需的临时 Pair 数组
    Pair* pairs = (Pair*)malloc((size_t)n * sizeof(Pair));
    assert(pairs != NULL);

    // 构造 (value, index) 对
    for (int i = 0; i < n; i++) {
        pairs[i].value = values[i];
        pairs[i].index = i;
    }

    // 使用插入排序，按 value 降序排列
    //
    // 插入排序思路：
    //   将数组分为"已排序"和"未排序"两部分，
    //   每次从未排序部分取第一个元素，插入到已排序部分的正确位置。
    for (int i = 1; i < n; i++) {
        Pair key = pairs[i];
        int j = i - 1;

        // 将比 key.value 小的元素向后移动
        // 降序：大的在前，所以找到 key.value < pairs[j].value 的位置
        while (j >= 0 && pairs[j].value < key.value) {
            pairs[j + 1] = pairs[j];
            j--;
        }
        pairs[j + 1] = key;
    }

    // 取前 actual_k 个元素的索引和值
    for (int i = 0; i < actual_k; i++) {
        indices[i] = pairs[i].index;
        out_values[i] = pairs[i].value;
    }

    // 释放临时数组
    free(pairs);
}
