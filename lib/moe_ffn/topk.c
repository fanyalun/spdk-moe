#include "topk.h"

void topk_select(const float *values, int n, int k, int *indices, float *out_values)
{
    int count = k < n ? k : n;
    if (count <= 0) {
        return;
    }
    for (int i = 0; i < n; i++) {
        int pos = i < count ? i : count;
        // Strict comparison preserves the original index order for equal scores.
        while (pos > 0 && out_values[pos - 1] < values[i]) {
            if (pos < count) {
                out_values[pos] = out_values[pos - 1];
                indices[pos] = indices[pos - 1];
            }
            pos--;
        }
        if (pos < count) {
            out_values[pos] = values[i];
            indices[pos] = i;
        }
    }
}
