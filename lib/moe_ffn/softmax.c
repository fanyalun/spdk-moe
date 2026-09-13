#include "softmax.h"
#include <math.h>

void softmax(float *x, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}
