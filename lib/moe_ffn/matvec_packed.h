#ifndef MATVEC_PACKED_H
#define MATVEC_PACKED_H

#include <stddef.h>

#define MOE_PACK_COLS 64

enum moe_kernel { MOE_KERNEL_SCALAR, MOE_KERNEL_AVX2, MOE_KERNEL_AVX512 };

int moe_kernel_supported(enum moe_kernel kernel);
int moe_kernel_parse(const char *name, enum moe_kernel *kernel);
const char *moe_kernel_name(enum moe_kernel kernel);
size_t matvec_packed_elements(int in_dim, int out_dim);
// Non-overlapping buffers; destination holds matvec_packed_elements() floats.
int matvec_pack(const float *row, int in_dim, int out_dim, float *packed);
// Explicit FMA and increasing input order; x, w and y must not overlap.
void matvec_ordered(const float *x, const float *w, int in_dim, int out_dim,
                    float *y, enum moe_kernel kernel, int packed);

#endif
