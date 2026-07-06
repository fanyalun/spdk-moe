#ifndef BDEV_MOE_H
#define BDEV_MOE_H

#include "spdk/bdev.h"

struct spdk_bdev *bdev_moe_create(const char *name,
				   const float *W_router,
				   const float *W_gate[],
				   const float *W_up[],
				   const float *W_down[],
				   int num_experts,
				   int d_model,
				   int d_ff);

int bdev_moe_initialize(void);
void bdev_moe_finish(void);

#endif /* BDEV_MOE_H */
