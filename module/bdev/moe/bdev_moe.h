/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef BDEV_MOE_H
#define BDEV_MOE_H

#include "spdk/bdev.h"
#include "spdk/bdev_module.h"

struct moe_create_opts {
	const char *name, *backend, *base_bdev, *weight_dir, *kernel, *diagnostics;
	int d_model, d_ff, num_experts, top_k, cache_slots;
	unsigned io_size, io_depth, prefetch;
	int compute_cpu;
	int compute_threads;
};

typedef void (*moe_create_done)(void *arg, int status);
int bdev_moe_create_async(const struct moe_create_opts *opts, moe_create_done done, void *arg);
void bdev_moe_delete(const char *name, spdk_bdev_unregister_cb cb, void *arg);
int bdev_moe_initialize(void);
void bdev_moe_finish(void);

#endif
