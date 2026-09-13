/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_REQUEST_H
#define MOE_REQUEST_H

#define MOE_REQUEST_MAX_K 8

struct moe_request {
	struct spdk_bdev_io *io;
	int slots[MOE_REQUEST_MAX_K];
	bool computed[MOE_REQUEST_MAX_K];
	int completed, loading, status;
	bool routed;
	uint64_t start, route_ticks, expert_ticks, combine_ticks;
	uint64_t weight_wait_ticks, weight_wait_start;
};

#endif
