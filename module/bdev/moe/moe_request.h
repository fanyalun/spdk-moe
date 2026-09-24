/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_REQUEST_H
#define MOE_REQUEST_H

#define MOE_REQUEST_MAX_K 8

struct moe_request {
	struct spdk_bdev_io *io;
	int slots[MOE_REQUEST_MAX_K];
	bool computed[MOE_REQUEST_MAX_K];
	unsigned ready[MOE_REQUEST_MAX_K], progress[MOE_REQUEST_MAX_K];
	bool busy[MOE_REQUEST_MAX_K], reading[MOE_REQUEST_MAX_K];
	bool failed[MOE_REQUEST_MAX_K];
	int stage[MOE_REQUEST_MAX_K];
	uint64_t first_compute[MOE_REQUEST_MAX_K];
	uint64_t read_submit[MOE_REQUEST_MAX_K][3];
	uint64_t read_start[MOE_REQUEST_MAX_K][3], read_end[MOE_REQUEST_MAX_K][3];
	uint64_t compute_start[MOE_REQUEST_MAX_K][3], compute_end[MOE_REQUEST_MAX_K][3];
	int completed, loading, status;
	bool routed;
	uint64_t start, route_ticks, expert_ticks, combine_ticks;
	uint64_t weight_wait_ticks, weight_wait_start;
};

#endif
