/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_REQUEST_H
#define MOE_REQUEST_H

#define MOE_REQUEST_MAX_K 8

struct moe_request {
	struct spdk_bdev_io *io;
	uint64_t trace_id;
	uint64_t wait_trace;
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
	bool fifo_queued;
	unsigned matrix_done[MOE_REQUEST_MAX_K];
	unsigned tile_ready[MOE_REQUEST_MAX_K][3], tile_done[MOE_REQUEST_MAX_K][3];
	unsigned tile_first[MOE_REQUEST_MAX_K], tile_count[MOE_REQUEST_MAX_K];
	bool tile_hit[MOE_REQUEST_MAX_K];
	unsigned tile_jobs;
	uint64_t start, route_ticks, expert_ticks, combine_ticks;
	uint64_t weight_wait_ticks, weight_wait_start;
};

#endif
