/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_STORE_H
#define MOE_STORE_H

#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "spdk/thread.h"
#include <pthread.h>
#include <stdatomic.h>

#define MOE_STORE_MIB (1024ULL * 1024)
#define MOE_STORE_MAX_IO 16
#define MOE_STORE_MAX_READS 8

struct moe_layout {
	uint64_t magic;
	uint32_t version, complete, d_model, d_ff, experts, pack_cols;
	uint64_t router_offset, router_bytes, expert_offset, expert_stride;
	uint64_t gate_bytes, down_bytes, total_bytes;
};

struct moe_store;
typedef void (*moe_store_done)(void *arg, int status);

struct moe_store_read {
	struct moe_store *store;
	void *buffer;
	uint64_t offset, next;
	unsigned outstanding;
	int status;
	bool active;
	moe_store_done done;
	void *arg;
};

struct moe_store_op {
	struct moe_store_read *read;
	uint64_t offset, length;
	struct spdk_bdev_io_wait_entry wait;
	bool submitted;
};

struct moe_store {
	struct spdk_bdev_desc *desc;
	struct spdk_io_channel *channel;
	struct spdk_thread *owner;
	struct moe_layout layout;
	char *directory;
	float *router;
	unsigned io_size, io_depth, active_io, peak_io;
	uint64_t read_bytes;
	atomic_bool removed;
	bool pumping, repump;
	void (*removed_cb)(void *);
	void *removed_arg;
	struct moe_store_read reads[MOE_STORE_MAX_READS];
	struct moe_store_op ops[MOE_STORE_MAX_IO];
	pthread_t importer;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	bool import_started, import_waiting;
	int import_status;
	void *import_buffer;
	uint64_t import_offset, import_length;
	bool import_flush;
	struct spdk_bdev_io_wait_entry import_wait;
	moe_store_done ready;
	void *ready_arg;
};

int moe_layout_init(struct moe_layout *layout, int d, int h, int experts, uint32_t block_size);
int moe_store_open(struct moe_store *store, const char *base, const char *directory,
		   int d, int h, int experts, bool strict_nvme, unsigned io_size,
		   unsigned io_depth, moe_store_done ready, void *arg);
int moe_store_read_expert(struct moe_store *store, int expert, void *buffer,
			  moe_store_done done, void *arg);
void moe_store_close(struct moe_store *store);

#endif
