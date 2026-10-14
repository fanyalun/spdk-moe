/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_STORE_H
#define MOE_STORE_H

#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "spdk/thread.h"
#include <pthread.h>
#include <stdatomic.h>
#include "moe_codec.h"

#define MOE_STORE_MIB (1024ULL * 1024)
#define MOE_STORE_MAX_IO 64
#define MOE_STORE_MAX_READS 24
#define MOE_STORE_MAX_DECODE_THREADS 4
#define MOE_STORE_MAX_OPS (2 * MOE_STORE_MAX_IO)

struct moe_layout {
	uint64_t magic;
	uint32_t version, complete, d_model, d_ff, experts, pack_cols;
	uint64_t router_offset, router_bytes, expert_offset, expert_stride;
	uint64_t gate_bytes, down_bytes, total_bytes;
};

struct moe_store;
typedef void (*moe_store_done)(void *arg, int status);
typedef void (*moe_store_progress)(void *arg, uint64_t ready_bytes);

struct moe_matrix_read {
	int expert;
	unsigned matrix;
	void *buffer;
	moe_store_done done;
	void *arg;
	uint64_t *first_submit_ticks;
	moe_store_progress progress;
};

struct moe_store_read {
	struct moe_store *store;
	void *buffer;
	uint64_t offset, next, length;
	unsigned codec_index;
	uint64_t *first_submit_ticks;
	unsigned outstanding;
	int status;
	bool active;
	moe_store_done done;
	void *arg;
	moe_store_progress progress;
	uint64_t ready_mask, ready_bytes;
};

struct moe_store_op {
	struct moe_store_read *read;
	uint64_t trace_id;
	uint64_t offset, length;
	struct spdk_bdev_io_wait_entry wait;
	bool submitted;
	bool io_pending;
	bool decoding;
	int decode_status;
	uint64_t decode_ticks;
	void *codec_buffer;
	struct moe_codec_block *codec;
};

struct moe_decode_worker {
	struct moe_store *store;
	pthread_t thread;
	unsigned index;
};

struct moe_store {
	struct spdk_bdev_desc *desc;
	struct spdk_io_channel *channel;
	struct spdk_thread *owner;
	struct moe_layout layout;
	char *directory;
	float *router;
	unsigned io_size, io_depth, active_io, peak_io;
	unsigned pending_io, op_count;
	unsigned decode_threads, decode_started;
	unsigned decode_pending, decode_peak;
	struct moe_decode_worker decoders[MOE_STORE_MAX_DECODE_THREADS];
	pthread_mutex_t decode_mutex;
	pthread_cond_t decode_condition;
	struct moe_store_op *decode_queue[MOE_STORE_MAX_OPS];
	unsigned decode_head, decode_tail, decode_count;
	bool decode_stop;
	uint64_t read_bytes;
	uint64_t submit_count, resource_wait_count;
	uint64_t meter_ticks, busy_ticks, depth_ticks, pending_idle_ticks;
	unsigned meter_active;
	bool diagnostics, meter_pending;
	atomic_bool removed;
	bool pumping, repump;
	bool cancel_batch_on_error;
	bool require_direct_source;
	bool lattice;
	struct moe_codec_block *codec_blocks;
	unsigned codec_gate_blocks, codec_expert_blocks;
	void *encode_buffer;
	uint64_t decoded_bytes, decode_ticks;
	void (*removed_cb)(void *);
	void *removed_arg;
	struct moe_store_read reads[MOE_STORE_MAX_READS];
	struct moe_store_op ops[MOE_STORE_MAX_OPS];
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
		   unsigned io_depth, bool lattice, moe_store_done ready, void *arg);
int moe_store_set_decode_threads(struct moe_store *store, unsigned count);
void *moe_store_decode_worker(void *arg);
int moe_store_read_expert(struct moe_store *store, int expert, void *buffer,
			  moe_store_done done, void *arg);
int moe_store_read_matrix(struct moe_store *store, int expert, unsigned matrix,
                          void *buffer, moe_store_done done, void *arg, uint64_t *first_submit_ticks);
int moe_store_read_batch(struct moe_store *store, const struct moe_matrix_read *reads,
			unsigned count);
void moe_store_cancel_reads(struct moe_store *store, int status);
void moe_store_close(struct moe_store *store);

#endif
