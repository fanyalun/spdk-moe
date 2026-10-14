/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/json.h"
#include "spdk/nvme_spec.h"
#include "moe_ffn/moe_config.h"
#include "moe_ffn/moe_workspace.h"
#include "moe_ffn/moe_affinity.h"
#include "moe_cache.h"
#include "moe_store.h"
#include "moe_request.h"
#include "bdev_moe.h"
#include "moe_trace.h"

struct moe_bdev;
struct moe_load_context {
	struct moe_bdev *moe;
	int selected, slot, matrix;
};

enum moe_job { MOE_JOB_NONE, MOE_JOB_ROUTE, MOE_JOB_EXPERT, MOE_JOB_COMBINE };

struct moe_bdev {
	struct spdk_bdev bdev;
	struct spdk_thread *owner;
	struct moe_cache cache;
	struct moe_store store;
	struct moe_workspace ws;
	struct moe_request request;
	struct moe_load_context loads[MOE_REQUEST_MAX_K];
	struct moe_load_context matrix_loads[MOE_REQUEST_MAX_K][3];
	float *router, *last_output;
	char *directory;
	bool matrix_fifo;
	unsigned warmup_requests, warmup_done, warmup_retries;
	uint32_t warmup_state;
	float *intermediate;
	bool store_open, output_valid, worker_busy, worker_stop, worker_started;
	bool closing, scheduling, reschedule;
	bool job_ready;
	bool registered;
	bool ever_requested;
	unsigned prefetch;
	int compute_cpu;
	int worker_cpus[1 + MOE_STORE_MAX_DECODE_THREADS];
	pthread_t worker;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	enum moe_job job;
	int selected, worker_rc;
	uint64_t job_start, sequence;
	FILE *diagnostics;
	moe_create_done ready;
	void *ready_arg;
};

struct moe_io_ctx {
	struct spdk_thread *thread;
	enum spdk_bdev_io_status status;
};

static int
bdev_moe_get_ctx_size(void)
{
	return sizeof(struct moe_io_ctx);
}

static struct spdk_bdev_module g_moe_if = {
	.name = "moe",
	.module_init = bdev_moe_initialize,
	.module_fini = bdev_moe_finish,
	.get_ctx_size = bdev_moe_get_ctx_size,
};
SPDK_BDEV_MODULE_REGISTER(moe, &g_moe_if)

static void request_schedule(struct moe_bdev *moe);
static void warmup_next(void *arg);

static void
complete_on_origin(void *arg)
{
	struct spdk_bdev_io *io = arg;
	struct moe_io_ctx *ctx = (void *)io->driver_ctx;

	spdk_bdev_io_complete(io, ctx->status);
}

static void
complete_io(struct spdk_bdev_io *io, int status)
{
	struct moe_io_ctx *ctx = (void *)io->driver_ctx;

	ctx->status = status ? SPDK_BDEV_IO_STATUS_FAILED : SPDK_BDEV_IO_STATUS_SUCCESS;
	if (ctx->thread == spdk_get_thread()) {
		complete_on_origin(io);
	} else {
		spdk_thread_send_msg(ctx->thread, complete_on_origin, io);
	}
}

static void
request_finish(struct moe_bdev *moe)
{
	struct moe_request *r = &moe->request;
	struct spdk_bdev_io *io = r->io;
	if (r->weight_wait_start) {
		r->weight_wait_ticks += spdk_get_ticks() - r->weight_wait_start;
		r->weight_wait_start = 0;
	}

	moe->output_valid = r->status == 0;
	for (int k = 0; k < moe->ws.top_k; k++) {
		if (r->failed[k] && r->slots[k] >= 0) {
			moe->cache.entries[r->slots[k]].state = MOE_CACHE_EMPTY;
		}
	}
	for (int i = 0; i < moe->cache.capacity; i++) {
		if (moe->cache.entries[i].state == MOE_CACHE_LOADING) {
			moe->cache.entries[i].state = MOE_CACHE_EMPTY;
		}
		moe->cache.entries[i].refs = 0;
	}
	if (moe->diagnostics && !r->warmup) {
		fprintf(moe->diagnostics,
			"{\"request\":%" PRIu64 ",\"status\":%d,\"ticks_hz\":%" PRIu64
			",\"total_ticks\":%" PRIu64 ",\"route_ticks\":%" PRIu64
			",\"expert_ticks\":%" PRIu64 ",\"combine_ticks\":%" PRIu64
			",\"weight_wait_ticks\":%" PRIu64
			",\"cache_hits_total\":%" PRIu64 ",\"cache_misses_total\":%" PRIu64
			",\"read_bytes_total\":%" PRIu64 ",\"io_peak\":%u,\"pipeline\":\"%s\",\"stages\":[",
			++moe->sequence, r->status, spdk_get_ticks_hz(), spdk_get_ticks() - r->start,
			r->route_ticks, r->expert_ticks, r->combine_ticks, r->weight_wait_ticks,
			moe->cache.hits, moe->cache.misses,
			moe->store.read_bytes, moe->store.peak_io,
			moe->matrix_fifo ? "matrix_fifo" : "matrix");
		bool comma = false;
		for (int k = 0; k < moe->ws.top_k; k++) {
			for (int stage = 0; stage < 3; stage++) {
				if (!r->read_start[k][stage] && !r->compute_start[k][stage]) {
					continue;
				}
				fprintf(moe->diagnostics,
					"%s{\"selected\":%d,\"stage\":%d,\"read_start\":%" PRIu64
					",\"read_submit\":%" PRIu64 ",\"read_end\":%" PRIu64 ",\"compute_start\":%" PRIu64
					",\"compute_end\":%" PRIu64 "}", comma ? "," : "", k, stage,
					r->read_start[k][stage], r->read_submit[k][stage], r->read_end[k][stage],
					r->compute_start[k][stage], r->compute_end[k][stage]);
				comma = true;
			}
		}
		uint64_t first = 0;
		for (int k = 0; k < moe->ws.top_k; k++) {
			if (r->first_compute[k] && (!first || r->first_compute[k] < first)) {
				first = r->first_compute[k];
			}
		}
		fprintf(moe->diagnostics, "],\"first_compute_wait_ticks\":%" PRIu64
			",\"io_submit_total\":%" PRIu64 ",\"io_resource_wait_total\":%" PRIu64
			",\"io_busy_ticks_total\":%" PRIu64 ",\"io_depth_ticks_total\":%" PRIu64
			",\"io_pending_idle_ticks_total\":%" PRIu64
			",\"decoded_bytes_total\":%" PRIu64 ",\"decode_ticks_total\":%" PRIu64
			",\"decode_threads\":%u,\"decode_peak\":%u,\"decode_pending\":%u"
			",\"codec_buffer_slots\":%u}\n",
			first ? first - r->start : 0, moe->store.submit_count,
			moe->store.resource_wait_count, moe->store.busy_ticks,
			moe->store.depth_ticks, moe->store.pending_idle_ticks,
			moe->store.decoded_bytes, moe->store.decode_ticks,
			moe->store.decode_threads, moe->store.decode_peak,
			moe->store.decode_pending, moe->store.op_count);
	}
	if (r->wait_trace) {
		moe_trace_end(r->wait_trace);
		r->wait_trace = 0;
	}
	moe_trace_end(r->trace_id);
	if (r->warmup) {
		r->warmup = false;
		moe->warmup_done += r->status == 0;
		spdk_thread_send_msg(moe->owner, warmup_next, moe);
		return;
	}
	moe_trace_capture(moe->sequence, false);
	r->io = NULL;
	complete_io(io, r->status);
}

static bool
stage_done(struct moe_bdev *moe, int selected, int status)
{
	struct moe_request *r = &moe->request;

	if (r->stage[selected] < 0) {
		return false;
	}
	r->busy[selected] = false;
	r->failed[selected] |= status != 0;
	if (!status) {
		r->progress[selected]++;
		if (r->progress[selected] == 3) {
			struct moe_cache_entry *e = &moe->cache.entries[r->slots[selected]];
			r->computed[selected] = true;
			r->completed++;
			e->state = MOE_CACHE_READY;
			e->refs = 0;
			e->last_used = ++moe->cache.clock;
		}
	}
	return true;
}

static int
compute_selected(struct moe_bdev *moe, struct moe_workspace *ws, int selected)
{
	struct moe_request *r = &moe->request;
	struct moe_cache_entry *e = &moe->cache.entries[r->slots[selected]];
	int stage = r->stage[selected];
	moe_trace_push("COMPUTE", moe->ws.indices[selected],
		       stage, 0);
	moe_trace_set_expert(moe->ws.indices[selected]);
	if (moe->diagnostics && !r->first_compute[selected]) {
		r->first_compute[selected] = spdk_get_ticks();
	}

	if (stage < 0) {
		int rc = moe_expert(ws, e->w_gate, e->w_up, e->w_down, selected, 1);
		moe_trace_pop();
		return rc;
	}
	float *gate = moe->intermediate + (size_t)selected * 2 * ws->d_ff;
	float *up = gate + ws->d_ff;
	const float *weight = stage == 0 ? e->w_gate : stage == 1 ? e->w_up : e->w_down;
	if (moe->diagnostics && !r->compute_start[selected][stage]) {
		r->compute_start[selected][stage] = spdk_get_ticks();
	}
	float *output = ws->expert_outputs + (size_t)selected * ws->d_model;
	int rc = moe_expert_stage(ws, stage, weight, gate, up, output, 1);
	if (moe->diagnostics) {
		r->compute_end[selected][stage] = spdk_get_ticks();
	}
	moe_trace_pop();
	return rc;
}

static void
worker_done(void *arg)
{
	struct moe_bdev *moe = arg;
	struct moe_request *r = &moe->request;
	enum moe_job job;
	int selected, status;

	pthread_mutex_lock(&moe->mutex);
	job = moe->job;
	selected = moe->selected;
	status = moe->worker_rc;
	moe->job = MOE_JOB_NONE;
	pthread_mutex_unlock(&moe->mutex);
	moe->worker_busy = false;
	uint64_t elapsed = moe->diagnostics ? spdk_get_ticks() - moe->job_start : 0;
	if (status) {
		r->status = status;
	}
	if (job == MOE_JOB_ROUTE) {
		r->routed = !status;
		r->route_ticks = elapsed;
		if (!status) {
			/* Pin every hit before reserving a victim for any miss. */
			for (int k = 0; k < moe->ws.top_k; k++) {
				int slot = moe_cache_find(&moe->cache, moe->ws.indices[k]);
				r->slots[k] = slot;
				if (slot >= 0) {
					moe->cache.entries[slot].refs = 1;
					r->ready[k] = 3;
				}
			}
		}
	} else if (job == MOE_JOB_EXPERT) {
		if (stage_done(moe, selected, status)) {
			r->expert_ticks += elapsed;
			request_schedule(moe);
			return;
		}
		r->busy[selected] = false;
		struct moe_cache_entry *e = &moe->cache.entries[r->slots[selected]];
		e->state = status ? MOE_CACHE_EMPTY : MOE_CACHE_READY;
		e->refs = 0;
		e->last_used = ++moe->cache.clock;
		r->expert_ticks += elapsed;
		r->computed[selected] = true;
		r->completed++;
	} else if (job == MOE_JOB_COMBINE) {
		r->combine_ticks = elapsed;
		request_finish(moe);
		return;
	}
	request_schedule(moe);
}

static void *
compute_worker(void *arg)
{
	struct moe_bdev *moe = arg;

	moe_trace_name_thread("CPU Compute / Router + Experts + Aggregate");
	pthread_mutex_lock(&moe->mutex);
	for (;;) {
		while (!moe->job_ready && !moe->worker_stop) {
			moe_trace_push("WAIT_WORKER", -1, -1, 0);
			pthread_cond_wait(&moe->condition, &moe->mutex);
			moe_trace_pop();
		}
		if (moe->worker_stop) {
			break;
		}
		enum moe_job job = moe->job;
		int selected = moe->selected;
		moe->job_ready = false;
		pthread_mutex_unlock(&moe->mutex);
		int rc = 0;
		if (job == MOE_JOB_ROUTE) {
			moe_trace_push("ROUTE", -1, -1, 0);
			rc = moe_route(&moe->ws, moe->ws.input, moe->router);
			moe_trace_pop();
		} else if (job == MOE_JOB_EXPERT) {
			rc = compute_selected(moe, &moe->ws, selected);
		} else {
			moe_trace_push("COMBINE", -1, -1, 0);
			rc = moe_combine(&moe->ws, moe->last_output);
			moe_trace_pop();
		}
		pthread_mutex_lock(&moe->mutex);
		moe->worker_rc = rc;
		/* Wait for the owner to acknowledge this job before accepting another one. */
		spdk_thread_send_msg(moe->owner, worker_done, moe);
	}
	pthread_mutex_unlock(&moe->mutex);
	return NULL;
}

static void
submit_job(struct moe_bdev *moe, enum moe_job job, int selected)
{
	moe->worker_busy = true;
	moe->job_start = moe->diagnostics ? spdk_get_ticks() : 0;
	pthread_mutex_lock(&moe->mutex);
	moe->job = job;
	moe->selected = selected;
	moe->job_ready = true;
	pthread_cond_signal(&moe->condition);
	pthread_mutex_unlock(&moe->mutex);
}

static bool
submit_expert(struct moe_bdev *moe, int selected)
{
	if (moe->request.wait_trace) {
		moe_trace_end(moe->request.wait_trace);
		moe->request.wait_trace = 0;
	}
	if (moe->request.weight_wait_start) {
		moe->request.weight_wait_ticks += spdk_get_ticks() - moe->request.weight_wait_start;
		moe->request.weight_wait_start = 0;
	}
	if (!moe->worker_busy) {
		submit_job(moe, MOE_JOB_EXPERT, selected);
		return true;
	}
	return false;
}

static void
matrix_loaded(void *arg, int status)
{
	struct moe_load_context *load = arg;
	struct moe_bdev *moe = load->moe;
	struct moe_request *r = &moe->request;
	int k = load->selected;

	r->reading[k] = false;
	if (moe->diagnostics) {
		r->read_end[k][r->ready[k]] = spdk_get_ticks();
	}
	if (status) {
		r->status = status;
		r->loading--;
		r->failed[k] = true;
	} else if (++r->ready[k] == 3) {
		moe->cache.entries[load->slot].state = MOE_CACHE_READY;
		r->loading--;
	}
	request_schedule(moe);
}

static void
matrix_read_next(struct moe_bdev *moe, int k)
{
	struct moe_request *r = &moe->request;
	struct moe_cache_entry *e = &moe->cache.entries[r->slots[k]];
	unsigned stage = r->ready[k];
	void *buffer = stage == 0 ? e->w_gate : stage == 1 ? e->w_up : e->w_down;

	r->reading[k] = true;
	if (moe->diagnostics) {
		r->read_start[k][stage] = spdk_get_ticks();
	}
	int rc = moe_store_read_matrix(&moe->store, moe->ws.indices[k], stage,
				       buffer, matrix_loaded, &moe->loads[k],
				       moe->diagnostics ? &r->read_submit[k][stage] : NULL);
	if (rc) {
		matrix_loaded(&moe->loads[k], rc);
	}
}

static void
fifo_matrix_complete(void *arg, int status)
{
	struct moe_load_context *load = arg;
	struct moe_bdev *moe = load->moe;
	struct moe_request *r = &moe->request;
	int k = load->selected;

	r->loading--;
	if (moe->diagnostics) {
		r->read_end[k][load->matrix] = spdk_get_ticks();
	}
	if (status) {
		r->status = status;
		r->failed[k] = true;
	} else {
		r->matrix_done[k] |= 1u << load->matrix;
		while (r->ready[k] < 3 && (r->matrix_done[k] & (1u << r->ready[k]))) {
			r->ready[k]++;
		}
		if (r->ready[k] == 3) {
			moe->cache.entries[load->slot].state = MOE_CACHE_READY;
		}
	}
	request_schedule(moe);
}

static void
fifo_queue_matrices(struct moe_bdev *moe)
{
	struct moe_request *r = &moe->request;
	struct moe_matrix_read batch[MOE_STORE_MAX_READS];
	unsigned count = 0;

	r->fifo_queued = true;
	/* Hits were pinned by ROUTE. Reserve all misses before starting any read. */
	for (int k = 0; k < moe->ws.top_k; k++) {
		if (r->slots[k] >= 0) {
			continue;
		}
		int slot = moe_cache_reserve(&moe->cache, moe->ws.indices[k]);
		if (slot < 0) {
			r->status = -ENOMEM;
			return;
		}
		r->slots[k] = slot;
	}
	for (int k = 0; k < moe->ws.top_k; k++) {
		if (r->ready[k] == 3) {
			continue;
		}
		int slot = r->slots[k];
		struct moe_cache_entry *e = &moe->cache.entries[slot];
		for (int stage = 0; stage < 3; stage++) {
			struct moe_load_context *load = &moe->matrix_loads[k][stage];
			*load = (struct moe_load_context){moe, k, slot, stage};
			if (moe->diagnostics) {
				r->read_start[k][stage] = spdk_get_ticks();
			}
			batch[count++] = (struct moe_matrix_read){
				.expert = moe->ws.indices[k], .matrix = stage,
				.buffer = stage == 0 ? e->w_gate : stage == 1 ? e->w_up : e->w_down,
				.done = fifo_matrix_complete, .arg = load,
				.first_submit_ticks = moe->diagnostics ? &r->read_submit[k][stage] : NULL,
			};
		}
	}
	r->loading = count;
	int rc = moe_store_read_batch(&moe->store, batch, count);
	if (rc) {
		r->loading = 0;
		r->status = rc;
	}
}

static void
matrix_schedule(struct moe_bdev *moe)
{
	struct moe_request *r = &moe->request;

	if (moe->matrix_fifo && !r->fifo_queued) {
		fifo_queue_matrices(moe);
	}
	for (int k = 0; !moe->matrix_fifo && k < moe->ws.top_k && !r->status; k++) {
		if (r->slots[k] >= 0 && r->ready[k] < 3 && !r->reading[k]) {
			matrix_read_next(moe, k);
		}
	}
	for (int k = 0; !moe->matrix_fifo && k < moe->ws.top_k && !r->status &&
	     r->loading < (int)moe->prefetch; k++) {
		if (r->computed[k] || r->slots[k] >= 0) {
			continue;
		}
		int slot = moe_cache_reserve(&moe->cache, moe->ws.indices[k]);
		if (slot < 0) {
			break;
		}
		r->slots[k] = slot;
		moe->loads[k] = (struct moe_load_context){moe, k, slot};
		r->loading++;
		matrix_read_next(moe, k);
	}
	/* Finish advanced stages first, retaining Top-K order within each class. */
	for (int priority = 0; priority < 4 && !r->status; priority++) {
		for (int k = 0; k < moe->ws.top_k; k++) {
			if (r->computed[k] || r->busy[k] || r->slots[k] < 0) {
				continue;
			}
			unsigned stage = r->progress[k];
			if (stage >= r->ready[k]) {
				continue;
			}
			bool whole = stage == 0 && r->ready[k] == 3;
			int rank = whole ? 2 : stage == 2 ? 0 : stage == 1 ? 1 : 3;
			if (rank != priority) {
				continue;
			}
			r->stage[k] = whole ? -1 : (int)stage;
			r->busy[k] = true;
			if (!submit_expert(moe, k)) {
				r->busy[k] = false;
				return;
			}
		}
	}
}

static void
request_schedule(struct moe_bdev *moe)
{
	struct moe_request *r = &moe->request;

	if (moe->scheduling) {
		moe->reschedule = true;
		return;
	}
	moe->scheduling = true;
	do {
		moe->reschedule = false;
		if (!r->io && !r->warmup) {
			break;
		}
		if (moe->closing || atomic_load(&moe->store.removed)) {
			r->status = -ENODEV;
		}
		if (r->status) {
			if (moe->matrix_fifo) {
				moe_store_cancel_reads(&moe->store, r->status);
			} else {
				r->loading = 0;
				for (int k = 0; k < moe->ws.top_k; k++) {
					r->loading += r->reading[k];
				}
			}
			if (!moe->worker_busy && !r->loading) {
				request_finish(moe);
			}
			break;
		}
		if (!r->routed) {
			break;
		}
		if (r->completed == moe->ws.top_k && !moe->worker_busy) {
			submit_job(moe, MOE_JOB_COMBINE, 0);
			break;
		}
		matrix_schedule(moe);
		if (!moe->worker_busy && r->loading && !r->wait_trace) {
			r->wait_trace = moe_trace_start("WAIT_MATRIX", -1, -1, 0);
		}
		if (moe->diagnostics && !moe->worker_busy &&
		    r->loading && !r->weight_wait_start) {
			r->weight_wait_start = spdk_get_ticks();
		}
		if (r->status) {
			moe->reschedule = true;
		}
	} while (moe->reschedule);
	moe->scheduling = false;
}

static void
process_io(void *arg)
{
	struct spdk_bdev_io *io = arg;
	struct moe_bdev *moe = io->bdev->ctxt;
	struct spdk_bdev_io_nvme_passthru_params *pt = &io->u.nvme_passthru;
	size_t bytes = (size_t)moe->ws.d_model * sizeof(float);
	const void *input = NULL;

	if (moe->closing) {
		complete_io(io, -ENODEV);
		return;
	}
	if (io->type == SPDK_BDEV_IO_TYPE_READ) {
		if (!moe->output_valid && moe->ever_requested) {
			complete_io(io, -EIO);
			return;
		}
		size_t copied = 0;
		for (int i = 0; i < io->u.bdev.iovcnt; i++) {
			size_t length = spdk_min(io->u.bdev.iovs[i].iov_len, bytes - copied);
			memcpy(io->u.bdev.iovs[i].iov_base, (char *)moe->last_output + copied, length);
			copied += length;
		}
		complete_io(io, copied == bytes ? 0 : -EINVAL);
		return;
	}
	moe->output_valid = false;
	moe->ever_requested = true;
	if (io->type == SPDK_BDEV_IO_TYPE_RESET || io->type == SPDK_BDEV_IO_TYPE_WRITE) {
		complete_io(io, moe->request.io ? -EBUSY : 0);
		return;
	}
	if (pt->cmd.opc == MOE_VENDOR_OPCODE) {
		if (io->type == SPDK_BDEV_IO_TYPE_NVME_IOV_MD) {
			if (pt->iovcnt == 1 && pt->iovs[0].iov_len == bytes) {
				input = pt->iovs[0].iov_base;
			}
		} else if (pt->nbytes == bytes) {
			input = pt->buf;
		}
	}
	if (!input || moe->request.io) {
		complete_io(io, -EINVAL);
		return;
	}
	memcpy(moe->ws.input, input, bytes);
	moe->request = (struct moe_request){.io = io};
	moe_trace_name_thread("CPU IO + Decode / SPDK owner");
	moe_trace_capture(moe->sequence + 1, true);
	moe->request.trace_id = moe_trace_start("REQUEST", -1, -1, moe->sequence + 1);
	for (int k = 0; k < moe->ws.top_k; k++) {
		moe->request.slots[k] = -1;
	}
	moe->request.start = moe->diagnostics ? spdk_get_ticks() : 0;
	submit_job(moe, MOE_JOB_ROUTE, 0);
}

static void
read_buffer_ready(struct spdk_io_channel *ch, struct spdk_bdev_io *io, bool success)
{
	struct moe_bdev *moe = io->bdev->ctxt;
	if (!success) {
		complete_io(io, -ENOMEM);
		return;
	}
	spdk_thread_send_msg(moe->owner, process_io, io);
}

static void
bdev_moe_submit_request(struct spdk_io_channel *ch, struct spdk_bdev_io *io)
{
	struct moe_bdev *moe = io->bdev->ctxt;
	struct moe_io_ctx *ctx = (void *)io->driver_ctx;

	ctx->thread = spdk_get_thread();
	if (io->type == SPDK_BDEV_IO_TYPE_READ) {
		spdk_bdev_io_get_buf(io, read_buffer_ready, moe->bdev.blocklen);
	} else {
		spdk_thread_send_msg(moe->owner, process_io, io);
	}
}

static bool
bdev_moe_io_type_supported(void *ctx, enum spdk_bdev_io_type type)
{
	return type == SPDK_BDEV_IO_TYPE_READ || type == SPDK_BDEV_IO_TYPE_WRITE ||
	       type == SPDK_BDEV_IO_TYPE_RESET || type == SPDK_BDEV_IO_TYPE_NVME_IO ||
	       type == SPDK_BDEV_IO_TYPE_NVME_IO_MD || type == SPDK_BDEV_IO_TYPE_NVME_IOV_MD;
}

static struct spdk_io_channel *
bdev_moe_get_io_channel(void *ctx)
{
	return spdk_get_io_channel(ctx);
}

static int
channel_create(void *io_device, void *ctx)
{
	return 0;
}

static void
channel_destroy(void *io_device, void *ctx)
{
}

static void
free_resources(struct moe_bdev *moe)
{
	if (moe->worker_started) {
		pthread_mutex_lock(&moe->mutex);
		moe->worker_stop = true;
		pthread_cond_signal(&moe->condition);
		pthread_mutex_unlock(&moe->mutex);
		pthread_join(moe->worker, NULL);
	}
	if (moe->store_open) {
		moe_store_close(&moe->store);
	}
	moe_cache_destroy(&moe->cache);
	moe_workspace_destroy(&moe->ws);
	if (moe->diagnostics) {
		fclose(moe->diagnostics);
	}
	pthread_mutex_destroy(&moe->mutex);
	pthread_cond_destroy(&moe->condition);
	free(moe->intermediate);
	free(moe->last_output);
	free(moe->directory);
	free(moe->bdev.name);
	free(moe);
}

static void
io_device_gone(void *arg)
{
	struct moe_bdev *moe = arg;
	if (moe->store_open) {
		moe_store_close(&moe->store);
		moe->store_open = false;
		moe->router = NULL;
	}
	spdk_bdev_destruct_done(&moe->bdev, 0);
	free_resources(moe);
}

static void
destruct_on_owner(void *arg)
{
	struct moe_bdev *moe = arg;
	moe->closing = true;
	assert(!moe->request.io && !moe->worker_busy);
	spdk_io_device_unregister(moe, io_device_gone);
}

static int
bdev_moe_destruct(void *ctx)
{
	struct moe_bdev *moe = ctx;
	spdk_thread_send_msg(moe->owner, destruct_on_owner, moe);
	return 1;
}

static int
bdev_moe_dump_info_json(void *ctx, struct spdk_json_write_ctx *w)
{
	struct moe_bdev *moe = ctx;

	spdk_json_write_named_object_begin(w, "moe");
	spdk_json_write_named_string(w, "pipeline", moe->matrix_fifo ? "matrix_fifo" : "matrix");
	spdk_json_write_named_uint32(w, "warmup_requests", moe->warmup_requests);
	spdk_json_write_named_uint32(w, "warmup_completed", moe->warmup_done);
	spdk_json_write_named_uint32(w, "warmup_retries", moe->warmup_retries);
	spdk_json_write_named_int32(w, "cache_slots", moe->cache.capacity);
	spdk_json_write_named_string(w, "cache_memory", "cpu");
	spdk_json_write_named_int32(w, "compute_threads", 1);
	spdk_json_write_named_uint32(w, "decode_threads", 1);
	spdk_json_write_named_string(w, "weight_encoding", "lattice");
	spdk_json_write_object_end(w);
	return 0;
}

static void
bdev_moe_write_config_json(struct spdk_bdev *bdev, struct spdk_json_write_ctx *w)
{
	struct moe_bdev *moe = bdev->ctxt;
	struct spdk_bdev *base = spdk_bdev_desc_get_bdev(moe->store.desc);

	spdk_json_write_object_begin(w);
	spdk_json_write_named_string(w, "method", "bdev_moe_create");
	spdk_json_write_named_object_begin(w, "params");
	spdk_json_write_named_string(w, "name", bdev->name);
	spdk_json_write_named_string(w, "weight_dir", moe->directory);
	// Replay starts with the fixed profile and may perform the same startup fallback.
	spdk_json_write_named_string(w, "pipeline", "matrix_fifo");
	spdk_json_write_named_uint32(w, "warmup_requests", moe->warmup_requests);
	spdk_json_write_named_string(w, "kernel", "avx2");
	spdk_json_write_named_string(w, "weight_encoding", "lattice");
	spdk_json_write_named_string(w, "backend", spdk_bdev_get_module_name(base));
	spdk_json_write_named_string(w, "base_bdev", spdk_bdev_get_name(base));
	spdk_json_write_named_uint32(w, "io_size", MOE_STORE_MIB);
	spdk_json_write_named_uint32(w, "io_depth", 16);
	spdk_json_write_named_int32(w, "d_model", moe->ws.d_model);
	spdk_json_write_named_int32(w, "d_ff", moe->ws.d_ff);
	spdk_json_write_named_int32(w, "num_experts", moe->ws.num_experts);
	spdk_json_write_named_int32(w, "top_k", moe->ws.top_k);
	spdk_json_write_named_int32(w, "cache_slots", 8);
	spdk_json_write_named_int32(w, "compute_threads", 1);
	spdk_json_write_named_uint32(w, "decode_threads", 1);
	spdk_json_write_named_int32(w, "compute_cpu", moe->compute_cpu);
	spdk_json_write_object_end(w);
	spdk_json_write_object_end(w);
}

static const struct spdk_bdev_fn_table g_moe_fn_table = {
	.destruct = bdev_moe_destruct,
	.submit_request = bdev_moe_submit_request,
	.io_type_supported = bdev_moe_io_type_supported,
	.get_io_channel = bdev_moe_get_io_channel,
	.dump_info_json = bdev_moe_dump_info_json,
	.write_config_json = bdev_moe_write_config_json,
};

static int
topology_id(int cpu, const char *field)
{
	char path[160];
	int value = -1;
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, field);
	FILE *f = fopen(path, "r");
	if (f) {
		if (fscanf(f, "%d", &value) != 1) {
			value = -1;
		}
		fclose(f);
	}
	return value;
}

static int
cpu_node(int cpu)
{
	char path[128];
	int node = -1;
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
	DIR *directory = opendir(path);
	if (!directory) {
		return -1;
	}
	struct dirent *entry;
	while ((entry = readdir(directory))) {
		if (strncmp(entry->d_name, "node", 4) == 0) {
			char *end;
			long value = strtol(entry->d_name + 4, &end, 10);
			if (end != entry->d_name + 4 && !*end && value >= 0 && value <= INT_MAX) {
				node = (int)value;
				break;
			}
		}
	}
	closedir(directory);
	return node;
}

static int
start_worker(struct moe_bdev *moe, pthread_t *thread, void *(*entry)(void *), void *arg, int index)
{
	int reactor_cpu = sched_getcpu();
	int package = topology_id(reactor_cpu, "physical_package_id");
	int core = topology_id(reactor_cpu, "core_id");
	int node = cpu_node(reactor_cpu);
	long count = sysconf(_SC_NPROCESSORS_CONF);
	int rc = EINVAL;

	for (int cpu = 0; cpu < count && cpu < CPU_SETSIZE; cpu++) {
		if (!moe_affinity_allowed(cpu) ||
		    ((index > 0 || moe->compute_cpu < 0) && moe_affinity_reserved(cpu))) {
			continue;
		}
		if (index == 0 && moe->compute_cpu >= 0 && cpu != moe->compute_cpu) {
			continue;
		}
		bool reactor = false;
		uint32_t assigned;
		SPDK_ENV_FOREACH_CORE(assigned) {
			if (cpu == (int)assigned ||
			    (topology_id(cpu, "physical_package_id") == topology_id(assigned, "physical_package_id") &&
			     topology_id(cpu, "core_id") == topology_id(assigned, "core_id"))) {
				reactor = true;
			}
		}
		for (int i = 0; i < index; i++) {
			if (topology_id(cpu, "physical_package_id") == topology_id(moe->worker_cpus[i], "physical_package_id") &&
			    topology_id(cpu, "core_id") == topology_id(moe->worker_cpus[i], "core_id")) {
				reactor = true;
			}
		}
		if (reactor || ((index > 0 || moe->compute_cpu < 0) &&
		    (topology_id(cpu, "physical_package_id") != package ||
		     topology_id(cpu, "core_id") == core || (node >= 0 && cpu_node(cpu) != node)))) {
			continue;
		}
		pthread_attr_t attr;
		cpu_set_t cpus;
		pthread_attr_init(&attr);
		CPU_ZERO(&cpus);
		CPU_SET(cpu, &cpus);
		rc = pthread_attr_setaffinity_np(&attr, sizeof(cpus), &cpus);
		if (!rc) {
			rc = pthread_create(thread, &attr, entry, arg);
		}
		pthread_attr_destroy(&attr);
		if (!rc) {
			moe->worker_cpus[index] = cpu;
			SPDK_NOTICELOG("MoE worker=%d compute_cpu=%d reactor_cpu=%d worker_node=%d reactor_node=%d\n",
				       index, cpu, reactor_cpu, cpu_node(cpu), node);
			return 0;
		}
	}
	SPDK_ERRLOG("No separate allowed physical CPU for MoE worker; specify compute_cpu\n");
	return -rc;
}

static int
start_device(struct moe_bdev *moe)
{
	int rc = start_worker(moe, &moe->worker, compute_worker, moe, 0);
	if (rc) {
		return rc;
	}
	moe->worker_started = true;
	for (unsigned i = 0; i < moe->store.decode_threads; i++) {
		struct moe_decode_worker *worker = &moe->store.decoders[i];
		worker->store = &moe->store;
		worker->index = i;
		rc = start_worker(moe, &worker->thread, moe_store_decode_worker,
				  worker, 1 + i);
		if (rc) {
			return rc;
		}
		moe->store.decode_started++;
	}
	spdk_thread_send_msg(moe->owner, warmup_next, moe);
	return 0;
}

static void
startup_reduce_cache(struct moe_bdev *moe, int slots, int status)
{
	assert(!moe->worker_busy &&
	       !moe->request.loading && !moe->store.active_io);
	if (moe->cache.entries) {
		struct moe_cache_entry *entry = &moe->cache.entries[slots - 1];
		moe->cache.release(entry->buffer);
		memset(entry, 0, sizeof(*entry));
	}
	moe->cache.capacity = slots - 1;
	moe->warmup_retries++;
	moe->warmup_done = 0;
	moe->warmup_state = 0x9e3779b9U;
	moe->request = (struct moe_request){};
	if (moe->cache.capacity < moe->ws.top_k) {
		moe->matrix_fifo = false;
	}
	SPDK_WARNLOG("MoE startup retry: status=%d cache_slots=%d->%d pipeline=%s; retry once\n",
		     status, slots, moe->cache.capacity,
		     moe->matrix_fifo ? "matrix_fifo" : "matrix");
}

static void
warmup_next(void *arg)
{
	struct moe_bdev *moe = arg;
	int status = moe->request.status;
	if (moe->closing || atomic_load(&moe->store.removed)) {
		status = -ENODEV;
	}
	if (status && !moe->closing && !moe->warmup_retries && moe->cache.capacity > 1) {
		startup_reduce_cache(moe, moe->cache.capacity, status);
		spdk_thread_send_msg(moe->owner, warmup_next, moe);
		return;
	}
	if (!status && moe->warmup_done < moe->warmup_requests) {
		moe->request = (struct moe_request){.warmup = true};
		for (int k = 0; k < moe->ws.top_k; k++) {
			moe->request.slots[k] = -1;
		}
		for (int i = 0; i < moe->ws.d_model; i++) {
			moe->warmup_state = moe->warmup_state * 1664525U + 1013904223U;
			moe->ws.input[i] = (float)(moe->warmup_state >> 8) / 16777216.0f * 0.2f - 0.1f;
		}
		moe->request.start = moe->diagnostics ? spdk_get_ticks() : 0;
		moe->request.trace_id = moe_trace_start("WARMUP", -1, -1, moe->warmup_done + 1);
		submit_job(moe, MOE_JOB_ROUTE, 0);
		return;
	}
	if (!status) {
		SPDK_NOTICELOG("MoE startup warmup completed: requests=%u read_bytes=%" PRIu64
			       " decoded_bytes=%" PRIu64 "; publishing device\n",
			       moe->warmup_done, moe->store.read_bytes,
			       moe->store.decoded_bytes);
		moe->output_valid = false;
		memset(moe->last_output, 0, (size_t)moe->ws.d_model * sizeof(float));
		moe->request = (struct moe_request){};
		moe->cache.hits = moe->cache.misses = moe->cache.evictions = moe->cache.read_bytes = 0;
		moe->store.read_bytes = moe->store.submit_count = moe->store.resource_wait_count = 0;
		moe->store.decoded_bytes = moe->store.decode_ticks = 0;
		moe->store.busy_ticks = moe->store.depth_ticks = moe->store.pending_idle_ticks = 0;
		moe->store.meter_ticks = 0;
		moe->store.peak_io = 0;
		spdk_io_device_register(moe, channel_create, channel_destroy, 0, moe->bdev.name);
		status = spdk_bdev_register(&moe->bdev);
		moe->registered = status == 0;
		if (status) {
			spdk_io_device_unregister(moe, NULL);
		}
	} else {
		SPDK_ERRLOG("MoE startup warmup failed: completed=%u/%u status=%d\n",
			    moe->warmup_done, moe->warmup_requests, status);
	}
	moe->ready(moe->ready_arg, status);
	if (status) {
		free_resources(moe);
	}
}

static void
store_removed(void *arg)
{
	struct moe_bdev *moe = arg;
	moe->closing = true;
	request_schedule(moe);
	if (moe->registered) {
		moe->registered = false;
		spdk_bdev_unregister(&moe->bdev, NULL, NULL);
	}
}

static void
store_ready(void *arg, int status)
{
	struct moe_bdev *moe = arg;

	if (!status) {
		moe->router = moe->store.router;
		int slots = moe->cache.capacity;
		for (int attempt = 0; attempt < 2; attempt++) {
			status = moe_cache_init_cpu_buffers(&moe->cache, slots - attempt,
							   moe->ws.d_model, moe->ws.d_ff, moe->directory,
							   moe->store.layout.gate_bytes,
							   moe->store.layout.expert_stride);
			if (status != -ENOMEM || attempt || moe->warmup_retries || slots <= 1) {
				break;
			}
			startup_reduce_cache(moe, slots, status);
		}
		if (status) {
			SPDK_ERRLOG("Expert cache allocation failed: memory=%s slots=%d slot_bytes=%" PRIu64
				    " status=%d; pool is not expanded\n",
				    "cpu", slots - moe->warmup_retries,
				    moe->store.layout.expert_stride, status);
		} else {
			SPDK_NOTICELOG("Expert cache prefaulted: memory=%s slots=%d bytes=%" PRIu64 "\n",
				       "cpu (2MiB THP advised)",
				       moe->cache.capacity,
				       moe->cache.capacity * moe->store.layout.expert_stride);
		}
	}
	if (!status) {
		status = moe_store_prepare_decoder(&moe->store);
	}
	if (!status) {
		status = start_device(moe);
		if (!status) {
			return;
		}
	}
	moe->ready(moe->ready_arg, status);
	if (status) {
		free_resources(moe);
	}
}

int
bdev_moe_create_async(const struct moe_create_opts *o, moe_create_done done, void *arg)
{
	struct moe_layout layout;
	int rc;

	if (!o || !o->name || !o->backend || !o->base_bdev || !o->weight_dir ||
	    o->d_model <= 0 || o->d_ff <= 0 || o->num_experts < 8 || o->top_k != 8 ||
	    o->d_model % MOE_PACK_COLS || o->d_ff % MOE_PACK_COLS ||
	    o->cache_slots != 8 || o->io_size != MOE_STORE_MIB || o->io_depth != 16 ||
	    o->compute_threads != 1 || o->decode_threads != 1 || o->warmup_requests > 64 ||
	    (o->pipeline && strcmp(o->pipeline, "matrix_fifo")) ||
	    (o->kernel && strcmp(o->kernel, "avx2")) ||
	    (o->weight_encoding && strcmp(o->weight_encoding, "lattice")) ||
	    (strcmp(o->backend, "nvme") && strcmp(o->backend, "aio")) ||
	    !moe_kernel_supported(MOE_KERNEL_AVX2)) {
		return -EINVAL;
	}
	if (spdk_bdev_get_by_name(o->name)) {
		return -EEXIST;
	}
	rc = moe_layout_init(&layout, o->d_model, o->d_ff, o->num_experts, 512);
	if (rc) {
		return rc;
	}
	uint64_t cache_bytes = layout.expert_stride * 8;
	if (cache_bytes + (129 + 256) * MOE_STORE_MIB > UINT64_C(2000000000)) {
		return -ENOMEM;
	}
	struct spdk_bdev *base = spdk_bdev_get_by_name(o->base_bdev);
	if (!base || strcmp(spdk_bdev_get_module_name(base), o->backend)) {
		return -ENODEV;
	}
	struct moe_bdev *moe = calloc(1, sizeof(*moe));
	if (!moe) {
		return -ENOMEM;
	}
	pthread_mutex_init(&moe->mutex, NULL);
	pthread_cond_init(&moe->condition, NULL);
	moe->owner = spdk_get_thread();
	moe->bdev.name = strdup(o->name);
	moe->directory = strdup(o->weight_dir);
	moe->last_output = calloc(o->d_model, sizeof(float));
	moe->intermediate = calloc((size_t)o->top_k * 2 * o->d_ff, sizeof(float));
	moe->ready = done;
	moe->ready_arg = arg;
	moe->matrix_fifo = true;
	moe->warmup_requests = o->warmup_requests;
	moe->warmup_state = 0x9e3779b9U;
	moe->prefetch = 2;
	moe->compute_cpu = o->compute_cpu;
	moe->bdev.product_name = "MoE FP32 offload";
	moe->bdev.blocklen = (uint32_t)o->d_model * sizeof(float);
	moe->bdev.phys_blocklen = moe->bdev.blocklen;
	moe->bdev.blockcnt = 1;
	moe->bdev.ctxt = moe;
	moe->bdev.fn_table = &g_moe_fn_table;
	moe->bdev.module = &g_moe_if;
	if (!moe->bdev.name || !moe->directory || !moe->last_output || !moe->intermediate) {
		rc = -ENOMEM;
		goto fail;
	}
	rc = moe_workspace_init(&moe->ws, o->d_model, o->d_ff, o->num_experts, o->top_k,
				MOE_KERNEL_AVX2);
	if (rc) {
		goto fail;
	}
	if (o->diagnostics) {
		moe->diagnostics = fopen(o->diagnostics, "a");
		if (!moe->diagnostics) {
			rc = -errno;
			goto fail;
		}
	}
	moe->cache.capacity = 8;
	rc = moe_store_open(&moe->store, o->base_bdev, moe->directory, o->d_model,
			    o->d_ff, o->num_experts, strcmp(o->backend, "nvme") == 0,
			    MOE_STORE_MIB, 16, true, store_ready, moe);
	if (rc) {
		moe->cache.capacity = 0;
		goto fail;
	}
	moe->store_open = true;
	moe->store.diagnostics = moe->diagnostics != NULL;
	moe->store.removed_cb = store_removed;
	moe->store.removed_arg = moe;
	rc = spdk_bdev_module_claim_bdev_desc(moe->store.desc,
					     SPDK_BDEV_CLAIM_READ_MANY_WRITE_ONE, NULL, &g_moe_if);
	if (rc) {
		// The importer is still waiting for its first owner-thread I/O message.
		atomic_store(&moe->store.removed, true);
	}
	return 0;
fail:
	free_resources(moe);
	return rc;
}

void
bdev_moe_delete(const char *name, spdk_bdev_unregister_cb cb, void *arg)
{
	int rc = spdk_bdev_unregister_by_name(name, &g_moe_if, cb, arg);
	if (rc && cb) {
		cb(arg, rc);
	}
}

SPDK_LOG_REGISTER_COMPONENT(bdev_moe)
