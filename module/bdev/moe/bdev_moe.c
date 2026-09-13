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
#include "moe_cache.h"
#include "moe_store.h"
#include "moe_request.h"
#include "bdev_moe.h"

struct moe_bdev;
struct moe_load_context {
	struct moe_bdev *moe;
	int selected, slot;
};

enum moe_job { MOE_JOB_NONE, MOE_JOB_ROUTE, MOE_JOB_EXPERT, MOE_JOB_COMBINE };

struct moe_expert_worker {
	struct moe_bdev *moe;
	struct moe_workspace ws;
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	bool started, busy, pending, stop;
	int selected, status;
	uint64_t start;
};

struct moe_bdev {
	struct spdk_bdev bdev;
	struct spdk_thread *owner;
	struct moe_cache cache;
	struct moe_store store;
	struct moe_workspace ws;
	struct moe_request request;
	struct moe_load_context loads[MOE_REQUEST_MAX_K];
	float *router, *last_output;
	char *directory;
	bool matrix_pipeline;
	float *intermediate;
	bool packed, store_open, output_valid, worker_busy, worker_stop, worker_started;
	bool closing, scheduling, reschedule;
	bool job_ready;
	bool registered;
	bool ever_requested;
	unsigned prefetch;
	int compute_cpu;
	int compute_threads, extra_busy, worker_cpus[4];
	struct moe_expert_worker extra[3];
	pthread_t worker;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	enum moe_job job;
	int selected, worker_rc;
	uint64_t job_start, sequence;
	uint64_t worker_load_ticks;
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
	for (int k = 0; moe->matrix_pipeline && k < moe->ws.top_k; k++) {
		if (r->failed[k] && r->slots[k] >= 0) {
			moe->cache.entries[r->slots[k]].state = MOE_CACHE_EMPTY;
		}
	}
	for (int i = 0; i < moe->cache.capacity; i++) {
		if (moe->matrix_pipeline && moe->cache.entries[i].state == MOE_CACHE_LOADING) {
			moe->cache.entries[i].state = MOE_CACHE_EMPTY;
		}
		moe->cache.entries[i].refs = 0;
	}
	if (moe->diagnostics) {
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
			moe->packed ? moe->store.read_bytes : moe->cache.read_bytes, moe->store.peak_io,
			moe->matrix_pipeline ? "matrix" : "expert");
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
		fprintf(moe->diagnostics, "],\"first_compute_wait_ticks\":%" PRIu64 "}\n",
			first ? first - r->start : 0);
	}
	r->io = NULL;
	complete_io(io, r->status);
}

static bool
stage_done(struct moe_bdev *moe, int selected, int status)
{
	struct moe_request *r = &moe->request;

	if (!moe->matrix_pipeline || r->stage[selected] < 0) {
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
	if (moe->diagnostics && !r->first_compute[selected]) {
		r->first_compute[selected] = spdk_get_ticks();
	}

	if (!moe->matrix_pipeline || stage < 0) {
		return moe_expert(ws, e->w_gate, e->w_up, e->w_down, selected, moe->packed);
	}
	float *gate = moe->intermediate + (size_t)selected * 2 * ws->d_ff;
	float *up = gate + ws->d_ff;
	const float *weight = stage == 0 ? e->w_gate : stage == 1 ? e->w_up : e->w_down;
	if (moe->diagnostics) {
		r->compute_start[selected][stage] = spdk_get_ticks();
	}
	int rc = moe_expert_stage(ws, stage, weight, gate, up,
				 ws->expert_outputs + (size_t)selected * ws->d_model, 1);
	if (moe->diagnostics) {
		r->compute_end[selected][stage] = spdk_get_ticks();
	}
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
		r->expert_ticks += elapsed - moe->worker_load_ticks;
		r->weight_wait_ticks += moe->worker_load_ticks;
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

	pthread_mutex_lock(&moe->mutex);
	for (;;) {
		while (!moe->job_ready && !moe->worker_stop) {
			pthread_cond_wait(&moe->condition, &moe->mutex);
		}
		if (moe->worker_stop) {
			break;
		}
		enum moe_job job = moe->job;
		int selected = moe->selected;
		moe->job_ready = false;
		pthread_mutex_unlock(&moe->mutex);
		int rc = 0;
		uint64_t load_ticks = 0;
		if (job == MOE_JOB_ROUTE) {
			rc = moe_route(&moe->ws, moe->ws.input, moe->router);
		} else if (job == MOE_JOB_EXPERT) {
			int slot = moe->request.slots[selected];
			struct moe_cache_entry *e = &moe->cache.entries[slot];
			if (!moe->packed && e->state == MOE_CACHE_LOADING) {
				uint64_t start = moe->diagnostics ? spdk_get_ticks() : 0;
				rc = moe_cache_load_file(&moe->cache, slot);
				load_ticks = moe->diagnostics ? spdk_get_ticks() - start : 0;
			}
			if (!rc) {
				rc = compute_selected(moe, &moe->ws, selected);
			}
		} else {
			rc = moe_combine(&moe->ws, moe->last_output);
		}
		pthread_mutex_lock(&moe->mutex);
		moe->worker_rc = rc;
		moe->worker_load_ticks = load_ticks;
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

static void
extra_done(void *arg)
{
	struct moe_expert_worker *worker = arg;
	struct moe_bdev *moe = worker->moe;
	struct moe_request *r = &moe->request;
	int selected = worker->selected;
	struct moe_cache_entry *e = &moe->cache.entries[r->slots[selected]];

	pthread_mutex_lock(&worker->mutex);
	int status = worker->status;
	pthread_mutex_unlock(&worker->mutex);
	worker->busy = false;
	moe->extra_busy--;
	if (status) {
		r->status = status;
	} else if (!moe->matrix_pipeline || r->stage[selected] < 0 || r->stage[selected] == 2) {
		memcpy(moe->ws.expert_outputs + (size_t)selected * moe->ws.d_model,
		       worker->ws.expert_outputs + (size_t)selected * moe->ws.d_model,
		       (size_t)moe->ws.d_model * sizeof(float));
	}
if (stage_done(moe, selected, status)) {
		if (moe->diagnostics) {
			r->expert_ticks += spdk_get_ticks() - worker->start;
		}
		request_schedule(moe);
		return;
	}
	r->busy[selected] = false;
	e->state = status ? MOE_CACHE_EMPTY : MOE_CACHE_READY;
	e->refs = 0;
	e->last_used = ++moe->cache.clock;
	r->computed[selected] = true;
	r->completed++;
	if (moe->diagnostics) {
		r->expert_ticks += spdk_get_ticks() - worker->start;
	}
	request_schedule(moe);
}

static void *
extra_worker(void *arg)
{
	struct moe_expert_worker *worker = arg;
	struct moe_bdev *moe = worker->moe;

	pthread_mutex_lock(&worker->mutex);
	for (;;) {
		while (!worker->pending && !worker->stop) {
			pthread_cond_wait(&worker->condition, &worker->mutex);
		}
		if (worker->stop) {
			break;
		}
		worker->pending = false;
		int selected = worker->selected;
		pthread_mutex_unlock(&worker->mutex);
		int rc = compute_selected(moe, &worker->ws, selected);
		pthread_mutex_lock(&worker->mutex);
		worker->status = rc;
		spdk_thread_send_msg(moe->owner, extra_done, worker);
	}
	pthread_mutex_unlock(&worker->mutex);
	return NULL;
}

static bool
submit_expert(struct moe_bdev *moe, int selected)
{
	if (moe->request.weight_wait_start) {
		moe->request.weight_wait_ticks += spdk_get_ticks() - moe->request.weight_wait_start;
		moe->request.weight_wait_start = 0;
	}
	if (!moe->worker_busy) {
		submit_job(moe, MOE_JOB_EXPERT, selected);
		return true;
	}
	for (int i = 0; i < moe->compute_threads - 1; i++) {
		struct moe_expert_worker *worker = &moe->extra[i];
		if (worker->busy) {
			continue;
		}
		worker->busy = true;
		moe->extra_busy++;
		worker->start = moe->diagnostics ? spdk_get_ticks() : 0;
		pthread_mutex_lock(&worker->mutex);
		worker->selected = selected;
		memcpy(worker->ws.input, moe->ws.input, (size_t)moe->ws.d_model * sizeof(float));
		worker->pending = true;
		pthread_cond_signal(&worker->condition);
		pthread_mutex_unlock(&worker->mutex);
		return true;
	}
	return false;
}

static void
expert_loaded(void *arg, int status)
{
	struct moe_load_context *load = arg;
	struct moe_bdev *moe = load->moe;
	struct moe_cache_entry *e = &moe->cache.entries[load->slot];

	moe->request.loading--;
	e->state = status ? MOE_CACHE_EMPTY : MOE_CACHE_READY;
	if (status) {
		moe->request.status = status;
		e->refs = 0;
	}
	request_schedule(moe);
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
matrix_schedule(struct moe_bdev *moe)
{
	struct moe_request *r = &moe->request;

	for (int k = 0; k < moe->ws.top_k && !r->status; k++) {
		if (r->slots[k] >= 0 && r->ready[k] < 3 && !r->reading[k]) {
			matrix_read_next(moe, k);
		}
	}
	for (int k = 0; k < moe->ws.top_k && !r->status &&
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
		if (!r->io) {
			break;
		}
		if (moe->closing || (moe->packed && atomic_load(&moe->store.removed))) {
			r->status = -ENODEV;
		}
		if (r->status) {
			if (moe->matrix_pipeline) {
				r->loading = 0;
				for (int k = 0; k < moe->ws.top_k; k++) {
					r->loading += r->reading[k];
				}
			}
			if (!moe->worker_busy && !moe->extra_busy && !r->loading) {
				request_finish(moe);
			}
			break;
		}
		if (!r->routed) {
			break;
		}
		if (r->completed == moe->ws.top_k && !moe->worker_busy && !moe->extra_busy) {
			submit_job(moe, MOE_JOB_COMBINE, 0);
			break;
		}
		if (moe->matrix_pipeline) {
			matrix_schedule(moe);
			if (moe->diagnostics && !moe->worker_busy && !moe->extra_busy &&
			    r->loading && !r->weight_wait_start) {
				r->weight_wait_start = spdk_get_ticks();
			}
			if (r->status) {
				moe->reschedule = true;
			}
			continue;
		}
		/* A freed slot admits the eighth expert without requiring eight resident experts. */
		for (int k = 0; moe->packed && k < moe->ws.top_k &&
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
			int rc = moe_store_read_expert(&moe->store, moe->ws.indices[k],
						moe->cache.entries[slot].buffer, expert_loaded, &moe->loads[k]);
			if (rc) {
				expert_loaded(&moe->loads[k], rc);
			}
			if (r->status) {
				break;
			}
		}
		if (r->status) {
			continue;
		}
		for (int k = 0; k < moe->ws.top_k; k++) {
			if ((int)moe->worker_busy + moe->extra_busy == moe->compute_threads) {
				break;
			}
			if (r->computed[k]) {
				continue;
			}
			int slot = r->slots[k];
			if (slot < 0 && !moe->packed) {
				slot = moe_cache_reserve(&moe->cache, moe->ws.indices[k]);
				r->slots[k] = slot;
			}
			if (slot >= 0 && (moe->cache.entries[slot].state == MOE_CACHE_READY || !moe->packed)) {
				if (moe->packed || moe->cache.entries[slot].state == MOE_CACHE_READY) {
					moe->cache.entries[slot].state = MOE_CACHE_IN_USE;
				}
				bool submitted = submit_expert(moe, k);
				assert(submitted);
				(void)submitted;
			}
		}
		if (moe->diagnostics && !moe->worker_busy && !moe->extra_busy &&
		    r->loading && !r->weight_wait_start) {
			r->weight_wait_start = spdk_get_ticks();
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
	for (int i = 0; i < 3; i++) {
		struct moe_expert_worker *worker = &moe->extra[i];
		if (worker->started) {
			pthread_mutex_lock(&worker->mutex);
			worker->stop = true;
			pthread_cond_signal(&worker->condition);
			pthread_mutex_unlock(&worker->mutex);
			pthread_join(worker->thread, NULL);
		}
		moe_workspace_destroy(&worker->ws);
		pthread_mutex_destroy(&worker->mutex);
		pthread_cond_destroy(&worker->condition);
	}
	if (moe->store_open) {
		moe_store_close(&moe->store);
	} else {
		free(moe->router);
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
	assert(!moe->request.io && !moe->worker_busy && !moe->extra_busy);
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
	spdk_json_write_named_string(w, "pipeline", moe->matrix_pipeline ? "matrix" : "expert");
	spdk_json_write_named_int32(w, "compute_threads", moe->compute_threads);
	spdk_json_write_object_end(w);
	return 0;
}

static void
bdev_moe_write_config_json(struct spdk_bdev *bdev, struct spdk_json_write_ctx *w)
{
	struct moe_bdev *moe = bdev->ctxt;

	spdk_json_write_object_begin(w);
	spdk_json_write_named_string(w, "method", "bdev_moe_create");
	spdk_json_write_named_object_begin(w, "params");
	spdk_json_write_named_string(w, "name", bdev->name);
	spdk_json_write_named_string(w, "weight_dir", moe->directory);
	spdk_json_write_named_string(w, "pipeline", moe->matrix_pipeline ? "matrix" : "expert");
	spdk_json_write_named_string(w, "kernel", moe_kernel_name(moe->ws.kernel));
	if (moe->packed) {
		struct spdk_bdev *base = spdk_bdev_desc_get_bdev(moe->store.desc);
		spdk_json_write_named_string(w, "backend", spdk_bdev_get_module_name(base));
		spdk_json_write_named_string(w, "base_bdev", spdk_bdev_get_name(base));
		spdk_json_write_named_uint32(w, "io_size", moe->store.io_size);
		spdk_json_write_named_uint32(w, "io_depth", moe->store.io_depth);
	} else {
		spdk_json_write_named_string(w, "backend", "file");
	}
	spdk_json_write_named_int32(w, "d_model", moe->ws.d_model);
	spdk_json_write_named_int32(w, "d_ff", moe->ws.d_ff);
	spdk_json_write_named_int32(w, "num_experts", moe->ws.num_experts);
	spdk_json_write_named_int32(w, "top_k", moe->ws.top_k);
	spdk_json_write_named_int32(w, "cache_slots", moe->cache.capacity);
	spdk_json_write_named_int32(w, "compute_threads", moe->compute_threads);
	spdk_json_write_named_int32(w, "compute_cpu", moe->compute_cpu);
	spdk_json_write_named_uint32(w, "prefetch", moe->prefetch);
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

static void *
dma_allocate(size_t size)
{
	return spdk_dma_zmalloc(size, 4096, NULL);
}

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
			SPDK_NOTICELOG("MoE worker=%d compute_cpu=%d reactor_cpu=%d\n", index, cpu, reactor_cpu);
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
	for (int i = 0; i < moe->compute_threads - 1; i++) {
		struct moe_expert_worker *worker = &moe->extra[i];
		rc = start_worker(moe, &worker->thread, extra_worker, worker, i + 1);
		if (rc) {
			return rc;
		}
		worker->started = true;
	}
	spdk_io_device_register(moe, channel_create, channel_destroy, 0, moe->bdev.name);
	rc = spdk_bdev_register(&moe->bdev);
	moe->registered = rc == 0;
	if (rc) {
		spdk_io_device_unregister(moe, NULL);
	}
	return rc;
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
		status = moe_cache_init_buffers(&moe->cache, moe->cache.capacity,
						moe->ws.d_model, moe->ws.d_ff, moe->directory,
						moe->store.layout.gate_bytes,
						moe->store.layout.expert_stride, dma_allocate, spdk_dma_free);
		if (status) {
			SPDK_ERRLOG("DMA cache allocation failed: slots=%d slot_bytes=%" PRIu64
				    " pool_budget=1536MiB; pool is not expanded\n",
				    slots, moe->store.layout.expert_stride);
		}
	}
	if (!status) {
		status = start_device(moe);
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
	enum moe_kernel kernel;
	bool packed;
	int rc;

	if (!o || !o->name || !o->backend || !o->weight_dir || !done ||
	    o->cache_slots < 1 || o->cache_slots > 8 || o->top_k < 1 || o->top_k > MOE_REQUEST_MAX_K ||
	    o->top_k > o->num_experts || o->d_model < 128 || o->d_model % 128 ||
	    o->prefetch < 1 || o->prefetch > 4 ||
	    (o->compute_threads != 1 && o->compute_threads != 2 && o->compute_threads != 4) ||
	    moe_kernel_parse(o->kernel, &kernel)) {
		return -EINVAL;
	}
	packed = strcmp(o->backend, "file") != 0;
	bool matrix = o->pipeline && !strcmp(o->pipeline, "matrix");
	if ((o->pipeline && strcmp(o->pipeline, "expert") && !matrix) || (matrix && !packed)) {
		return -EINVAL;
	}
	if (!packed && o->compute_threads != 1) {
		return -EINVAL;
	}
	if (packed && (!o->base_bdev || (strcmp(o->backend, "aio") && strcmp(o->backend, "nvme")))) {
		return -EINVAL;
	}
	if (spdk_bdev_get_by_name(o->name)) {
		return -EEXIST;
	}
	rc = moe_layout_init(&layout, o->d_model, o->d_ff, o->num_experts, 512);
	if (rc) {
		return rc;
	}
	uint64_t cache_bytes = layout.expert_stride * o->cache_slots;
	if (cache_bytes + 128 * MOE_STORE_MIB + MOE_STORE_MIB > 1536 * MOE_STORE_MIB) {
		SPDK_ERRLOG("MoE memory admission failed: cache=%" PRIu64
			    " reserve=129MiB DMA_budget=1536MiB\n", cache_bytes);
		return -ENOMEM;
	}
	struct moe_bdev *moe = calloc(1, sizeof(*moe));
	if (!moe) {
		return -ENOMEM;
	}
	pthread_mutex_init(&moe->mutex, NULL);
	pthread_cond_init(&moe->condition, NULL);
	for (int i = 0; i < 3; i++) {
		moe->extra[i].moe = moe;
		pthread_mutex_init(&moe->extra[i].mutex, NULL);
		pthread_cond_init(&moe->extra[i].condition, NULL);
	}
	moe->owner = spdk_get_thread();
	moe->bdev.name = strdup(o->name);
	moe->directory = strdup(o->weight_dir);
	moe->last_output = calloc(o->d_model, sizeof(float));
	moe->ready = done;
	moe->ready_arg = arg;
	moe->packed = packed;
	moe->matrix_pipeline = matrix;
	if (matrix) {
		size_t bytes = (size_t)o->top_k * 2 * o->d_ff * sizeof(float);
		moe->intermediate = malloc(bytes);
		if (!moe->intermediate) {
			rc = -ENOMEM;
			goto fail;
		}
		memset(moe->intermediate, 0, bytes);
	}
	moe->prefetch = o->prefetch;
	moe->compute_cpu = o->compute_cpu;
	moe->compute_threads = o->compute_threads;
	moe->bdev.product_name = "MoE FP32 offload";
	moe->bdev.blocklen = (uint32_t)o->d_model * sizeof(float);
	moe->bdev.phys_blocklen = moe->bdev.blocklen;
	moe->bdev.blockcnt = 1;
	moe->bdev.ctxt = moe;
	moe->bdev.fn_table = &g_moe_fn_table;
	moe->bdev.module = &g_moe_if;
	if (!moe->bdev.name || !moe->directory || !moe->last_output) {
		rc = -ENOMEM;
		goto fail;
	}
	rc = moe_workspace_init(&moe->ws, o->d_model, o->d_ff, o->num_experts, o->top_k, kernel);
	if (rc) {
		goto fail;
	}
	for (int i = 0; i < moe->compute_threads - 1; i++) {
		rc = moe_workspace_init(&moe->extra[i].ws, o->d_model, o->d_ff,
					o->num_experts, o->top_k, kernel);
		if (rc) {
			goto fail;
		}
	}
	if (o->diagnostics) {
		moe->diagnostics = fopen(o->diagnostics, "a");
		if (!moe->diagnostics) {
			rc = -errno;
			goto fail;
		}
	}
	SPDK_NOTICELOG("MoE kernel=%s cache=%d slots bytes<=%" PRIu64 " backend=%s\n",
		       moe_kernel_name(kernel), o->cache_slots, cache_bytes, o->backend);
	if (packed) {
		struct spdk_bdev *base = spdk_bdev_get_by_name(o->base_bdev);
		if (!base || (strcmp(o->backend, "aio") == 0 &&
			     strcmp(spdk_bdev_get_module_name(base), "aio"))) {
			rc = -ENODEV;
			goto fail;
		}
		moe->cache.capacity = o->cache_slots;
		rc = moe_store_open(&moe->store, o->base_bdev, moe->directory, o->d_model,
				    o->d_ff, o->num_experts, strcmp(o->backend, "nvme") == 0,
				    o->io_size, o->io_depth, store_ready, moe);
		if (rc) {
			moe->cache.capacity = 0;
			goto fail;
		}
		moe->store_open = true;
		moe->store.removed_cb = store_removed;
		moe->store.removed_arg = moe;
		rc = spdk_bdev_module_claim_bdev_desc(moe->store.desc,
						      SPDK_BDEV_CLAIM_READ_MANY_WRITE_ONE, NULL, &g_moe_if);
		if (rc) {
			/* The importer is still waiting for its first owner-thread I/O message. */
			atomic_store(&moe->store.removed, true);
		}
		return 0;
	}
	char path[4096];
	int n = snprintf(path, sizeof(path), "%s/W_router_%dx%d.bin", o->weight_dir,
			 o->d_model, o->num_experts);
	if (n < 0 || (size_t)n >= sizeof(path)) {
		rc = -ENAMETOOLONG;
		goto fail;
	}
	FILE *file = fopen(path, "rb");
	if (!file) {
		rc = -errno;
		goto fail;
	}
	moe->router = malloc(layout.router_bytes);
	bool valid = moe->router && fread(moe->router, 1, layout.router_bytes, file) == layout.router_bytes &&
		     fgetc(file) == EOF && !ferror(file);
	fclose(file);
	if (!valid) {
		rc = -EIO;
		goto fail;
	}
	rc = moe_cache_init(&moe->cache, o->cache_slots, o->d_model, o->d_ff, moe->directory);
	if (!rc) {
		rc = start_device(moe);
	}
	if (rc) {
		goto fail;
	}
	done(arg, 0);
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
