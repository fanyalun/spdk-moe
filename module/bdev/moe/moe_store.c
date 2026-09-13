/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/bdev_module.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "moe_ffn/matvec_packed.h"
#include "moe_store.h"

#define MOE_STORE_MAGIC UINT64_C(0x31454f4d4b445053)
#define MOE_IMPORT_LIMIT (64 * MOE_STORE_MIB)

static uint64_t
align_up(uint64_t value, uint64_t alignment)
{
	return (value + alignment - 1) / alignment * alignment;
}

int
moe_layout_init(struct moe_layout *l, int d, int h, int experts, uint32_t block_size)
{
	uint64_t gate = matvec_packed_elements(d, h) * sizeof(float);
	uint64_t down = matvec_packed_elements(h, d) * sizeof(float);

	if (!gate || !down || experts <= 0 || !block_size || MOE_STORE_MIB % block_size ||
	    (uint64_t)d * h * sizeof(float) + MOE_STORE_MIB > MOE_IMPORT_LIMIT ||
	    (uint64_t)d * experts * sizeof(float) + MOE_STORE_MIB > MOE_IMPORT_LIMIT) {
		return -EINVAL;
	}
	*l = (struct moe_layout){
		.magic = MOE_STORE_MAGIC, .version = 1, .d_model = d, .d_ff = h,
		.experts = experts, .pack_cols = MOE_PACK_COLS,
		.router_offset = MOE_STORE_MIB,
		.router_bytes = (uint64_t)d * experts * sizeof(float),
		.gate_bytes = align_up(gate, block_size),
		.down_bytes = align_up(down, block_size),
	};
	l->expert_offset = l->router_offset + align_up(l->router_bytes, MOE_STORE_MIB);
	l->expert_stride = align_up(2 * l->gate_bytes + l->down_bytes, MOE_STORE_MIB);
	if ((uint64_t)experts > (UINT64_MAX - l->expert_offset) / l->expert_stride) {
		return -EOVERFLOW;
	}
	l->total_bytes = l->expert_offset + (uint64_t)experts * l->expert_stride;
	return 0;
}

static void store_pump(struct moe_store *s);

static void
base_event(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *arg)
{
	struct moe_store *s = arg;

	if (type == SPDK_BDEV_EVENT_REMOVE) {
		atomic_store(&s->removed, true);
		store_pump(s);
		if (s->removed_cb) {
			s->removed_cb(s->removed_arg);
		}
	}
}

static void
import_signal(struct moe_store *s, int status)
{
	pthread_mutex_lock(&s->mutex);
	s->import_status = status;
	s->import_waiting = false;
	pthread_cond_signal(&s->condition);
	pthread_mutex_unlock(&s->mutex);
}

static void
import_complete(struct spdk_bdev_io *io, bool success, void *arg)
{
	struct moe_store *s = arg;

	spdk_bdev_free_io(io);
	import_signal(s, success ? 0 : -EIO);
}

static void
import_submit(void *arg)
{
	struct moe_store *s = arg;
	int rc;

	if (atomic_load(&s->removed)) {
		import_signal(s, -ENODEV);
		return;
	}
	if (s->import_flush) {
		rc = spdk_bdev_flush(s->desc, s->channel, 0, s->layout.total_bytes, import_complete, s);
	} else {
		rc = spdk_bdev_write(s->desc, s->channel, s->import_buffer, s->import_offset,
				     s->import_length, import_complete, s);
	}
	if (rc == -ENOMEM) {
		s->import_wait.bdev = spdk_bdev_desc_get_bdev(s->desc);
		s->import_wait.cb_fn = import_submit;
		s->import_wait.cb_arg = s;
		rc = spdk_bdev_queue_io_wait(s->import_wait.bdev, s->channel, &s->import_wait);
	}
	if (rc) {
		import_signal(s, rc);
	}
}

static int
import_io(struct moe_store *s, uint64_t offset, uint64_t length, bool flush)
{
	int rc;

	pthread_mutex_lock(&s->mutex);
	s->import_offset = offset;
	s->import_length = length;
	s->import_flush = flush;
	s->import_waiting = true;
	rc = spdk_thread_send_msg(s->owner, import_submit, s);
	if (rc) {
		s->import_waiting = false;
		pthread_mutex_unlock(&s->mutex);
		return rc;
	}
	while (s->import_waiting) {
		pthread_cond_wait(&s->condition, &s->mutex);
	}
	rc = s->import_status;
	pthread_mutex_unlock(&s->mutex);
	return rc;
}

static int
read_source(const char *path, void *raw, size_t bytes)
{
	struct stat st;
	int fd = open(path, O_RDONLY | O_DIRECT);
	bool direct = true;
	ssize_t count;

	if (fd < 0) {
		direct = false;
		fd = open(path, O_RDONLY);
	}
	if (fd < 0) {
		return -errno;
	}
	if (fstat(fd, &st) || st.st_size != (off_t)bytes) {
		close(fd);
		return -EINVAL;
	}
	do {
		count = pread(fd, raw, align_up(bytes, 4096), 0);
	} while (count < 0 && errno == EINTR);
	if (count < 0 && errno == EINVAL && direct) {
		close(fd);
		direct = false;
		fd = open(path, O_RDONLY);
		if (fd < 0) {
			return -errno;
		}
		do {
			count = pread(fd, raw, bytes, 0);
		} while (count < 0 && errno == EINTR);
	}
	close(fd);
	if (!direct) {
		SPDK_NOTICELOG("MoE import source uses buffered I/O: %s\n", path);
	}
	return count == (ssize_t)bytes ? 0 : -EIO;
}

static int
import_matrix(struct moe_store *s, const float *raw, int rows, int cols,
	      uint64_t offset, uint64_t bytes, bool packed)
{
	uint64_t elements = packed ? matvec_packed_elements(rows, cols) : (uint64_t)rows * cols;

	for (uint64_t start = 0; start < bytes; start += MOE_STORE_MIB) {
		uint64_t length = spdk_min(MOE_STORE_MIB, bytes - start);
		float *dest = s->import_buffer;
		uint64_t first = start / sizeof(float);
		uint64_t limit = spdk_min(elements, first + length / sizeof(float));

		memset(dest, 0, length);
		if (!packed && first < limit) {
			memcpy(dest, raw + first, (limit - first) * sizeof(float));
		} else {
			for (uint64_t q = first; q < limit; q += MOE_PACK_COLS) {
				uint64_t col = q / ((uint64_t)rows * MOE_PACK_COLS) * MOE_PACK_COLS;
				uint64_t row = (q / MOE_PACK_COLS) % rows;
				uint64_t width = spdk_min(MOE_PACK_COLS, (uint64_t)cols - col);
				memcpy(dest + q - first, raw + row * cols + col, width * sizeof(float));
			}
		}
		int rc = import_io(s, offset + start, length, false);
		if (rc) {
			return rc;
		}
	}
	return 0;
}

static void
import_finished(void *arg)
{
	struct moe_store *s = arg;
	int status = s->import_status;

	pthread_join(s->importer, NULL);
	s->import_started = false;
	spdk_dma_free(s->import_buffer);
	s->import_buffer = NULL;
	s->ready(s->ready_arg, status);
}

static void *
import_worker(void *arg)
{
	struct moe_store *s = arg;
	struct moe_layout *l = &s->layout;
	uint64_t raw_bytes = spdk_max((uint64_t)l->d_model * l->d_ff * sizeof(float), l->router_bytes);
	float *raw = NULL;
	char path[4096];
	int rc = posix_memalign((void **)&raw, 4096, align_up(raw_bytes, 4096));

	if (rc) {
		rc = -rc;
		goto done;
	}
	memset(s->import_buffer, 0, MOE_STORE_MIB);
	memcpy(s->import_buffer, l, sizeof(*l));
	rc = import_io(s, 0, MOE_STORE_MIB, false);
	if (rc || (rc = import_io(s, 0, 0, true))) {
		goto done;
	}
	int n = snprintf(path, sizeof(path), "%s/W_router_%ux%u.bin", s->directory, l->d_model, l->experts);
	if (n < 0 || (size_t)n >= sizeof(path)) {
		rc = -ENAMETOOLONG;
		goto done;
	}
	rc = read_source(path, raw, l->router_bytes);
	if (rc) {
		goto done;
	}
	memcpy(s->router, raw, l->router_bytes);
	rc = import_matrix(s, raw, l->d_model, l->experts, l->router_offset,
			   l->expert_offset - l->router_offset, false);
	for (uint32_t e = 0; !rc && e < l->experts; e++) {
		const char *names[] = {"W_gate", "W_up", "W_down"};
		uint64_t offset = l->expert_offset + e * l->expert_stride;

		for (int m = 0; !rc && m < 3; m++) {
			n = snprintf(path, sizeof(path), "%s/%s_%u_%ux%u.bin", s->directory,
				     names[m], e, l->d_model, l->d_ff);
			if (n < 0 || (size_t)n >= sizeof(path)) {
				rc = -ENAMETOOLONG;
				break;
			}
			rc = read_source(path, raw, (uint64_t)l->d_model * l->d_ff * sizeof(float));
			if (!rc) {
				uint64_t bytes = m == 2 ? l->down_bytes : l->gate_bytes;
				rc = import_matrix(s, raw, m == 2 ? l->d_ff : l->d_model,
						   m == 2 ? l->d_model : l->d_ff, offset, bytes, true);
				offset += bytes;
			}
		}
	}
	if (!rc) {
		rc = import_io(s, 0, 0, true);
	}
	if (!rc) {
		l->complete = 1;
		memset(s->import_buffer, 0, MOE_STORE_MIB);
		memcpy(s->import_buffer, l, sizeof(*l));
		rc = import_io(s, 0, MOE_STORE_MIB, false);
	}
	if (!rc) {
		rc = import_io(s, 0, 0, true);
	}
done:
	free(raw);
	s->import_status = rc;
	spdk_thread_send_msg(s->owner, import_finished, s);
	return NULL;
}

int
moe_store_open(struct moe_store *s, const char *base, const char *directory,
	       int d, int h, int experts, bool strict_nvme, unsigned io_size,
	       unsigned io_depth, moe_store_done ready, void *arg)
{
	struct spdk_bdev *bdev;
	int rc;

	memset(s, 0, sizeof(*s));
	atomic_init(&s->removed, false);
	s->owner = spdk_get_thread();
	pthread_mutex_init(&s->mutex, NULL);
	pthread_cond_init(&s->condition, NULL);
	s->ready = ready;
	s->ready_arg = arg;
	rc = spdk_bdev_open_ext(base, true, base_event, s, &s->desc);
	if (rc) {
		goto fail;
	}
	bdev = spdk_bdev_desc_get_bdev(s->desc);
	if ((strict_nvme && strcmp(spdk_bdev_get_module_name(bdev), "nvme")) ||
	    !spdk_bdev_io_type_supported(bdev, SPDK_BDEV_IO_TYPE_FLUSH)) {
		rc = -ENOTSUP;
		goto fail;
	}
	rc = moe_layout_init(&s->layout, d, h, experts, spdk_bdev_get_block_size(bdev));
	if (rc) {
		goto fail;
	}
	if (io_depth < 1 || io_depth > MOE_STORE_MAX_IO || io_size < 4096 ||
	    io_size % spdk_bdev_get_block_size(bdev) ||
	    spdk_bdev_get_buf_align(bdev) > 4096 ||
	    s->layout.total_bytes / spdk_bdev_get_block_size(bdev) > spdk_bdev_get_num_blocks(bdev) ||
	    spdk_bdev_get_write_unit_size(bdev) != 1) {
		rc = -EINVAL;
		goto fail;
	}
	s->channel = spdk_bdev_get_io_channel(s->desc);
	s->directory = strdup(directory);
	s->router = malloc(s->layout.router_bytes);
	s->import_buffer = spdk_dma_zmalloc(MOE_STORE_MIB,
					  spdk_max(4096, spdk_bdev_get_buf_align(bdev)), NULL);
	if (!s->channel || !s->directory || !s->router || !s->import_buffer) {
		rc = -ENOMEM;
		goto fail;
	}
	s->io_size = io_size;
	s->io_depth = io_depth;
	SPDK_NOTICELOG("MoE store module=%s bytes=%" PRIu64 " import_buffer<=64MiB\n",
		       spdk_bdev_get_module_name(bdev), s->layout.total_bytes);
	rc = pthread_create(&s->importer, NULL, import_worker, s);
	if (rc) {
		rc = -rc;
		goto fail;
	}
	s->import_started = true;
	return 0;
fail:
	moe_store_close(s);
	return rc;
}

static void
read_finish_op(struct moe_store_op *op, int status)
{
	struct moe_store_read *r = op->read;
	struct moe_store *s = r->store;

	if (op->submitted) {
		s->active_io--;
		if (!status) {
			s->read_bytes += op->length;
		}
	}
	if (status) {
		r->status = status;
	}
	r->outstanding--;
	op->read = NULL;
	op->submitted = false;
}

static void
read_complete(struct spdk_bdev_io *io, bool success, void *arg)
{
	struct moe_store_op *op = arg;
	struct moe_store *s = op->read->store;

	spdk_bdev_free_io(io);
	read_finish_op(op, success ? 0 : -EIO);
	store_pump(s);
}

static void
read_submit(void *arg)
{
	struct moe_store_op *op = arg;
	struct moe_store_read *r = op->read;
	struct moe_store *s = r->store;
	int rc;

	if (atomic_load(&s->removed) || r->status) {
		read_finish_op(op, -EIO);
		store_pump(s);
		return;
	}
	rc = spdk_bdev_read(s->desc, s->channel, (char *)r->buffer + op->offset,
			    r->offset + op->offset, op->length, read_complete, op);
	if (rc == -ENOMEM) {
		op->wait.bdev = spdk_bdev_desc_get_bdev(s->desc);
		op->wait.cb_fn = read_submit;
		op->wait.cb_arg = op;
		rc = spdk_bdev_queue_io_wait(op->wait.bdev, s->channel, &op->wait);
	} else if (!rc) {
		op->submitted = true;
		s->active_io++;
		s->peak_io = spdk_max(s->peak_io, s->active_io);
	}
	if (rc) {
		read_finish_op(op, rc);
		store_pump(s);
	}
}

static void
store_pump(struct moe_store *s)
{
	if (s->pumping) {
		s->repump = true;
		return;
	}
	s->pumping = true;
	do {
	s->repump = false;
	for (unsigned t = 0; t < MOE_STORE_MAX_READS; t++) {
		struct moe_store_read *r = &s->reads[t];
		if (!r->active) {
			continue;
		}
		if (atomic_load(&s->removed)) {
			r->status = -ENODEV;
		}
		for (unsigned i = 0; !r->status && r->next < s->layout.expert_stride && i < s->io_depth; i++) {
			struct moe_store_op *op = &s->ops[i];
			if (op->read) {
				continue;
			}
			op->read = r;
			op->offset = r->next;
			op->length = spdk_min(s->io_size, s->layout.expert_stride - r->next);
			r->next += op->length;
			r->outstanding++;
			read_submit(op);
		}
		if ((r->status || r->next == s->layout.expert_stride) && !r->outstanding) {
			r->active = false;
			r->done(r->arg, r->status);
		}
	}
	} while (s->repump);
	s->pumping = false;
}

int
moe_store_read_expert(struct moe_store *s, int expert, void *buffer,
		      moe_store_done done, void *arg)
{
	if (!s->layout.complete || expert < 0 ||
	    (uint32_t)expert >= s->layout.experts || atomic_load(&s->removed)) {
		return -EINVAL;
	}
	for (unsigned t = 0; t < MOE_STORE_MAX_READS; t++) {
		struct moe_store_read *r = &s->reads[t];
		if (!r->active) {
			*r = (struct moe_store_read){
				.store = s, .buffer = buffer, .active = true, .done = done, .arg = arg,
				.offset = s->layout.expert_offset + (uint64_t)expert * s->layout.expert_stride,
			};
			store_pump(s);
			return 0;
		}
	}
	return -EAGAIN;
}

void
moe_store_close(struct moe_store *s)
{
	assert(!s->import_started);
	for (unsigned i = 0; i < MOE_STORE_MAX_READS; i++) {
		assert(!s->reads[i].active);
	}
	if (s->channel) {
		spdk_put_io_channel(s->channel);
	}
	if (s->desc) {
		spdk_bdev_close(s->desc);
	}
	spdk_dma_free(s->import_buffer);
	free(s->directory);
	free(s->router);
	pthread_mutex_destroy(&s->mutex);
	pthread_cond_destroy(&s->condition);
	memset(s, 0, sizeof(*s));
}
