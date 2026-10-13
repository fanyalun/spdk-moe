/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "../../module/bdev/moe/moe_store.c"

static struct spdk_bdev_io_wait_entry *g_wait;
static bool g_starve;
static struct { spdk_bdev_io_completion_cb callback; void *arg; } g_pending[16];
static unsigned g_count, g_done, g_failed;
static uint64_t g_low, g_high;
static uint64_t g_ticks;
static uint64_t g_offsets[128];
static unsigned g_submitted;
static uint64_t g_io_bytes = 4096, g_ready[4];
static unsigned g_progress;

uint64_t
spdk_get_ticks(void)
{
	return ++g_ticks;
}

int
spdk_bdev_read(struct spdk_bdev_desc *desc, struct spdk_io_channel *channel, void *buffer,
	       uint64_t offset, uint64_t bytes, spdk_bdev_io_completion_cb callback, void *arg)
{
	if (g_starve) {
		return -ENOMEM;
	}
	assert(bytes == g_io_bytes && offset % 4096 == 0 && g_count < 16);
	if (g_high) {
		assert(offset >= g_low && offset + bytes <= g_high);
	}
	memset(buffer, 0x5a, bytes);
	assert(g_submitted < SPDK_COUNTOF(g_offsets));
	g_offsets[g_submitted++] = offset;
	g_pending[g_count].callback = callback;
	g_pending[g_count++].arg = arg;
	return 0;
}

struct spdk_bdev *
spdk_bdev_desc_get_bdev(struct spdk_bdev_desc *desc)
{
	return (void *)1;
}

uint32_t
spdk_bdev_get_block_size(const struct spdk_bdev *bdev)
{
	return 512;
}

int
spdk_bdev_queue_io_wait(struct spdk_bdev *bdev, struct spdk_io_channel *ch,
			struct spdk_bdev_io_wait_entry *entry)
{
	assert(g_wait == NULL);
	g_wait = entry;
	return 0;
}

void
spdk_bdev_free_io(struct spdk_bdev_io *io)
{
}

static void
done(void *arg, int status)
{
	g_done++;
	g_failed += status != 0;
}

static void
progress(void *arg, uint64_t ready_bytes)
{
	assert(g_progress < SPDK_COUNTOF(g_ready));
	g_ready[g_progress++] = ready_bytes;
}

static void
complete_index(unsigned index)
{
	assert(index < g_count);
	spdk_bdev_io_completion_cb cb = g_pending[index].callback;
	void *arg = g_pending[index].arg;
	g_pending[index] = g_pending[--g_count];
	cb((void *)1, true, arg);
}

static void
drain(bool fail)
{
	while (g_count) {
		unsigned index = --g_count;
		spdk_bdev_io_completion_cb cb = g_pending[index].callback;
		void *arg = g_pending[index].arg;
		cb((void *)1, !fail, arg);
		fail = false;
	}
}

int
main(void)
{
	struct moe_store s = {0};
	unsigned char buffer[5 * 4096];
	s.layout.experts = 1;
	s.layout.complete = 1;
	s.layout.expert_stride = sizeof(buffer);
	s.io_size = 4096;
	/* One wait first, then exercise four-way completion and draining after error. */
	s.io_depth = 1;
	atomic_init(&s.removed, false);
	g_starve = true;
	assert(moe_store_read_expert(&s, 0, buffer, done, NULL) == 0);
	assert(g_wait && !g_done && s.active_io == 0);
	struct spdk_bdev_io_wait_entry *wait = g_wait;
	g_wait = NULL;
	g_starve = false;
	wait->cb_fn(wait->cb_arg);
	drain(false);
	assert(g_done == 1 && !g_failed && !s.active_io && s.read_bytes == sizeof(buffer));
	for (unsigned i = 0; i < sizeof(buffer); i++) {
		assert(buffer[i] == 0x5a);
	}
	s.io_depth = 4;
	assert(moe_store_read_expert(&s, 0, buffer, done, NULL) == 0);
	drain(true);
	assert(g_done == 2 && g_failed == 1 && !s.active_io && s.peak_io <= 4);
	assert(moe_store_read_expert(&s, 0, buffer, done, NULL) == 0);
	atomic_store(&s.removed, true);
	drain(false);
	assert(g_done == 3 && g_failed == 2 && !s.active_io);
	atomic_store(&s.removed, false);
	s.layout.expert_offset = 4096;
	s.layout.gate_bytes = 8192;
	s.layout.down_bytes = 4096;
	s.layout.expert_stride = 32768;
	for (unsigned matrix = 0; matrix < 3; matrix++) {
		unsigned length = matrix == 2 ? 4096 : 8192;
		g_low = 4096 + matrix * 8192;
		g_high = g_low + length;
		memset(buffer, 0xa5, sizeof(buffer));
		unsigned before = g_done;
		assert(!moe_store_read_matrix(&s, 0, matrix, buffer + 4096, done, NULL, NULL));
		drain(false);
		assert(g_done == before + 1);
		for (unsigned i = 0; i < sizeof(buffer); i++) {
			assert(buffer[i] == (i >= 4096 && i < 4096 + length ? 0x5a : 0xa5));
		}
		before = g_failed;
		assert(!moe_store_read_matrix(&s, 0, matrix, buffer + 4096, done, NULL, NULL));
		drain(true);
		assert(g_failed == before + 1 && !s.active_io);
		s.io_depth = 1;
		g_starve = true;
		uint64_t first_submit = UINT64_MAX;
		assert(!moe_store_read_matrix(&s, 0, matrix, buffer + 4096, done, NULL, &first_submit));
		assert(g_wait && first_submit == 0);
		wait = g_wait;
		g_wait = NULL;
		g_starve = false;
		wait->cb_fn(wait->cb_arg);
		assert(first_submit > 0);
		drain(false);
		assert(!s.active_io);
		assert(!moe_store_read_matrix(&s, 0, matrix, buffer + 4096, done, NULL, NULL));
		atomic_store(&s.removed, true);
		before = g_failed;
		drain(false);
		assert(g_failed == before + 1 && !s.active_io);
		atomic_store(&s.removed, false);
		s.io_depth = 4;
	}
	assert(moe_store_read_matrix(&s, 0, 3, buffer, done, NULL, NULL) == -EINVAL);
	g_high = 0;
	g_submitted = 0;
	unsigned char matrices[3][8192];
	struct moe_matrix_read batch[3];
	for (unsigned i = 0; i < 3; i++) {
		batch[i] = (struct moe_matrix_read){
			.expert = 0, .matrix = i, .buffer = matrices[i], .done = done,
		};
	}
	unsigned before = g_done;
	assert(!moe_store_read_batch(&s, batch, 3));
	/* Up is submitted before Gate completes, without interleaving Gate chunks. */
	assert(g_count == 4 && g_done == before);
	assert(g_offsets[0] == 4096 && g_offsets[1] == 8192);
	assert(g_offsets[2] == 12288 && g_offsets[3] == 16384);
	assert(moe_store_read_batch(&s, batch, 3) == -EBUSY);
	drain(false);
	assert(g_done == before + 3 && !s.active_io && g_offsets[4] == 20480);
	before = g_failed;
	assert(!moe_store_read_batch(&s, batch, 3));
	drain(true);
	assert(g_failed == before + 3 && !s.active_io);
	before = g_failed;
	assert(!moe_store_read_batch(&s, batch, 3));
	moe_store_cancel_reads(&s, -ECANCELED);
	drain(false);
	assert(g_failed == before + 3 && !s.active_io);
	/* Cancellation must retain a resource-wait callback until it runs. */
	s.io_depth = 1;
	g_starve = true;
	before = g_done;
	assert(!moe_store_read_batch(&s, batch, 3));
	moe_store_cancel_reads(&s, -ECANCELED);
	assert(g_wait && g_done == before + 2);
	wait = g_wait;
	g_wait = NULL;
	g_starve = false;
	wait->cb_fn(wait->cb_arg);
	assert(g_done == before + 3 && !s.active_io && !g_count);
	/* Publish only a contiguous prefix, while later reads remain in flight. */
	unsigned char *large = malloc(4 * MOE_STORE_MIB);
	assert(large);
	s.layout.gate_bytes = s.layout.down_bytes = 4 * MOE_STORE_MIB;
	s.layout.expert_stride = 12 * MOE_STORE_MIB;
	s.io_size = g_io_bytes = MOE_STORE_MIB;
	s.io_depth = 4;
	g_submitted = 0;
	batch[0] = (struct moe_matrix_read){
		.expert = 0, .matrix = 0, .buffer = large, .done = done, .progress = progress,
	};
	before = g_done;
	assert(!moe_store_read_batch(&s, batch, 1) && g_count == 4);
	complete_index(0);
	assert(g_progress == 1 && g_ready[0] == MOE_STORE_MIB && g_done == before);
	complete_index(2);
	assert(g_progress == 1 && g_done == before);
	complete_index(1);
	assert(g_progress == 2 && g_ready[1] == 3 * MOE_STORE_MIB && g_done == before);
	complete_index(0);
	assert(g_progress == 3 && g_ready[2] == 4 * MOE_STORE_MIB && g_done == before + 1);
	assert(!s.active_io && !g_count);
	free(large);
	puts("PASS: matrix bounds/canaries/failures, ENOMEM wait/retry, out-of-order completions, error drain, removal, exactly-once callbacks");
	return 0;
}
