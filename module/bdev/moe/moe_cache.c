/* SPDX-License-Identifier: BSD-3-Clause */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "moe_cache.h"

static void *
cpu_allocate(size_t bytes)
{
	const size_t page = 2 * 1024 * 1024;
	void *buffer;

	if (bytes > SIZE_MAX - (page - 1)) {
		return NULL;
	}
	size_t length = (bytes + page - 1) / page * page;
	if (posix_memalign(&buffer, page, length)) {
		return NULL;
	}
	// THP is advisory; ordinary pages are valid for CPU-only decoded weights.
	(void)madvise(buffer, length, MADV_HUGEPAGE);
	return buffer;
}

int
moe_cache_init_buffers(struct moe_cache *c, int capacity, int d, int h,
		       const char *directory, size_t gate_bytes, size_t buffer_bytes,
		       void *(*allocate)(size_t), void (*release)(void *))
{
	if (!c || capacity <= 0 || d <= 0 || h <= 0 || !directory || !allocate || !release ||
	    !gate_bytes || gate_bytes > SIZE_MAX / 2 || buffer_bytes <= 2 * gate_bytes ||
	    (size_t)capacity > SIZE_MAX / buffer_bytes) {
		return -EINVAL;
	}
	memset(c, 0, sizeof(*c));
	c->entries = calloc(capacity, sizeof(*c->entries));
	if (!c->entries) {
		return -ENOMEM;
	}
	c->capacity = capacity;
	c->d_model = d;
	c->d_ff = h;
	c->weight_dir = directory;
	c->buffer_bytes = buffer_bytes;
	c->release = release;
	for (int i = 0; i < capacity; i++) {
		struct moe_cache_entry *e = &c->entries[i];
		e->buffer = allocate(buffer_bytes);
		if (!e->buffer) {
			moe_cache_destroy(c);
			return -ENOMEM;
		}
		memset(e->buffer, 0, buffer_bytes);
		e->w_gate = e->buffer;
		e->w_up = (float *)((char *)e->buffer + gate_bytes);
		e->w_down = (float *)((char *)e->buffer + 2 * gate_bytes);
		e->expert_id = -1;
	}
	return 0;
}

int
moe_cache_init_cpu_buffers(struct moe_cache *c, int capacity, int d, int h,
			   const char *directory, size_t gate_bytes, size_t buffer_bytes)
{
	int rc = moe_cache_init_buffers(c, capacity, d, h, directory, gate_bytes, buffer_bytes,
					cpu_allocate, free);
	if (!rc) {
		c->cpu_memory = true;
	}
	return rc;
}

int
moe_cache_find(struct moe_cache *c, int expert_id)
{
	for (int i = 0; i < c->capacity; i++) {
		if (c->entries[i].state == MOE_CACHE_READY && c->entries[i].expert_id == expert_id) {
			c->hits++;
			c->entries[i].last_used = ++c->clock;
			return i;
		}
	}
	return -1;
}

int
moe_cache_reserve(struct moe_cache *c, int expert_id)
{
	int victim = -1;
	for (int i = 0; i < c->capacity; i++) {
		struct moe_cache_entry *e = &c->entries[i];
		if (e->refs || e->state == MOE_CACHE_LOADING || e->state == MOE_CACHE_IN_USE) {
			continue;
		}
		if (e->state == MOE_CACHE_EMPTY) {
			victim = i;
			break;
		}
		if (victim < 0 || e->last_used < c->entries[victim].last_used) {
			victim = i;
		}
	}
	if (victim >= 0) {
		struct moe_cache_entry *e = &c->entries[victim];
		c->misses++;
		c->evictions += e->state == MOE_CACHE_READY;
		e->state = MOE_CACHE_LOADING;
		e->expert_id = expert_id;
		e->refs = 1;
		e->last_used = ++c->clock;
	}
	return victim;
}

void
moe_cache_destroy(struct moe_cache *c)
{
	if (!c) {
		return;
	}
	for (int i = 0; c->entries && i < c->capacity; i++) {
		if (c->entries[i].buffer) {
			c->release(c->entries[i].buffer);
		}
	}
	free(c->entries);
	memset(c, 0, sizeof(*c));
}
