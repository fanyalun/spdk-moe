/* SPDX-License-Identifier: BSD-3-Clause */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "moe_cache.h"

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
moe_cache_init(struct moe_cache *c, int capacity, int d, int h, const char *directory)
{
	if (d <= 0 || h <= 0 || (size_t)d > SIZE_MAX / sizeof(float) / (size_t)h / 3) {
		return -EINVAL;
	}
	size_t bytes = (size_t)d * h * sizeof(float);
	return moe_cache_init_buffers(c, capacity, d, h, directory, bytes, 3 * bytes, malloc, free);
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

int
moe_cache_load_file(struct moe_cache *c, int slot)
{
	struct moe_cache_entry *e = &c->entries[slot];
	const char *names[] = {"W_gate", "W_up", "W_down"};
	float *buffers[] = {e->w_gate, e->w_up, e->w_down};
	size_t elements = (size_t)c->d_model * c->d_ff;
	char path[4096];

	for (int m = 0; m < 3; m++) {
		struct stat st;
		int n = snprintf(path, sizeof(path), "%s/%s_%d_%dx%d.bin", c->weight_dir, names[m],
				 e->expert_id, c->d_model, c->d_ff);
		if (n < 0 || (size_t)n >= sizeof(path)) {
			return -ENAMETOOLONG;
		}
		FILE *f = fopen(path, "rb");
		if (!f) {
			return -errno;
		}
		if (fstat(fileno(f), &st) || st.st_size != (off_t)(elements * sizeof(float))) {
			fclose(f);
			return -EINVAL;
		}
		size_t got = fread(buffers[m], sizeof(float), elements, f);
		c->read_bytes += got * sizeof(float);
		int error = ferror(f);
		fclose(f);
		if (error || got != elements) {
			return -EIO;
		}
	}
	return 0;
}

int
moe_cache_get(struct moe_cache *c, int expert_id,
	      const float **gate, const float **up, const float **down)
{
	if (!c || !c->entries || expert_id < 0 || !gate || !up || !down) {
		return -EINVAL;
	}
	int slot = moe_cache_find(c, expert_id);
	if (slot < 0) {
		slot = moe_cache_reserve(c, expert_id);
		if (slot < 0) {
			return -EAGAIN;
		}
		int rc = moe_cache_load_file(c, slot);
		c->entries[slot].refs = 0;
		c->entries[slot].state = rc ? MOE_CACHE_EMPTY : MOE_CACHE_READY;
		if (rc) {
			return rc;
		}
	}
	*gate = c->entries[slot].w_gate;
	*up = c->entries[slot].w_up;
	*down = c->entries[slot].w_down;
	return 0;
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
