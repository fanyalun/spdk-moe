/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_CACHE_H
#define MOE_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum moe_cache_state { MOE_CACHE_EMPTY, MOE_CACHE_LOADING, MOE_CACHE_READY, MOE_CACHE_IN_USE };

struct moe_cache_entry {
	int expert_id;
	float *w_gate, *w_up, *w_down;
	void *buffer;
	uint64_t last_used;
	enum moe_cache_state state;
	unsigned refs;
};

struct moe_cache {
	struct moe_cache_entry *entries;
	int capacity, d_model, d_ff;
	const char *weight_dir;
	uint64_t clock, hits, misses, evictions, read_bytes;
	size_t buffer_bytes;
	void (*release)(void *);
};

int moe_cache_init(struct moe_cache *cache, int capacity, int d_model, int d_ff,
		   const char *weight_dir);
int moe_cache_init_buffers(struct moe_cache *cache, int capacity, int d_model, int d_ff,
			   const char *directory, size_t gate_bytes, size_t buffer_bytes,
			   void *(*allocate)(size_t), void (*release)(void *));
int moe_cache_find(struct moe_cache *cache, int expert_id);
int moe_cache_reserve(struct moe_cache *cache, int expert_id);
int moe_cache_load_file(struct moe_cache *cache, int slot);
int moe_cache_get(struct moe_cache *cache, int expert_id,
		  const float **w_gate, const float **w_up, const float **w_down);
void moe_cache_destroy(struct moe_cache *cache);

#endif
