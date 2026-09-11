#ifndef MOE_CACHE_H
#define MOE_CACHE_H

#include <stdbool.h>
#include <stdint.h>

struct moe_cache_entry {
	int expert_id;
	float *w_gate;
	float *w_up;
	float *w_down;
	uint64_t last_used;
	bool valid;
};

struct moe_cache {
	struct moe_cache_entry *entries;
	int capacity;
	int d_model;
	int d_ff;
	const char *weight_dir;
	uint64_t clock;
};

int moe_cache_init(struct moe_cache *cache, int capacity, int d_model, int d_ff,
		   const char *weight_dir);
int moe_cache_get(struct moe_cache *cache, int expert_id,
		  const float **w_gate, const float **w_up, const float **w_down);
void moe_cache_destroy(struct moe_cache *cache);

#endif
