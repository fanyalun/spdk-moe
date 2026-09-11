#include "spdk/stdinc.h"
#include "spdk/log.h"
#include "moe_cache.h"

static float *load_weight(const char *dir, const char *name, int expert_id,
			  int d_model, int d_ff, size_t elements)
{
	char path[512];
	FILE *file;
	float *data;

	snprintf(path, sizeof(path), "%s/%s_%d_%dx%d.bin", dir, name, expert_id,
		 d_model, d_ff);
	file = fopen(path, "rb");
	if (file == NULL) {
		return NULL;
	}
	data = malloc(elements * sizeof(*data));
	if (data == NULL || fread(data, sizeof(*data), elements, file) != elements) {
		free(data);
		fclose(file);
		return NULL;
	}
	fclose(file);
	return data;
}

int moe_cache_init(struct moe_cache *cache, int capacity, int d_model, int d_ff,
		   const char *weight_dir)
{
	if (cache == NULL || capacity <= 0 || d_model <= 0 || d_ff <= 0) {
		return -EINVAL;
	}
	memset(cache, 0, sizeof(*cache));
	cache->entries = calloc((size_t)capacity, sizeof(*cache->entries));
	if (cache->entries == NULL) {
		return -ENOMEM;
	}
	cache->capacity = capacity;
	cache->d_model = d_model;
	cache->d_ff = d_ff;
	cache->weight_dir = weight_dir;
	return 0;
}

int moe_cache_get(struct moe_cache *cache, int expert_id,
		  const float **w_gate, const float **w_up, const float **w_down)
{
	struct moe_cache_entry *entry = NULL;
	int free_slot = -1;
	int victim = -1;
	int i;

	for (i = 0; i < cache->capacity; i++) {
		if (cache->entries[i].valid && cache->entries[i].expert_id == expert_id) {
			entry = &cache->entries[i];
			break;
		}
		if (!cache->entries[i].valid && free_slot < 0) {
			free_slot = i;
		}
		if (cache->entries[i].valid &&
		    (victim < 0 || cache->entries[i].last_used < cache->entries[victim].last_used)) {
			victim = i;
		}
	}
	if (entry == NULL) {
		entry = &cache->entries[free_slot >= 0 ? free_slot : victim];
		free(entry->w_gate);
		free(entry->w_up);
		free(entry->w_down);
		memset(entry, 0, sizeof(*entry));
		entry->expert_id = expert_id;
		entry->w_gate = load_weight(cache->weight_dir, "W_gate", expert_id,
					    cache->d_model, cache->d_ff,
					    (size_t)cache->d_model * cache->d_ff);
		entry->w_up = load_weight(cache->weight_dir, "W_up", expert_id,
					  cache->d_model, cache->d_ff,
					  (size_t)cache->d_model * cache->d_ff);
		entry->w_down = load_weight(cache->weight_dir, "W_down", expert_id,
					    cache->d_model, cache->d_ff,
					    (size_t)cache->d_ff * cache->d_model);
		if (entry->w_gate == NULL || entry->w_up == NULL || entry->w_down == NULL) {
			free(entry->w_gate);
			free(entry->w_up);
			free(entry->w_down);
			memset(entry, 0, sizeof(*entry));
			return -ENOENT;
		}
		entry->valid = true;
		SPDK_NOTICELOG("MoE cache miss: loaded expert %d\n", expert_id);
	} else {
		SPDK_NOTICELOG("MoE cache hit: expert %d\n", expert_id);
	}
	entry->last_used = ++cache->clock;
	*w_gate = entry->w_gate;
	*w_up = entry->w_up;
	*w_down = entry->w_down;
	return 0;
}

void moe_cache_destroy(struct moe_cache *cache)
{
	int i;
	if (cache == NULL) return;
	for (i = 0; i < cache->capacity; i++) {
		free(cache->entries[i].w_gate);
		free(cache->entries[i].w_up);
		free(cache->entries[i].w_down);
	}
	free(cache->entries);
	memset(cache, 0, sizeof(*cache));
}
