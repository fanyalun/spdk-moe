/* SPDX-License-Identifier: BSD-3-Clause */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "moe_cache.h"

static unsigned allocations, releases;

static void *
fail_second(size_t bytes)
{
	return ++allocations == 2 ? NULL : malloc(bytes);
}

static void
release_counted(void *buffer)
{
	releases++;
	free(buffer);
}

int
main(void)
{
	struct moe_cache cache;

	assert(!moe_cache_init_cpu_buffers(&cache, 2, 2, 4, ".", 32, 96));
	assert(cache.cpu_memory && cache.capacity == 2);
	for (int i = 0; i < 2; i++) {
		assert((uintptr_t)cache.entries[i].buffer % (2 * 1024 * 1024) == 0);
		assert(cache.entries[i].w_up == cache.entries[i].w_gate + 8);
		assert(cache.entries[i].w_down == cache.entries[i].w_gate + 16);
		for (unsigned j = 0; j < 24; j++) {
			assert(cache.entries[i].w_gate[j] == 0);
		}
	}
	int first = moe_cache_reserve(&cache, 10);
	int second = moe_cache_reserve(&cache, 11);
	assert(first >= 0 && second >= 0 && first != second);
	assert(moe_cache_reserve(&cache, 12) == -1);
	cache.entries[first].state = MOE_CACHE_READY;
	cache.entries[first].refs = 0;
	assert(moe_cache_find(&cache, 10) == first);
	cache.entries[first].refs = 1;
	assert(moe_cache_reserve(&cache, 12) == -1);
	cache.entries[first].refs = 0;
	assert(moe_cache_reserve(&cache, 12) == first);
	assert(cache.entries[second].expert_id == 11);
	assert(cache.entries[first].state == MOE_CACHE_LOADING);
	moe_cache_destroy(&cache);
	assert(!cache.entries && !cache.capacity);
	assert(moe_cache_init_buffers(&cache, 2, 2, 4, ".", 32, 96,
				      fail_second, release_counted) == -ENOMEM);
	assert(allocations == 2 && releases == 1 && !cache.entries);
	puts("PASS: aligned/prefaulted CPU cache, reference protection, eviction, partial-allocation cleanup");
	return 0;
}
