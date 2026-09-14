// SPDX-License-Identifier: BSD-3-Clause
#include <assert.h>
#include "../../dpdk/lib/eal/linux/eal_moe_memory.h"

int
main(void)
{
	const uint64_t pool = UINT64_C(1536) * 1024 * 1024;
	uint32_t pages = 100000;
	assert(eal_moe_limit_pages(pool, 2097152, 1, &pages) == 0 && pages == 768);
	pages = 512;
	assert(eal_moe_limit_pages(pool, 2097152, 1, &pages) == 0 && pages == 512);
	pages = 0;
	assert(eal_moe_limit_pages(pool, 2097152, 1, &pages) == 0 && pages == 0);
	assert(eal_moe_limit_pages(pool, 1073741824, 1, &pages) != 0);
	assert(eal_moe_limit_pages(pool, 2097152, 2, &pages) != 0);
	assert(eal_moe_limit_pages(0, 2097152, 1, &pages) != 0);
	assert(eal_moe_limit_pages(pool + 1, 2097152, 1, &pages) != 0);
	assert(eal_moe_limit_pages(UINT64_C(1) << 63, 2097152, 1, &pages) != 0);
	return 0;
}
