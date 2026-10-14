/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/bdev.h"

int __real_posix_memalign(void **buffer, size_t align, size_t size);
int __wrap_posix_memalign(void **buffer, size_t align, size_t size);

int
__wrap_posix_memalign(void **buffer, size_t align, size_t size)
{
	static unsigned slots;
	if (getenv("MOE_TEST_STARTUP_ALLOC_FAIL") && align == 2 * 1024 * 1024 &&
	    size == 2 * 1024 * 1024 && ++slots == 8) {
		puts("startup_allocation_failure_injected=1");
		return ENOMEM;
	}
	return __real_posix_memalign(buffer, align, size);
}

void *__real_spdk_dma_zmalloc_socket(size_t size, size_t align, uint64_t *unused, int numa_id);
void *__wrap_spdk_dma_zmalloc_socket(size_t size, size_t align, uint64_t *unused, int numa_id);

void *
__wrap_spdk_dma_zmalloc_socket(size_t size, size_t align, uint64_t *unused, int numa_id)
{
	static unsigned slots;
	if (getenv("MOE_TEST_STARTUP_ALLOC_FAIL") && size == 1024 * 1024 && ++slots == 8) {
		puts("startup_allocation_failure_injected=1");
		return NULL;
	}
	return __real_spdk_dma_zmalloc_socket(size, align, unused, numa_id);
}

int __real_spdk_bdev_read(struct spdk_bdev_desc *desc, struct spdk_io_channel *ch,
			 void *buf, uint64_t offset, uint64_t bytes,
			 spdk_bdev_io_completion_cb cb, void *arg);
int __wrap_spdk_bdev_read(struct spdk_bdev_desc *desc, struct spdk_io_channel *ch,
			 void *buf, uint64_t offset, uint64_t bytes,
			 spdk_bdev_io_completion_cb cb, void *arg);

int
__wrap_spdk_bdev_read(struct spdk_bdev_desc *desc, struct spdk_io_channel *ch,
		     void *buf, uint64_t offset, uint64_t bytes,
		     spdk_bdev_io_completion_cb cb, void *arg)
{
	static bool failed;
	if (getenv("MOE_TEST_STARTUP_READ_FAIL") && !failed && offset >= 2 * 1024 * 1024) {
		failed = true;
		puts("startup_read_failure_injected=1");
		return -EIO;
	}
	return __real_spdk_bdev_read(desc, ch, buf, offset, bytes, cb, arg);
}
