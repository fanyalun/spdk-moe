/* SPDX-License-Identifier: BSD-3-Clause
 *
 * moe_subsystem.c — MoE bdev 权重加载与模块生命周期。
 *
 * 模块初始化时从 MOE_WEIGHT_DIR 读取预生成权重文件，创建名为
 * moe_bdev_0 的 bdev，供 NVMe-oF namespace 引用。
 */

#include "spdk/stdinc.h"

#include "spdk/bdev_module.h"
#include "spdk/log.h"

#include "moe_ffn/moe_config.h"

#include "bdev_moe.h"

#define MOE_BDEV_NAME "moe_bdev_0"
#define MOE_PATH_MAX 512

int g_moe_io_device;

static float *
moe_load_float_file(const char *path, size_t elems)
{
	FILE *fp;
	float *data;
	size_t nread;

	fp = fopen(path, "rb");
	if (fp == NULL) {
		return NULL;
	}

	data = malloc(elems * sizeof(float));
	if (data == NULL) {
		fclose(fp);
		return NULL;
	}

	nread = fread(data, sizeof(float), elems, fp);
	if (nread != elems) {
		fprintf(stderr, "failed to read %s\n", path);
		free(data);
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	return data;
}

static float *
moe_load_router(void)
{
	char path[MOE_PATH_MAX];

	snprintf(path, sizeof(path), "%s/W_router_%dx%d.bin",
		 MOE_WEIGHT_DIR, MOE_D_MODEL, MOE_ROUTER_COLS);

	return moe_load_float_file(path, MOE_ROUTER_ELEMS);
}

static int
moe_bdev_create_cb(void *io_device, void *ctx_buf)
{
	return 0;
}

static void
moe_bdev_destroy_cb(void *io_device, void *ctx_buf)
{
}

int
bdev_moe_initialize(void)
{
	float *W_router;
	struct spdk_bdev *bdev;
	const char *slots_text = getenv("MOE_CACHE_SLOTS");
	char *end;
	long slots = 2;

	if (slots_text != NULL) {
		errno = 0;
		slots = strtol(slots_text, &end, 10);
		if (errno != 0 || end == slots_text || *end != '\0' ||
		    slots < 1 || slots > MOE_NUM_EXPERTS) {
			SPDK_ERRLOG("MOE_CACHE_SLOTS must be between 1 and %d\n", MOE_NUM_EXPERTS);
			return -EINVAL;
		}
	}

	spdk_io_device_register(&g_moe_io_device, moe_bdev_create_cb, moe_bdev_destroy_cb,
				0, "moe_bdev");

	W_router = moe_load_router();
	if (W_router == NULL) {
		spdk_io_device_unregister(&g_moe_io_device, NULL);
		return -EIO;
	}

	bdev = bdev_moe_create(MOE_BDEV_NAME, W_router, (int)slots,
			       MOE_NUM_EXPERTS, MOE_D_MODEL, MOE_D_FF);
	if (bdev == NULL) {
		free(W_router);
		spdk_io_device_unregister(&g_moe_io_device, NULL);
		SPDK_ERRLOG("failed to create %s\n", MOE_BDEV_NAME);
		return -ENOMEM;
	}

	SPDK_NOTICELOG("created %s from %s\n", MOE_BDEV_NAME, MOE_WEIGHT_DIR);
	SPDK_NOTICELOG("MoE file cache: %ld slots, expert weights loaded on demand\n", slots);
	return 0;
}

static void
bdev_moe_unregister_cb(void *io_device)
{
	spdk_bdev_module_fini_done();
}

void
bdev_moe_finish(void)
{
	spdk_io_device_unregister(&g_moe_io_device, bdev_moe_unregister_cb);
}
