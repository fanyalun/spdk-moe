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
	assert(fp != NULL);

	data = malloc(elems * sizeof(float));
	assert(data != NULL);

	nread = fread(data, sizeof(float), elems, fp);
	if (nread != elems) {
		fprintf(stderr, "failed to read %s\n", path);
		abort();
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

static float *
moe_load_expert_weight(const char *name, int expert_id, size_t elems)
{
	char path[MOE_PATH_MAX];

	snprintf(path, sizeof(path), "%s/%s_%d_%dx%d.bin",
		 MOE_WEIGHT_DIR, name, expert_id, MOE_D_MODEL, MOE_D_FF);

	return moe_load_float_file(path, elems);
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
	const float *W_gate[MOE_NUM_EXPERTS];
	const float *W_up[MOE_NUM_EXPERTS];
	const float *W_down[MOE_NUM_EXPERTS];
	struct spdk_bdev *bdev;
	int e;

	spdk_io_device_register(&g_moe_io_device, moe_bdev_create_cb, moe_bdev_destroy_cb,
				0, "moe_bdev");

	W_router = moe_load_router();
	for (e = 0; e < MOE_NUM_EXPERTS; e++) {
		W_gate[e] = moe_load_expert_weight("W_gate", e, MOE_EXPERT_ELEMS);
		W_up[e] = moe_load_expert_weight("W_up", e, MOE_EXPERT_ELEMS);
		W_down[e] = moe_load_expert_weight("W_down", e, MOE_DOWN_ELEMS);
	}

	bdev = bdev_moe_create(MOE_BDEV_NAME, W_router, W_gate, W_up, W_down,
			       MOE_NUM_EXPERTS, MOE_D_MODEL, MOE_D_FF);
	if (bdev == NULL) {
		SPDK_ERRLOG("failed to create %s\n", MOE_BDEV_NAME);
		return -ENOMEM;
	}

	SPDK_NOTICELOG("created %s from %s\n", MOE_BDEV_NAME, MOE_WEIGHT_DIR);
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
