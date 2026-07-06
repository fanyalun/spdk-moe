/* SPDX-License-Identifier: BSD-3-Clause
 *
 * bdev_moe.c — MoE FFN 自定义 bdev。
 *
 * 该模块提供一个标准 NVM namespace，并通过 vendor opcode 接收输入向量，
 * 在 target 侧运行 SwiGLU MoE FFN。由于 MOE_VENDOR_OPCODE=0xC1 是
 * host-to-controller 方向命令，计算结果缓存到 bdev 中，随后由标准 READ 返回。
 */

#include "spdk/stdinc.h"

#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/nvme_spec.h"

#include "moe_ffn/moe_config.h"
#include "moe_ffn/swiglu_moe.h"

#include "bdev_moe.h"

struct moe_bdev {
	struct spdk_bdev bdev;
	TAILQ_ENTRY(moe_bdev) link;

	const float *W_router;
	const float *W_gate[MOE_NUM_EXPERTS];
	const float *W_up[MOE_NUM_EXPERTS];
	const float *W_down[MOE_NUM_EXPERTS];

	float *last_output;
	bool output_valid;
	int num_experts;
	int d_model;
	int d_ff;
};

struct moe_io_channel {
	int unused;
};

static TAILQ_HEAD(, moe_bdev) g_moe_bdevs = TAILQ_HEAD_INITIALIZER(g_moe_bdevs);

static int bdev_moe_get_ctx_size(void);

static struct spdk_bdev_module g_moe_if = {
	.name = "moe",
	.module_init = bdev_moe_initialize,
	.module_fini = bdev_moe_finish,
	.async_fini = true,
	.get_ctx_size = bdev_moe_get_ctx_size,
};

SPDK_BDEV_MODULE_REGISTER(moe, &g_moe_if)

static int
bdev_moe_get_ctx_size(void)
{
	return 0;
}

static int
bdev_moe_destruct(void *ctx)
{
	struct moe_bdev *moe = ctx;
	int e;

	TAILQ_REMOVE(&g_moe_bdevs, moe, link);

	free((void *)moe->W_router);
	for (e = 0; e < moe->num_experts; e++) {
		free((void *)moe->W_gate[e]);
		free((void *)moe->W_up[e]);
		free((void *)moe->W_down[e]);
	}
	free(moe->last_output);
	free(moe->bdev.name);
	free(moe);

	return 0;
}

static void
moe_copy_to_iovs(struct iovec *iovs, int iovcnt, const void *src, size_t len)
{
	const uint8_t *p = src;
	size_t copied = 0;
	int i;

	for (i = 0; i < iovcnt && copied < len; i++) {
		size_t n = spdk_min(iovs[i].iov_len, len - copied);
		memcpy(iovs[i].iov_base, p + copied, n);
		copied += n;
	}
}

static void
moe_zero_iovs(struct iovec *iovs, int iovcnt)
{
	int i;

	for (i = 0; i < iovcnt; i++) {
		memset(iovs[i].iov_base, 0, iovs[i].iov_len);
	}
}

static size_t
moe_iovs_len(const struct iovec *iovs, int iovcnt)
{
	size_t len = 0;
	int i;

	for (i = 0; i < iovcnt; i++) {
		len += iovs[i].iov_len;
	}

	return len;
}

static void
bdev_moe_handle_vendor(struct moe_bdev *moe, struct spdk_bdev_io *bdev_io,
		       const void *buf, size_t nbytes)
{
	if (nbytes != MOE_INPUT_BYTES) {
		SPDK_ERRLOG("MoE vendor command data size %zu != %d\n", nbytes, MOE_INPUT_BYTES);
		spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_FAILED);
		return;
	}

	swiglu_moe(buf, moe->W_router, moe->W_gate, moe->W_up, moe->W_down,
		   MOE_NUM_EXPERTS, MOE_TOP_K, MOE_D_MODEL, MOE_D_FF, moe->last_output);
	moe->output_valid = true;
	spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_SUCCESS);
}

static void
bdev_moe_submit_request(struct spdk_io_channel *ch, struct spdk_bdev_io *bdev_io)
{
	struct moe_bdev *moe = bdev_io->bdev->ctxt;
	struct spdk_bdev_io_nvme_passthru_params *pt;

	switch (bdev_io->type) {
	case SPDK_BDEV_IO_TYPE_READ:
		if (moe->output_valid) {
			moe_copy_to_iovs(bdev_io->u.bdev.iovs, bdev_io->u.bdev.iovcnt,
					 moe->last_output, MOE_INPUT_BYTES);
		} else {
			moe_zero_iovs(bdev_io->u.bdev.iovs, bdev_io->u.bdev.iovcnt);
		}
		spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_SUCCESS);
		break;
	case SPDK_BDEV_IO_TYPE_WRITE:
	case SPDK_BDEV_IO_TYPE_RESET:
		spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_SUCCESS);
		break;
	case SPDK_BDEV_IO_TYPE_NVME_IO:
	case SPDK_BDEV_IO_TYPE_NVME_IO_MD:
		pt = &bdev_io->u.nvme_passthru;
		if (pt->cmd.opc != MOE_VENDOR_OPCODE) {
			spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_FAILED);
			break;
		}
		bdev_moe_handle_vendor(moe, bdev_io, pt->buf, pt->nbytes);
		break;
	case SPDK_BDEV_IO_TYPE_NVME_IOV_MD:
		pt = &bdev_io->u.nvme_passthru;
		if (pt->cmd.opc != MOE_VENDOR_OPCODE) {
			spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_FAILED);
			break;
		}
		if (pt->iovcnt != 1 || moe_iovs_len(pt->iovs, pt->iovcnt) != MOE_INPUT_BYTES) {
			SPDK_ERRLOG("MoE vendor command expects one %d-byte SGL\n", MOE_INPUT_BYTES);
			spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_FAILED);
			break;
		}
		bdev_moe_handle_vendor(moe, bdev_io, pt->iovs[0].iov_base, MOE_INPUT_BYTES);
		break;
	default:
		spdk_bdev_io_complete(bdev_io, SPDK_BDEV_IO_STATUS_FAILED);
		break;
	}
}

static bool
bdev_moe_io_type_supported(void *ctx, enum spdk_bdev_io_type io_type)
{
	switch (io_type) {
	case SPDK_BDEV_IO_TYPE_READ:
	case SPDK_BDEV_IO_TYPE_WRITE:
	case SPDK_BDEV_IO_TYPE_RESET:
	case SPDK_BDEV_IO_TYPE_NVME_IO:
	case SPDK_BDEV_IO_TYPE_NVME_IO_MD:
	case SPDK_BDEV_IO_TYPE_NVME_IOV_MD:
		return true;
	default:
		return false;
	}
}

static struct spdk_io_channel *
bdev_moe_get_io_channel(void *ctx)
{
	extern int g_moe_io_device;

	return spdk_get_io_channel(&g_moe_io_device);
}

static const struct spdk_bdev_fn_table g_moe_fn_table = {
	.destruct = bdev_moe_destruct,
	.submit_request = bdev_moe_submit_request,
	.io_type_supported = bdev_moe_io_type_supported,
	.get_io_channel = bdev_moe_get_io_channel,
};

struct spdk_bdev *
bdev_moe_create(const char *name, const float *W_router, const float *W_gate[],
		const float *W_up[], const float *W_down[], int num_experts,
		int d_model, int d_ff)
{
	struct moe_bdev *moe;
	int e, rc;

	moe = calloc(1, sizeof(*moe));
	assert(moe != NULL);

	moe->bdev.name = strdup(name);
	assert(moe->bdev.name != NULL);
	moe->bdev.product_name = "MoE FFN disk";
	moe->bdev.blocklen = MOE_INPUT_BYTES;
	moe->bdev.phys_blocklen = MOE_INPUT_BYTES;
	moe->bdev.blockcnt = 1;
	moe->bdev.ctxt = moe;
	moe->bdev.fn_table = &g_moe_fn_table;
	moe->bdev.module = &g_moe_if;

	moe->W_router = W_router;
	for (e = 0; e < num_experts; e++) {
		moe->W_gate[e] = W_gate[e];
		moe->W_up[e] = W_up[e];
		moe->W_down[e] = W_down[e];
	}
	moe->num_experts = num_experts;
	moe->d_model = d_model;
	moe->d_ff = d_ff;
	moe->last_output = malloc(MOE_INPUT_BYTES);
	assert(moe->last_output != NULL);

	rc = spdk_bdev_register(&moe->bdev);
	if (rc != 0) {
		bdev_moe_destruct(moe);
		return NULL;
	}

	TAILQ_INSERT_TAIL(&g_moe_bdevs, moe, link);
	return &moe->bdev;
}

SPDK_LOG_REGISTER_COMPONENT(bdev_moe)
