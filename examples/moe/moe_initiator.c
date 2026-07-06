/* SPDX-License-Identifier: BSD-3-Clause
 *
 * moe_initiator.c — MoE FFN NVMe-oF 发起端端到端测试。
 *
 * 程序连接 moe_tgt，发送 MOE_VENDOR_OPCODE 提交 input，随后通过标准
 * READ 取回 target 缓存的 output，并使用同一批权重在本地计算 expected。
 */

#include "spdk/stdinc.h"

#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/nvme_spec.h"
#include "spdk/string.h"

#include "moe_ffn/moe_config.h"
#include "moe_ffn/swiglu_moe.h"
#include "moe_ffn/test_utils.h"

#define MOE_PATH_MAX 512
#define MOE_PRINT_EDGE 5

struct moe_weights {
	float *W_router;
	const float *W_gate[MOE_NUM_EXPERTS];
	const float *W_up[MOE_NUM_EXPERTS];
	const float *W_down[MOE_NUM_EXPERTS];
};

struct moe_completion {
	bool done;
	bool failed;
};

static struct spdk_nvme_ctrlr *g_ctrlr;
static struct spdk_nvme_ns *g_ns;

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
moe_load_expert_weight(const char *name, int expert_id, size_t elems)
{
	char path[MOE_PATH_MAX];

	snprintf(path, sizeof(path), "%s/%s_%d_%dx%d.bin",
		 MOE_WEIGHT_DIR, name, expert_id, MOE_D_MODEL, MOE_D_FF);
	return moe_load_float_file(path, elems);
}

static void
moe_load_weights(struct moe_weights *weights)
{
	char path[MOE_PATH_MAX];
	int e;

	snprintf(path, sizeof(path), "%s/W_router_%dx%d.bin",
		 MOE_WEIGHT_DIR, MOE_D_MODEL, MOE_ROUTER_COLS);
	weights->W_router = moe_load_float_file(path, MOE_ROUTER_ELEMS);

	for (e = 0; e < MOE_NUM_EXPERTS; e++) {
		weights->W_gate[e] = moe_load_expert_weight("W_gate", e, MOE_EXPERT_ELEMS);
		weights->W_up[e] = moe_load_expert_weight("W_up", e, MOE_EXPERT_ELEMS);
		weights->W_down[e] = moe_load_expert_weight("W_down", e, MOE_DOWN_ELEMS);
	}
}

static void
moe_free_weights(struct moe_weights *weights)
{
	int e;

	free(weights->W_router);
	for (e = 0; e < MOE_NUM_EXPERTS; e++) {
		free((void *)weights->W_gate[e]);
		free((void *)weights->W_up[e]);
		free((void *)weights->W_down[e]);
	}
}

static void
moe_fill_input(float *input)
{
	int i;

	srand(MOE_SEED_DEFAULT);
	for (i = 0; i < MOE_D_MODEL; i++) {
		input[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
	}
}

static bool
probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	 struct spdk_nvme_ctrlr_opts *opts)
{
	return true;
}

static void
attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	  struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts)
{
	int nsid;
	struct spdk_nvme_ns *ns;

	g_ctrlr = ctrlr;
	for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
	     nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
		ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
		if (ns != NULL && spdk_nvme_ns_is_active(ns)) {
			g_ns = ns;
			return;
		}
	}
}

static void
io_complete(void *arg, const struct spdk_nvme_cpl *cpl)
{
	struct moe_completion *completion = arg;

	completion->done = true;
	if (spdk_nvme_cpl_is_error(cpl)) {
		completion->failed = true;
	}
}

static int
process_admin_completions(void)
{
	int rc;

	if (g_ctrlr == NULL) {
		return 0;
	}

	rc = spdk_nvme_ctrlr_process_admin_completions(g_ctrlr);
	return rc < 0 ? -EIO : 0;
}

static int
wait_for_completion(struct spdk_nvme_qpair *qpair, struct moe_completion *completion)
{
	int rc;

	while (!completion->done) {
		rc = spdk_nvme_qpair_process_completions(qpair, 0);
		if (rc < 0) {
			return -EIO;
		}
		rc = process_admin_completions();
		if (rc != 0) {
			return rc;
		}
	}

	return completion->failed ? -EIO : 0;
}

static float
max_abs_error(const float *a, const float *b, int len, int *first_diff)
{
	float max_err = 0.0f;
	int i;

	*first_diff = -1;
	for (i = 0; i < len; i++) {
		float err = fabsf(a[i] - b[i]);
		if (err > max_err) {
			max_err = err;
		}
		if (*first_diff < 0 && err >= MOE_MAX_ABS_ERROR) {
			*first_diff = i;
		}
	}

	return max_err;
}

int
main(int argc, char **argv)
{
	struct spdk_env_opts env_opts;
	struct spdk_nvme_transport_id trid = {};
	struct spdk_nvme_qpair *qpair;
	struct spdk_nvme_detach_ctx *detach_ctx = NULL;
	struct spdk_nvme_cmd cmd = {};
	struct moe_completion completion;
	struct moe_weights weights = {};
	float *input, *output, *expected;
	float err;
	int rc, first_diff;

	spdk_env_opts_init(&env_opts);
	env_opts.name = "moe_initiator";
	env_opts.no_huge = true;
	env_opts.mem_size = 512;
	if (spdk_env_init(&env_opts) < 0) {
		fprintf(stderr, "Unable to initialize SPDK env\n");
		return 1;
	}

	moe_load_weights(&weights);
	input = spdk_zmalloc(MOE_INPUT_BYTES, 0x1000, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	output = spdk_zmalloc(MOE_INPUT_BYTES, 0x1000, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	expected = malloc(MOE_INPUT_BYTES);
	assert(input != NULL && output != NULL && expected != NULL);
	moe_fill_input(input);

	spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_TCP);
	snprintf(trid.traddr, sizeof(trid.traddr), "%s", MOE_TARGET_ADDR);
	snprintf(trid.trsvcid, sizeof(trid.trsvcid), "%d", MOE_TARGET_PORT);
	snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", MOE_NQN);
	trid.adrfam = SPDK_NVMF_ADRFAM_IPV4;

	rc = spdk_nvme_probe(&trid, NULL, probe_cb, attach_cb, NULL);
	if (rc != 0 || g_ctrlr == NULL || g_ns == NULL) {
		fprintf(stderr, "NVMe-oF 连接失败: %s:%d %s\n",
			MOE_TARGET_ADDR, MOE_TARGET_PORT, MOE_NQN);
		rc = 1;
		goto out;
	}

	qpair = spdk_nvme_ctrlr_alloc_io_qpair(g_ctrlr, NULL, 0);
	if (qpair == NULL) {
		fprintf(stderr, "spdk_nvme_ctrlr_alloc_io_qpair failed\n");
		rc = 1;
		goto out;
	}

	cmd.opc = MOE_VENDOR_OPCODE;
	cmd.nsid = spdk_nvme_ns_get_id(g_ns);
	completion = (struct moe_completion){};
	rc = spdk_nvme_ctrlr_cmd_io_raw(g_ctrlr, qpair, &cmd, input, MOE_INPUT_BYTES,
					 io_complete, &completion);
	if (rc != 0 || wait_for_completion(qpair, &completion) != 0) {
		fprintf(stderr, "MoE vendor command failed\n");
		rc = 1;
		goto qpair_out;
	}

	completion = (struct moe_completion){};
	rc = spdk_nvme_ns_cmd_read(g_ns, qpair, output, 0, 1, io_complete, &completion, 0);
	if (rc != 0 || wait_for_completion(qpair, &completion) != 0) {
		fprintf(stderr, "MoE output read failed\n");
		rc = 1;
		goto qpair_out;
	}

	swiglu_moe(input, weights.W_router, weights.W_gate, weights.W_up, weights.W_down,
		   MOE_NUM_EXPERTS, MOE_TOP_K, MOE_D_MODEL, MOE_D_FF, expected);
	err = max_abs_error(output, expected, MOE_D_MODEL, &first_diff);

	printf("============================================================\n");
	printf("  MoE FFN SPDK 端到端测试报告\n");
	printf("  d_model = %d, d_ff = %d,\n", MOE_D_MODEL, MOE_D_FF);
	printf("  num_experts = %d, top_k = %d\n", MOE_NUM_EXPERTS, MOE_TOP_K);
	printf("  target: %s:%d\n", MOE_TARGET_ADDR, MOE_TARGET_PORT);
	printf("============================================================\n\n");
	print_vector(output, MOE_D_MODEL, MOE_PRINT_EDGE, MOE_PRINT_EDGE, "SPDK 返回 output 向量");
	printf("\n");
	print_vector(expected, MOE_D_MODEL, MOE_PRINT_EDGE, MOE_PRINT_EDGE, "本地计算 expected 向量");
	printf("\n");
	print_stats(output, MOE_D_MODEL, "SPDK output");
	printf("\n");
	print_stats(expected, MOE_D_MODEL, "本地 expected");
	printf("\n逐元素校验:\n");
	printf("  max_absolute_error = %.6f\n", err);
	printf("  容差 = %.6f\n\n", MOE_MAX_ABS_ERROR);
	if (err < MOE_MAX_ABS_ERROR) {
		printf("结果: PASS\n\n");
		rc = 0;
	} else {
		printf("结果: FAIL\n");
		if (first_diff >= 0) {
			printf("首个差异元素: index=%d output=%.6f expected=%.6f\n",
			       first_diff, output[first_diff], expected[first_diff]);
		}
		rc = 1;
	}
	printf("测试完成。\n");

qpair_out:
	spdk_nvme_ctrlr_free_io_qpair(qpair);
out:
	if (g_ctrlr != NULL) {
		spdk_nvme_detach_async(g_ctrlr, &detach_ctx);
		if (detach_ctx != NULL) {
			spdk_nvme_detach_poll(detach_ctx);
		}
	}
	spdk_free(input);
	spdk_free(output);
	free(expected);
	moe_free_weights(&weights);
	spdk_env_fini();
	return rc;
}
