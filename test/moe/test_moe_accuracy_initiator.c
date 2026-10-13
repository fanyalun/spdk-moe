/* SPDX-License-Identifier: BSD-3-Clause */

#define MOE_INDEPENDENT_REFERENCE
int moe_single_accuracy_main(int argc, char **argv);
#define main moe_single_accuracy_main
#include "../../examples/moe/moe_initiator.c"
#undef main

int
main(int argc, char **argv)
{
	struct spdk_env_opts env;
	struct spdk_nvme_transport_id trid = {};
	struct spdk_nvme_qpair *qpair = NULL;
	struct spdk_nvme_detach_ctx *detach = NULL;
	struct moe_weights weights = {};
	float *input = NULL, *output = NULL, *expected = NULL;
	unsigned count = 1, seed = 42;
	int rc = 1;

	if (argc > 3) {
		fprintf(stderr, "usage: %s [count [seed]]\n", argv[0]);
		return 1;
	}
	for (int i = 1; i < argc; i++) {
		char *end;
		errno = 0;
		unsigned long value = strtoul(argv[i], &end, 10);
		if (errno || *end || end == argv[i] || value > UINT_MAX || (i == 1 && !value)) {
			return 1;
		}
		if (i == 1) {
			count = value;
		} else {
			seed = value;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	spdk_env_opts_init(&env);
	env.opts_size = sizeof(env);
	env.name = "moe_accuracy_batch";
	env.no_huge = true;
	env.no_pci = true;
	env.mem_size = 512;
	env.core_mask = getenv("MOE_INITIATOR_CPU_MASK");
	if (!env.core_mask) {
		env.core_mask = "0x1";
	}
	if (spdk_env_init(&env) < 0) {
		return 1;
	}
	moe_load_weights(&weights);
	input = spdk_dma_zmalloc(MOE_INPUT_BYTES, 4096, NULL);
	output = spdk_dma_zmalloc(MOE_INPUT_BYTES, 4096, NULL);
	expected = malloc(MOE_INPUT_BYTES);
	if (!weights.W_router || !input || !output || !expected) {
		goto out;
	}
	spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_TCP);
	snprintf(trid.traddr, sizeof(trid.traddr), "%s", MOE_TARGET_ADDR);
	snprintf(trid.trsvcid, sizeof(trid.trsvcid), "%d", MOE_TARGET_PORT);
	snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", MOE_NQN);
	trid.adrfam = SPDK_NVMF_ADRFAM_IPV4;
	if (spdk_nvme_probe(&trid, NULL, probe_cb, attach_cb, NULL) || !g_ns || !g_ctrlr) {
		goto out;
	}
	qpair = spdk_nvme_ctrlr_alloc_io_qpair(g_ctrlr, NULL, 0);
	if (!qpair) {
		goto out;
	}
	srand(seed);
	for (unsigned sample = 0; sample < count; sample++) {
		for (int i = 0; i < MOE_D_MODEL; i++) {
			input[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
		}
		memset(output, 0xff, MOE_INPUT_BYTES);
		struct spdk_nvme_cmd cmd = {.opc = MOE_VENDOR_OPCODE,
			.nsid = spdk_nvme_ns_get_id(g_ns)};
		struct moe_completion completion = {};
		if (spdk_nvme_ctrlr_cmd_io_raw(g_ctrlr, qpair, &cmd, input, MOE_INPUT_BYTES,
					     io_complete, &completion) ||
		    wait_for_completion(qpair, &completion)) {
			goto out;
		}
		completion = (struct moe_completion){};
		if (spdk_nvme_ns_cmd_read(g_ns, qpair, output, 0, 1, io_complete, &completion, 0) ||
		    wait_for_completion(qpair, &completion) ||
		    moe_compute_expected(input, weights.W_router, expected)) {
			goto out;
		}
		int first_diff;
		float error = max_abs_error(output, expected, MOE_D_MODEL, &first_diff);
		int worst = 0;
		for (int i = 0; i < MOE_D_MODEL; i++) {
			if (!isfinite(output[i]) || !isfinite(expected[i]) ||
			    fabsf(output[i] - expected[i]) > fabsf(output[worst] - expected[worst])) {
				worst = i;
			}
		}
		printf("seed=%u sample=%u max_absolute_error=%.9g worst_index=%d pass=%d\n",
		       seed, sample, error, worst, isfinite(error) && error < MOE_MAX_ABS_ERROR);
		if (!isfinite(error) || !(error < MOE_MAX_ABS_ERROR)) {
			goto out;
		}
	}
	rc = 0;
out:
	if (qpair) {
		spdk_nvme_ctrlr_free_io_qpair(qpair);
	}
	if (g_ctrlr) {
		spdk_nvme_detach_async(g_ctrlr, &detach);
		if (detach) {
			spdk_nvme_detach_poll(detach);
		}
	}
	spdk_dma_free(input);
	spdk_dma_free(output);
	free(expected);
	moe_free_weights(&weights);
	spdk_env_fini();
	return rc;
}
