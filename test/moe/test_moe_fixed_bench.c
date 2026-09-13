// SPDX-License-Identifier: BSD-3-Clause

// Reuse the official transport and clock helpers without changing its executable.
int moe_time_bench_main(int argc, char **argv);
#define main moe_time_bench_main
#include "../../examples/moe/moe_bench.c"
#undef main

int
main(int argc, char **argv)
{
	struct spdk_env_opts env_opts;
	struct spdk_nvme_qpair *qpair = NULL;
	struct spdk_nvme_detach_ctx *detach = NULL;
	float *input = NULL, *output = NULL;
	long values[4];
	char core_list[32];
	FILE *file = NULL;
	int rc = 1;

	if (argc != 6) {
		fprintf(stderr, "usage: %s count seed cpu repeat output.jsonl\n", argv[0]);
		return 1;
	}
	for (int i = 0; i < 4; i++) {
		char *end;
		errno = 0;
		values[i] = strtol(argv[i + 1], &end, 10);
		if (errno || end == argv[i + 1] || *end || values[i] < 0 || values[i] > INT_MAX) {
			return 1;
		}
	}
	if (!values[0] || values[2] >= CPU_SETSIZE || values[3] > 1 || bind_cpu(values[2])) {
		return 1;
	}
	file = fopen(argv[5], "w");
	if (!file) {
		return 1;
	}
	env_opts.opts_size = sizeof(env_opts);
	spdk_env_opts_init(&env_opts);
	snprintf(core_list, sizeof(core_list), "[%ld]", values[2]);
	env_opts.core_mask = core_list;
	env_opts.name = "moe_fixed_bench";
	env_opts.no_huge = true;
	env_opts.no_pci = true;
	env_opts.mem_size = SPDK_BENCH_MEM_MB;
	if (spdk_env_init(&env_opts) < 0) {
		fclose(file);
		return 1;
	}
	if (sched_getcpu() != values[2]) {
		fprintf(stderr, "initiator CPU differs from requested CPU\n");
		goto out;
	}
	printf("initiator_cpu=%d count=%ld seed=%ld repeat=%ld\n",
	       sched_getcpu(), values[0], values[1], values[3]);
	input = spdk_zmalloc(MOE_INPUT_BYTES, SPDK_DMA_ALIGN, NULL,
			     SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	output = spdk_zmalloc(MOE_INPUT_BYTES, SPDK_DMA_ALIGN, NULL,
			      SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	if (!input || !output || connect_target(&qpair)) {
		goto out;
	}
	srand(values[1]);
	for (long i = 0; i < values[0]; i++) {
		uint64_t hash = UINT64_C(14695981039346656037);
		if (!values[3] || i == 0) {
			fill_input_random(input);
		}
		for (size_t j = 0; j < MOE_INPUT_BYTES; j++) {
			hash = (hash ^ ((unsigned char *)input)[j]) * UINT64_C(1099511628211);
		}
		uint64_t start = now_ns();
		if (submit_moe_request(qpair, input, output)) {
			goto out;
		}
		uint64_t elapsed = now_ns() - start;
		for (int j = 0; j < MOE_D_MODEL; j++) {
			if (!isfinite(output[j])) {
				goto out;
			}
		}
		fprintf(file, "{\"request\":%ld,\"input_hash\":\"%016" PRIx64
			"\",\"latency_ns\":%" PRIu64 "}\n", i, hash, elapsed);
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
	spdk_free(input);
	spdk_free(output);
	spdk_env_fini();
	if (fclose(file)) {
		rc = 1;
	}
	return rc;
}
