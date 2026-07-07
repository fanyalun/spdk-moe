/* SPDX-License-Identifier: BSD-3-Clause
 *
 * moe_bench.c — MoE FFN SPDK 远程推理性能评测 initiator。
 *
 * 每个样本在计时前生成随机 input，计时区间覆盖一次 vendor opcode 远程
 * 计算和一次 READ 取回 output 的完整 NVMe-oF 往返路径。统计策略参考 fio：
 * CLOCK_MONOTONIC、warmup/runtime 时间控制、对数分桶直方图。
 */

#include "spdk/stdinc.h"

#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/nvme_spec.h"
#include "spdk/string.h"

#include "moe_ffn/moe_config.h"

#include <sched.h>

#define FIO_IO_U_PLAT_BITS 6
#define FIO_IO_U_PLAT_VAL (1 << FIO_IO_U_PLAT_BITS)
#define FIO_IO_U_PLAT_GROUP_NR 29
#define FIO_IO_U_PLAT_NR (FIO_IO_U_PLAT_GROUP_NR * FIO_IO_U_PLAT_VAL)

#define DEFAULT_WARMUP_SEC 3
#define DEFAULT_RUNTIME_SEC 10
#define DEFAULT_JSON_OUTPUT "./moe_bench_result.json"
#define SPDK_BENCH_MEM_MB 512
#define SPDK_DMA_ALIGN 0x1000

struct bench_opts {
	int seed;
	int cpu;
	int warmup_sec;
	int runtime_sec;
	const char *json_output;
};

struct latency_stats {
	uint64_t buckets[FIO_IO_U_PLAT_NR];
	uint64_t samples;
	uint64_t min_ns;
	uint64_t max_ns;
	long double mean;
	long double m2;
};

struct moe_completion {
	bool done;
	bool failed;
};

struct percentile_def {
	double pct;
	const char *json_key;
};

static const struct percentile_def g_percentiles[] = {
	{ 1.0, "p1" }, { 5.0, "p5" }, { 10.0, "p10" }, { 20.0, "p20" },
	{ 30.0, "p30" }, { 40.0, "p40" }, { 50.0, "p50" }, { 60.0, "p60" },
	{ 70.0, "p70" }, { 80.0, "p80" }, { 90.0, "p90" }, { 95.0, "p95" },
	{ 99.0, "p99" }, { 99.5, "p99_5" }, { 99.9, "p99_9" },
	{ 99.95, "p99_95" }, { 99.99, "p99_99" },
};

static struct spdk_nvme_ctrlr *g_ctrlr;
static struct spdk_nvme_ns *g_ns;

static uint64_t
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void
fill_input_random(float *input)
{
	int i;

	for (i = 0; i < MOE_D_MODEL; i++) {
		input[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
	}
}

static unsigned int
plat_val_to_idx(uint64_t val)
{
	unsigned int msb, error_bits, base, offset, idx;

	if (val == 0) {
		msb = 0;
	} else {
		msb = (sizeof(val) * 8) - __builtin_clzll(val) - 1;
	}

	/* 小值不舍入，直接精确映射到桶。 */
	if (msb <= FIO_IO_U_PLAT_BITS) {
		return (unsigned int)val;
	}

	/* 丢弃低位 error_bits，保留 6 位作为组内索引。 */
	error_bits = msb - FIO_IO_U_PLAT_BITS;
	base = (error_bits + 1) << FIO_IO_U_PLAT_BITS;
	offset = (FIO_IO_U_PLAT_VAL - 1) & (val >> error_bits);
	idx = (base + offset) < (FIO_IO_U_PLAT_NR - 1) ?
	      (base + offset) : (FIO_IO_U_PLAT_NR - 1);

	return idx;
}

static uint64_t
plat_idx_to_val(unsigned int idx)
{
	unsigned int error_bits;
	uint64_t k, base;

	assert(idx < FIO_IO_U_PLAT_NR);

	if (idx < (FIO_IO_U_PLAT_VAL << 1)) {
		return idx;
	}

	error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1;
	base = 1ULL << (error_bits + FIO_IO_U_PLAT_BITS);
	k = idx % FIO_IO_U_PLAT_VAL;

	return base + (uint64_t)((k + 0.5) * (1ULL << error_bits));
}

static void
stats_reset(struct latency_stats *stats)
{
	memset(stats, 0, sizeof(*stats));
	stats->min_ns = UINT64_MAX;
}

static void
stats_record(struct latency_stats *stats, uint64_t latency_ns)
{
	long double delta, delta2;
	unsigned int idx;

	idx = plat_val_to_idx(latency_ns);
	stats->buckets[idx]++;
	stats->samples++;

	if (latency_ns < stats->min_ns) {
		stats->min_ns = latency_ns;
	}
	if (latency_ns > stats->max_ns) {
		stats->max_ns = latency_ns;
	}

	/* Welford: mean_n = mean_(n-1) + delta / n, M2 += delta * delta2。 */
	delta = (long double)latency_ns - stats->mean;
	stats->mean += delta / (long double)stats->samples;
	delta2 = (long double)latency_ns - stats->mean;
	stats->m2 += delta * delta2;
}

static long double
stats_stddev(const struct latency_stats *stats)
{
	if (stats->samples == 0) {
		return 0.0L;
	}

	return sqrtl(stats->m2 / (long double)stats->samples);
}

static void
calc_percentiles(const struct latency_stats *stats, uint64_t *out)
{
	uint64_t sum = 0;
	size_t pidx = 0;
	unsigned int i;

	for (i = 0; i < FIO_IO_U_PLAT_NR && pidx < SPDK_COUNTOF(g_percentiles); i++) {
		sum += stats->buckets[i];
		while (pidx < SPDK_COUNTOF(g_percentiles) &&
		       (long double)sum >= (long double)g_percentiles[pidx].pct / 100.0L *
		       (long double)stats->samples) {
			out[pidx] = plat_idx_to_val(i);
			pidx++;
		}
	}

	while (pidx < SPDK_COUNTOF(g_percentiles)) {
		out[pidx++] = stats->max_ns;
	}
}

static void
select_unit(const struct latency_stats *stats, const char **unit, double *scale)
{
	long double stddev = stats_stddev(stats);

	*unit = "ns";
	*scale = 1.0;

	if (stats->min_ns > 2000000ULL && stats->max_ns > 99999999ULL && stddev > 1000000.0L) {
		*unit = "ms";
		*scale = 1000000.0;
	} else if (stats->min_ns > 2000ULL && stats->max_ns > 99999ULL && stddev > 1000.0L) {
		*unit = "us";
		*scale = 1000.0;
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
	completion->failed = spdk_nvme_cpl_is_error(cpl);
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

static int
submit_moe_request(struct spdk_nvme_qpair *qpair, float *input, float *output)
{
	struct spdk_nvme_cmd cmd = {};
	struct moe_completion completion;
	int rc;

	cmd.opc = MOE_VENDOR_OPCODE;
	cmd.nsid = spdk_nvme_ns_get_id(g_ns);
	completion = (struct moe_completion){};
	rc = spdk_nvme_ctrlr_cmd_io_raw(g_ctrlr, qpair, &cmd, input, MOE_INPUT_BYTES,
					 io_complete, &completion);
	if (rc != 0 || wait_for_completion(qpair, &completion) != 0) {
		return -EIO;
	}

	completion = (struct moe_completion){};
	rc = spdk_nvme_ns_cmd_read(g_ns, qpair, output, 0, 1, io_complete, &completion, 0);
	if (rc != 0 || wait_for_completion(qpair, &completion) != 0) {
		return -EIO;
	}

	return 0;
}

static int
bind_cpu(int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	return sched_setaffinity(0, sizeof(set), &set);
}

static void
print_help(const char *prog)
{
	printf("用法: %s [选项]\n\n", prog);
	printf("选项:\n");
	printf("  --seed=<int>          input 随机种子（默认: %d）\n", MOE_SEED_DEFAULT);
	printf("  --cpu=<int>           绑定 CPU 核心（默认: 0）\n");
	printf("  --warmup=<int>        预热秒数（默认: %d）\n", DEFAULT_WARMUP_SEC);
	printf("  --runtime=<int>       正式测量秒数（默认: %d）\n", DEFAULT_RUNTIME_SEC);
	printf("  --json-output=<path>  JSON 输出路径（默认: %s）\n", DEFAULT_JSON_OUTPUT);
	printf("  --help                打印帮助\n");
}

static int
parse_int_arg(const char *arg, const char *prefix, int *value)
{
	size_t len = strlen(prefix);
	char *end = NULL;
	long v;

	if (strncmp(arg, prefix, len) != 0) {
		return 0;
	}
	if (arg[len] == '\0') {
		return -1;
	}

	v = strtol(arg + len, &end, 10);
	if (*end != '\0' || v < 0 || v > INT_MAX) {
		return -1;
	}

	*value = (int)v;
	return 1;
}

static int
parse_args(int argc, char **argv, struct bench_opts *opts)
{
	int i, r;

	*opts = (struct bench_opts){
		.seed = MOE_SEED_DEFAULT,
		.cpu = 0,
		.warmup_sec = DEFAULT_WARMUP_SEC,
		.runtime_sec = DEFAULT_RUNTIME_SEC,
		.json_output = DEFAULT_JSON_OUTPUT,
	};

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0) {
			print_help(argv[0]);
			exit(0);
		}
		r = parse_int_arg(argv[i], "--seed=", &opts->seed);
		if (r != 0) {
			if (r < 0) return -1;
			continue;
		}
		r = parse_int_arg(argv[i], "--cpu=", &opts->cpu);
		if (r != 0) {
			if (r < 0) return -1;
			continue;
		}
		r = parse_int_arg(argv[i], "--warmup=", &opts->warmup_sec);
		if (r != 0) {
			if (r < 0) return -1;
			continue;
		}
		r = parse_int_arg(argv[i], "--runtime=", &opts->runtime_sec);
		if (r != 0) {
			if (r < 0) return -1;
			continue;
		}
		if (strncmp(argv[i], "--json-output=", strlen("--json-output=")) == 0) {
			opts->json_output = argv[i] + strlen("--json-output=");
			if (opts->json_output[0] == '\0') {
				return -1;
			}
			continue;
		}
		fprintf(stderr, "无法识别的参数: %s\n", argv[i]);
		return -1;
	}

	return 0;
}

static int
connect_target(struct spdk_nvme_qpair **qpair)
{
	struct spdk_nvme_transport_id trid = {};
	int rc;

	spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_TCP);
	snprintf(trid.traddr, sizeof(trid.traddr), "%s", MOE_TARGET_ADDR);
	snprintf(trid.trsvcid, sizeof(trid.trsvcid), "%d", MOE_TARGET_PORT);
	snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", MOE_NQN);
	trid.adrfam = SPDK_NVMF_ADRFAM_IPV4;

	rc = spdk_nvme_probe(&trid, NULL, probe_cb, attach_cb, NULL);
	if (rc != 0 || g_ctrlr == NULL || g_ns == NULL) {
		fprintf(stderr, "NVMe-oF 连接失败: %s:%d %s\n",
			MOE_TARGET_ADDR, MOE_TARGET_PORT, MOE_NQN);
		return -EIO;
	}

	*qpair = spdk_nvme_ctrlr_alloc_io_qpair(g_ctrlr, NULL, 0);
	if (*qpair == NULL) {
		fprintf(stderr, "spdk_nvme_ctrlr_alloc_io_qpair failed\n");
		return -ENOMEM;
	}

	return 0;
}

static int
run_phase(struct spdk_nvme_qpair *qpair, float *input, float *output,
	  int seconds, struct latency_stats *stats, bool record, double *elapsed_sec)
{
	uint64_t start_ns, deadline_ns, end_ns;
	int rc;

	start_ns = now_ns();
	deadline_ns = start_ns + (uint64_t)seconds * 1000000000ULL;

	for (;;) {
		uint64_t t0, t1;

		fill_input_random(input);
		t0 = now_ns();
		rc = submit_moe_request(qpair, input, output);
		t1 = now_ns();
		if (rc != 0) {
			return rc;
		}
		if (record) {
			stats_record(stats, t1 - t0);
		}
		if (t1 >= deadline_ns) {
			break;
		}
	}

	end_ns = now_ns();
	*elapsed_sec = (double)(end_ns - start_ns) / 1000000000.0;
	return 0;
}

static int
write_json(const char *path, const struct bench_opts *opts,
	   const struct latency_stats *stats, const uint64_t *pct_values,
	   double runtime_sec)
{
	FILE *fp;
	long double stddev = stats_stddev(stats);
	long double cv = stats->mean > 0.0L ? stddev / stats->mean : 0.0L;
	double rate = runtime_sec > 0.0 ? (double)stats->samples / runtime_sec : 0.0;
	size_t i;

	fp = fopen(path, "w");
	if (fp == NULL) {
		return -errno;
	}

	fprintf(fp, "{\n");
	fprintf(fp, "  \"completed\": true,\n");
	fprintf(fp, "  \"parameters\": {\n");
	fprintf(fp, "    \"d_model\": %d,\n", MOE_D_MODEL);
	fprintf(fp, "    \"d_ff\": %d,\n", MOE_D_FF);
	fprintf(fp, "    \"num_experts\": %d,\n", MOE_NUM_EXPERTS);
	fprintf(fp, "    \"top_k\": %d,\n", MOE_TOP_K);
	fprintf(fp, "    \"seed\": %d,\n", opts->seed);
	fprintf(fp, "    \"cpu\": %d,\n", opts->cpu);
	fprintf(fp, "    \"warmup_sec\": %d,\n", opts->warmup_sec);
	fprintf(fp, "    \"runtime_sec\": %d\n", opts->runtime_sec);
	fprintf(fp, "  },\n");
	fprintf(fp, "  \"latency_ns\": {\n");
	fprintf(fp, "    \"min\": %" PRIu64 ",\n", stats->min_ns == UINT64_MAX ? 0 : stats->min_ns);
	fprintf(fp, "    \"max\": %" PRIu64 ",\n", stats->max_ns);
	fprintf(fp, "    \"avg\": %.3Lf,\n", stats->mean);
	fprintf(fp, "    \"stddev\": %.3Lf,\n", stddev);
	fprintf(fp, "    \"cv\": %.6Lf,\n", cv);
	fprintf(fp, "    \"percentiles\": {\n");
	for (i = 0; i < SPDK_COUNTOF(g_percentiles); i++) {
		fprintf(fp, "      \"%s\": %" PRIu64 "%s\n",
			g_percentiles[i].json_key, pct_values[i],
			i + 1 == SPDK_COUNTOF(g_percentiles) ? "" : ",");
	}
	fprintf(fp, "    }\n");
	fprintf(fp, "  },\n");
	fprintf(fp, "  \"throughput\": {\n");
	fprintf(fp, "    \"total_calls\": %" PRIu64 ",\n", stats->samples);
	fprintf(fp, "    \"runtime_sec\": %.6f,\n", runtime_sec);
	fprintf(fp, "    \"rate_calls_per_sec\": %.6f\n", rate);
	fprintf(fp, "  }\n");
	fprintf(fp, "}\n");
	fclose(fp);
	return 0;
}

static void
print_report(const struct bench_opts *opts, const struct latency_stats *stats,
	     const uint64_t *pct_values, double runtime_sec, const char *json_output,
	     bool json_ok)
{
	const char *unit;
	double scale;
	long double stddev = stats_stddev(stats);
	long double cv = stats->mean > 0.0L ? stddev / stats->mean : 0.0L;
	double rate = runtime_sec > 0.0 ? (double)stats->samples / runtime_sec : 0.0;
	size_t i;

	select_unit(stats, &unit, &scale);

	printf("================================================================\n");
	printf("  MoE FFN SPDK 性能评测报告\n");
	printf("  d_model = %d, d_ff = %d, num_experts = %d, top_k = %d\n",
	       MOE_D_MODEL, MOE_D_FF, MOE_NUM_EXPERTS, MOE_TOP_K);
	printf("  seed = %d, CPU: %d, warmup = %ds, runtime = %ds\n",
	       opts->seed, opts->cpu, opts->warmup_sec, opts->runtime_sec);
	printf("  target: %s:%d, opcode = 0x%02x\n", MOE_TARGET_ADDR, MOE_TARGET_PORT, MOE_VENDOR_OPCODE);
	printf("================================================================\n\n");

	printf("延迟统计 (Latency, vendor cmd + read):\n");
	printf("  min       = %.3f %s\n", (double)stats->min_ns / scale, unit);
	printf("  max       = %.3f %s\n", (double)stats->max_ns / scale, unit);
	printf("  avg       = %.3f %s\n", (double)stats->mean / scale, unit);
	printf("  stddev    = %.3f %s\n", (double)stddev / scale, unit);
	printf("  CV        = %.6Lf\n\n", cv);

	printf("百分位分布 (Percentiles):\n");
	for (i = 0; i < SPDK_COUNTOF(g_percentiles); i++) {
		printf("  %6.2f%% = %.3f %s\n", g_percentiles[i].pct,
		       (double)pct_values[i] / scale, unit);
	}

	printf("\n吞吐量 (Throughput):\n");
	printf("  total_calls = %" PRIu64 "\n", stats->samples);
	printf("  runtime     = %.3f s\n", runtime_sec);
	printf("  rate        = %.6f calls/sec\n", rate);

	if (json_ok) {
		printf("\nJSON 结果已写入: %s\n", json_output);
	}
}

int
main(int argc, char **argv)
{
	struct bench_opts opts;
	struct spdk_env_opts env_opts;
	struct spdk_nvme_qpair *qpair = NULL;
	struct spdk_nvme_detach_ctx *detach_ctx = NULL;
	struct latency_stats stats;
	uint64_t pct_values[SPDK_COUNTOF(g_percentiles)] = {};
	float *input = NULL, *output = NULL;
	double warmup_elapsed = 0.0, runtime_elapsed = 0.0;
	int rc = 0, json_rc;

	if (parse_args(argc, argv, &opts) != 0) {
		print_help(argv[0]);
		return 1;
	}

	if (bind_cpu(opts.cpu) != 0) {
		fprintf(stderr, "sched_setaffinity(cpu=%d) failed: %s\n", opts.cpu, strerror(errno));
		return 1;
	}

	env_opts.opts_size = sizeof(env_opts);
	spdk_env_opts_init(&env_opts);
	env_opts.name = "moe_bench";
	env_opts.no_huge = true;
	env_opts.mem_size = SPDK_BENCH_MEM_MB;
	if (spdk_env_init(&env_opts) < 0) {
		fprintf(stderr, "Unable to initialize SPDK env\n");
		return 1;
	}

	srand(opts.seed);
	input = spdk_zmalloc(MOE_INPUT_BYTES, SPDK_DMA_ALIGN, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	output = spdk_zmalloc(MOE_INPUT_BYTES, SPDK_DMA_ALIGN, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	assert(input != NULL && output != NULL);

	rc = connect_target(&qpair);
	if (rc != 0) {
		goto out;
	}

	stats_reset(&stats);
	rc = run_phase(qpair, input, output, opts.warmup_sec, &stats, false, &warmup_elapsed);
	if (rc != 0) {
		fprintf(stderr, "warmup phase failed\n");
		goto out;
	}

	stats_reset(&stats);
	rc = run_phase(qpair, input, output, opts.runtime_sec, &stats, true, &runtime_elapsed);
	if (rc != 0) {
		fprintf(stderr, "runtime phase failed\n");
		goto out;
	}

	calc_percentiles(&stats, pct_values);
	json_rc = write_json(opts.json_output, &opts, &stats, pct_values, runtime_elapsed);
	print_report(&opts, &stats, pct_values, runtime_elapsed, opts.json_output, json_rc == 0);
	if (json_rc != 0) {
		fprintf(stderr, "JSON 写入失败: %s: %s\n", opts.json_output, strerror(-json_rc));
		rc = 1;
	}

	(void)warmup_elapsed;

out:
	if (qpair != NULL) {
		spdk_nvme_ctrlr_free_io_qpair(qpair);
	}
	if (g_ctrlr != NULL) {
		spdk_nvme_detach_async(g_ctrlr, &detach_ctx);
		if (detach_ctx != NULL) {
			spdk_nvme_detach_poll(detach_ctx);
		}
	}
	spdk_free(input);
	spdk_free(output);
	spdk_env_fini();
	return rc == 0 ? 0 : 1;
}
