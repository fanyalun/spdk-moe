/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/nvme_spec.h"
#include <math.h>

#ifdef MOE_FULL_TEST
#define D 2048
#define H 7168
#define N 256
#else
#define D 128
#define H 256
#define N 9
#endif
#define K 8

void reference_swiglu_moe(const float *, const float *, const float *[], const float *[],
			 const float *[], int, int, int, int, float *);
void reference_matvec_mul(const float *, const float *, int, int, float *);
void reference_topk_select(const float *, int, int, int *, float *);
void reference_softmax(float *, int);
void reference_swiglu_ffn(const float *, const float *, const float *, const float *, int, int, float *);

static struct spdk_bdev_desc *g_desc;
static struct spdk_io_channel *g_channel;
static float *g_input, *g_output;
static float g_expected[D];
static float *g_router;
static const char *g_directory;
static int g_round;
static bool g_removed;
static bool g_failure_expected;
static struct spdk_poller *g_remove_poller;
static struct spdk_poller *g_heartbeat_poller;
static unsigned g_heartbeats;

static void
finish(int status)
{
	spdk_poller_unregister(&g_heartbeat_poller);
	if (g_channel) {
		spdk_put_io_channel(g_channel);
		g_channel = NULL;
	}
	if (g_desc) {
		spdk_bdev_close(g_desc);
		g_desc = NULL;
	}
	spdk_app_stop(status);
}

static void
removed(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *arg)
{
	g_removed = true;
}

static void send_request(void);
static float *load(const char *dir, const char *name, int expert, size_t count);

static int
compute_expected(void)
{
	float logits[N], weights[K], expert[D];
	int indices[K];
	reference_matvec_mul(g_input, g_router, D, N, logits);
	reference_topk_select(logits, N, K, indices, weights);
	reference_softmax(weights, K);
	memset(g_expected, 0, sizeof(g_expected));
	for (int k = 0; k < K; k++) {
		float *gate = load(g_directory, "W_gate", indices[k], D * H);
		float *up = load(g_directory, "W_up", indices[k], D * H);
		float *down = load(g_directory, "W_down", indices[k], D * H);
		if (!gate || !up || !down) {
			free(gate);
			free(up);
			free(down);
			return -1;
		}
		reference_swiglu_ffn(g_input, gate, up, down, D, H, expert);
		free(gate);
		free(up);
		free(down);
		for (int j = 0; j < D; j++) {
			g_expected[j] = fmaf(weights[k], expert[j], g_expected[j]);
		}
	}
	return 0;
}

static void
read_done(struct spdk_bdev_io *io, bool success, void *arg)
{
	spdk_bdev_free_io(io);
	if (g_round >= 32 || g_failure_expected) {
		printf("invalid_output_read_failed=%d\n", !success);
		finish(success ? 1 : 0);
		return;
	}
	float max_error = 0.0f;
	for (int i = 0; i < D; i++) {
		float error = fabsf(g_expected[i] - g_output[i]);
		if (!isfinite(g_expected[i]) || !isfinite(g_output[i]) || !(error < 1e-5f)) {
			fprintf(stderr, "round=%d first_error=%d reference=%.9g actual=%.9g\n",
				g_round, i, g_expected[i], g_output[i]);
			finish(1);
			return;
		}
		max_error = fmaxf(max_error, error);
	}
	printf("round=%d success=%d max_error=%.9g\n", g_round, success, max_error);
	if (!success || g_removed) {
		finish(1);
		return;
	}
	if (getenv("MOE_TEST_SLOW")) {
		printf("reactor_heartbeats_during_slow_io=%u\n", g_heartbeats);
		finish(g_heartbeats >= 10 ? 0 : 1);
		return;
	}
	g_round++;
	send_request();
}

static void
vendor_done(struct spdk_bdev_io *io, bool success, void *arg)
{
	spdk_bdev_free_io(io);
	if (success != (g_round < 32 && !g_failure_expected)) {
		fprintf(stderr, "unexpected vendor status round=%d success=%d\n", g_round, success);
		finish(1);
		return;
	}
	int rc = spdk_bdev_read(g_desc, g_channel, g_output, 0, D * sizeof(float), read_done, NULL);
	if (rc) {
		finish(g_failure_expected && g_removed ? 0 : 1);
	}
}

static int
heartbeat(void *arg)
{
	g_heartbeats++;
	return SPDK_POLLER_BUSY;
}

static int
remove_base(void *arg)
{
	spdk_poller_unregister(&g_remove_poller);
	struct spdk_bdev *base = spdk_bdev_get_by_name("weight_aio");
	if (base) {
		spdk_bdev_unregister(base, NULL, NULL);
	}
	return SPDK_POLLER_BUSY;
}

static void
send_request(void)
{
	if (g_round == 1 && getenv("MOE_TEST_TRUNCATE")) {
		if (truncate(getenv("MOE_TEST_TRUNCATE"), 512)) {
			finish(1);
			return;
		}
		g_failure_expected = true;
	}
	if (g_round == 1 && getenv("MOE_TEST_REMOVE")) {
		g_failure_expected = true;
		g_remove_poller = SPDK_POLLER_REGISTER(remove_base, NULL, 100);
	}
	uint32_t state = 42 + ((g_round == 1 && getenv("MOE_TEST_REPEAT_INPUT")) ? 0 : g_round);
	for (int i = 0; i < D; i++) {
		state = state * 1664525U + 1013904223U;
		g_input[i] = (float)(state >> 8) / 16777216.0f * 0.2f - 0.1f;
	}
	if (g_round == 32) {
		g_input[0] = NAN;
	} else {
		if (compute_expected()) {
			finish(1);
			return;
		}
	}
	struct spdk_nvme_cmd cmd = {.opc = 0xc1};
	if (getenv("MOE_TEST_SLOW") && !g_heartbeat_poller) {
		g_heartbeat_poller = SPDK_POLLER_REGISTER(heartbeat, NULL, 1000);
	}
	int rc = spdk_bdev_nvme_io_passthru(g_desc, g_channel, &cmd, g_input,
					   D * sizeof(float), vendor_done, NULL);
	if (rc) {
		finish(1);
	}
}

static float *
load(const char *dir, const char *name, int expert, size_t count)
{
	char path[4096];
	int n;
	if (expert < 0) {
		n = snprintf(path, sizeof(path), "%s/W_router_%dx%d.bin", dir, D, N);
	} else {
		n = snprintf(path, sizeof(path), "%s/%s_%d_%dx%d.bin", dir, name, expert, D, H);
	}
	if (n < 0 || (size_t)n >= sizeof(path)) {
		return NULL;
	}
	FILE *f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	float *data = malloc(count * sizeof(float));
	if (!data || fread(data, sizeof(float), count, f) != count) {
		free(data);
		data = NULL;
	}
	fclose(f);
	return data;
}

static void
started(void *arg)
{
	const char *dir = getenv("MOE_TEST_WEIGHTS");
	if (!dir) {
		finish(1);
		return;
	}
	g_router = load(dir, NULL, -1, D * N);
	g_directory = dir;
	g_input = spdk_dma_zmalloc(D * sizeof(float), 4096, NULL);
	g_output = spdk_dma_zmalloc(D * sizeof(float), 4096, NULL);
	if (!g_router || !g_input || !g_output ||
	    spdk_bdev_open_ext("moe_test", true, removed, NULL, &g_desc)) {
		finish(1);
		return;
	}
	g_channel = spdk_bdev_get_io_channel(g_desc);
	if (!g_channel) {
		finish(1);
		return;
	}
	send_request();
}

int
main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	struct spdk_app_opts opts;
	spdk_app_opts_init(&opts, sizeof(opts));
	opts.name = "moe_pipeline_test";
	opts.no_huge = true;
	opts.no_pci = true;
	opts.mem_size = 512;
#ifdef MOE_FULL_TEST
	opts.mem_size = 1536;
#endif
	int rc = spdk_app_parse_args(argc, argv, &opts, NULL, NULL, NULL, NULL);
	if (rc != SPDK_APP_PARSE_ARGS_SUCCESS) {
		return 1;
	}
	rc = spdk_app_start(&opts, started, NULL);
	spdk_dma_free(g_input);
	spdk_dma_free(g_output);
	free(g_router);
	spdk_app_fini();
	return rc;
}
