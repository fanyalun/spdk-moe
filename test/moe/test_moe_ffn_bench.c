// SPDX-License-Identifier: BSD-3-Clause
#define _POSIX_C_SOURCE 200809L
#include "moe_workspace.h"
#include "reference_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
seconds(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

int
main(void)
{
	const int d = 2048, h = 7168;
	size_t elements = (size_t)d * h;
	float *row[3] = {0}, *packed[3] = {0};
	float input[2048], expected[2048];
	struct moe_workspace ws = {0};
	int rc = 1;

	for (int m = 0; m < 3; m++) {
		row[m] = malloc(elements * sizeof(float));
		packed[m] = malloc(elements * sizeof(float));
		if (!row[m] || !packed[m]) {
			goto out;
		}
		for (size_t i = 0; i < elements; i++) {
			row[m][i] = ((int)((i * (m + 1) + m) % 127) - 63) * 0.001f;
		}
		matvec_pack(row[m], m == 2 ? h : d, m == 2 ? d : h, packed[m]);
	}
	for (int i = 0; i < d; i++) {
		input[i] = ((i % 31) - 15) * 0.001f;
	}
	reference_swiglu_ffn(input, row[0], row[1], row[2], d, h, expected);
	if (moe_workspace_init(&ws, d, h, 1, 1, MOE_KERNEL_AVX2)) {
		goto out;
	}
	memcpy(ws.input, input, sizeof(input));
	for (int round = 0; round < 3; round++) {
		for (int k = 0; k < 2; k++) {
			ws.kernel = (round % 2 ? 1 - k : k) ? MOE_KERNEL_AVX512 : MOE_KERNEL_AVX2;
			if (!moe_kernel_supported(ws.kernel)) {
				continue;
			}
			if (moe_expert(&ws, packed[0], packed[1], packed[2], 0, 1)) {
				goto out;
			}
			float error = 0;
			for (int i = 0; i < d; i++) {
				float delta = fabsf(expected[i] - ws.expert_outputs[i]);
				if (!isfinite(expected[i]) || !isfinite(ws.expert_outputs[i]) || !(delta < 1e-5f)) {
					goto out;
				}
				error = fmaxf(error, delta);
			}
			double start = seconds();
			for (int repeat = 0; repeat < 10; repeat++) {
				if (moe_expert(&ws, packed[0], packed[1], packed[2], 0, 1)) {
					goto out;
				}
			}
			printf("kernel=%s round=%d mean_ms=%.6f max_error=%g\n",
			       moe_kernel_name(ws.kernel), round, (seconds() - start) * 100, error);
		}
	}
	rc = 0;
out:
	moe_workspace_destroy(&ws);
	for (int m = 0; m < 3; m++) {
		free(row[m]);
		free(packed[m]);
	}
	return rc;
}
