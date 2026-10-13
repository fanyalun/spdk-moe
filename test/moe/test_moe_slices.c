/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "moe_ffn/moe_workspace.h"

int
main(void)
{
	const unsigned d = 128, h = 320;
	float *raw = malloc(d * h * sizeof(float));
	float *packed[3];
	uint32_t random = 42;
	for (unsigned stage = 0; stage < 3; stage++) {
		packed[stage] = malloc(d * h * sizeof(float));
		assert(packed[stage]);
		for (unsigned i = 0; i < d * h; i++) {
			random = random * 1664525u + 1013904223u;
			raw[i] = ((int)(random % 2001) - 1000) / 1048576.0f;
		}
		assert(!matvec_pack(raw, stage == 2 ? h : d, stage == 2 ? d : h, packed[stage]));
	}
	for (int kernel = MOE_KERNEL_SCALAR; kernel <= MOE_KERNEL_AVX512; kernel++) {
		if (!moe_kernel_supported(kernel)) {
			continue;
		}
		struct moe_workspace whole, tiled;
		assert(!moe_workspace_init(&whole, d, h, 1, 1, kernel));
		assert(!moe_workspace_init(&tiled, d, h, 1, 1, kernel));
		for (unsigned i = 0; i < d; i++) {
			whole.input[i] = tiled.input[i] = ((int)(i % 11) - 5) / 16.0f;
		}
		assert(!moe_expert(&whole, packed[0], packed[1], packed[2], 0, 1));
		for (unsigned width = 64; width <= 512; width *= 2) {
			for (unsigned stage = 0; stage < 3; stage++) {
				unsigned total = stage == 2 ? d : h;
				for (unsigned first = 0; first < total; first += width) {
					unsigned count = width < total - first ? width : total - first;
					assert(!moe_expert_slice(&tiled, stage, packed[stage], tiled.gate,
							 tiled.up, tiled.expert_outputs, first, count));
				}
			}
			assert(!memcmp(whole.gate, tiled.gate, h * sizeof(float)));
			assert(!memcmp(whole.up, tiled.up, h * sizeof(float)));
			assert(!memcmp(whole.expert_outputs, tiled.expert_outputs, d * sizeof(float)));
		}
		assert(moe_expert_slice(&tiled, 0, packed[0], tiled.gate, tiled.up,
					tiled.expert_outputs, 1, 64) == -EINVAL);
		moe_workspace_destroy(&whole);
		moe_workspace_destroy(&tiled);
	}
	for (unsigned i = 0; i < 3; i++) {
		free(packed[i]);
	}
	free(raw);
	puts("PASS: all supported kernels and 64/128/256/512 slices are bitwise identical");
	return 0;
}
