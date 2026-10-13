/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_CODEC_H
#define MOE_CODEC_H

#include "spdk/stdinc.h"

struct moe_codec_block {
	uint32_t words, bytes;
	int32_t scale, base;
	uint32_t bits, checksum;
};

_Static_assert(sizeof(struct moe_codec_block) == 24, "stable block index size");

int moe_codec_encode(const float *input, uint32_t words, void *output, size_t capacity,
		     struct moe_codec_block *block);
int moe_codec_decode(const void *input, size_t bytes, float *output,
		     const struct moe_codec_block *block);

#endif
