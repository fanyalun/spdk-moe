/* SPDX-License-Identifier: BSD-3-Clause */

#include "../../module/bdev/moe/moe_codec.c"

static void
check(const float *data, unsigned count, unsigned expected)
{
	struct moe_codec_block block;
	unsigned char *encoded = malloc(count * 4 + 16);
	float *restored = malloc(count * 4);
	assert(encoded && restored);
	assert(!moe_codec_encode(data, count, encoded, count * 4 + 16, &block));
	assert(block.bits == expected);
	unsigned char *tight = malloc(block.bytes + 16);
	assert(tight);
	memcpy(tight, encoded, block.bytes + 16);
	assert(!moe_codec_decode(tight, block.bytes + 16, restored, &block));
	assert(!memcmp(data, restored, count * 4));
	assert(moe_codec_decode(tight, block.bytes + 15, restored, &block) == -EINVAL);
	block.checksum ^= UINT32_MAX;
	assert(!moe_codec_decode(tight, block.bytes + 16, restored, &block));
	assert(!memcmp(data, restored, count * 4));
	struct moe_codec_block invalid = block;
	invalid.bytes++;
	assert(moe_codec_decode(encoded, count * 4 + 16, restored, &invalid) == -EINVAL);
	invalid = block;
	invalid.bits = 33;
	assert(moe_codec_decode(tight, block.bytes + 16, restored, &invalid) == -EINVAL);
	invalid = block;
	invalid.scale = 31;
	assert(moe_codec_decode(tight, block.bytes + 16, restored, &invalid) == -EINVAL);
	free(tight);
	free(encoded);
	free(restored);
}

int
main(void)
{
	float values[129];
	for (unsigned count = 1; count <= (sizeof(values) / sizeof(values[0])); count++) {
		memset(values, 0, sizeof(values));
		check(values, count, 0);
		for (unsigned bits = 1; bits <= 31; bits++) {
			uint32_t high = (UINT32_C(1) << bits) - (bits > 24 ? 1u << (bits - 24) : 1);
			for (unsigned i = 0; i < count; i++) {
				values[i] = (float)(i % 4 == 0 ? 0 : i % 4 == 1 ? high : i % 4 == 2 ? 1 : high / 2);
			}
			if (count >= 3) {
				unsigned expected = bits;
				check(values, count, expected);
			}
		}
	}
	uint32_t special[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234,
			      0x7fa01234, 1, 0x80000001};
	memcpy(values, special, sizeof(special));
	check(values, (sizeof(special) / sizeof(special[0])), 32);
	uint32_t state = 123;
	for (unsigned round = 0; round < 100; round++) {
		for (unsigned i = 0; i < (sizeof(values) / sizeof(values[0])); i++) {
			state = state * 1664525u + 1013904223u;
			memcpy(&values[i], &state, 4);
		}
		check(values, (sizeof(values) / sizeof(values[0])), 32);
	}
	puts("PASS: widths, tight guards, tails, raw fallback, NaN/Inf/signed-zero bits, "
	     "ignored checksum, metadata guards");
	return 0;
}
