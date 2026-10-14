/* SPDX-License-Identifier: BSD-3-Clause */

#include "../../module/bdev/moe/moe_codec.c"

void
research_pack(const float *input, float *output, unsigned rows, unsigned cols,
	      unsigned first, unsigned words)
{
	for (unsigned i = 0; i < words;) {
		unsigned position = first + i;
		unsigned tile = position / (rows * 64);
		unsigned inner = position % (rows * 64);
		unsigned count = 64 - inner % 64;
		count = count < words - i ? count : words - i;
		memcpy(output + i, input + (inner / 64) * cols + tile * 64 + inner % 64,
		       count * sizeof(float));
		i += count;
	}
}

void
research_integers(const float *input, uint32_t *output, unsigned words,
		  const struct moe_codec_block *b)
{
	double factor = ldexp(1.0, b->scale);
	for (unsigned i = 0; i < words; i++) {
		output[i] = (uint32_t)((int64_t)(int32_t)((double)input[i] * factor) - b->base);
	}
}

void
research_restore_integers(const uint32_t *input, float *output, unsigned words,
			 const struct moe_codec_block *b)
{
	float factor = ldexpf(1.0f, -b->scale);
	for (unsigned i = 0; i < words; i++) {
		output[i] = (float)(int32_t)((int64_t)b->base + input[i]) * factor;
	}
}

void
research_shuffle(const void *input, void *output, unsigned words, int bit, int inverse)
{
	const unsigned char *source = input;
	unsigned char *dest = output;
	if (!bit) {
		for (unsigned i = 0; i < words; i++) {
			for (unsigned j = 0; j < 4; j++) {
				if (inverse) {
					dest[4 * i + j] = source[j * words + i];
				} else {
					dest[j * words + i] = source[4 * i + j];
				}
			}
		}
		return;
	}
	unsigned stride = (words + 7) / 8;
	if (inverse) {
		uint32_t *values = output;
		for (unsigned i = 0; i < words; i++) {
			uint32_t value = 0;
			for (unsigned j = 0; j < 32; j++) {
				value |= (uint32_t)((source[j * stride + i / 8] >> (i % 8)) & 1) << j;
			}
			values[i] = value;
		}
	} else {
		const uint32_t *values = input;
		memset(dest, 0, 32 * stride);
		for (unsigned i = 0; i < words; i++) {
			for (unsigned j = 0; j < 32; j++) {
				dest[j * stride + i / 8] |= ((values[i] >> j) & 1) << (i % 8);
			}
		}
	}
}

int64_t
research_split_encode(const float *input, unsigned words, void *output, size_t capacity,
		      struct moe_codec_block *groups, float *positive, float *negative)
{
	unsigned count[2] = {0, 0};
	unsigned mask_bytes = (words + 7) / 8;
	unsigned char *mask = output;
	float *temp[2] = {positive, negative};
	if (capacity < mask_bytes) {
		return -EINVAL;
	}
	memset(mask, 0, mask_bytes);
	memset(groups, 0, sizeof(*groups) * 2);
	for (unsigned i = 0; i < words; i++) {
		unsigned sign = float_bits(input[i]) >> 31;
		mask[i / 8] |= sign << (i % 8);
		memcpy(temp[sign] + count[sign]++, input + i, sizeof(float));
	}
	size_t used = mask_bytes;
	for (unsigned j = 0; j < 2; j++) {
		if (count[j]) {
			int rc = moe_codec_encode(temp[j], count[j], (char *)output + used,
					  capacity - used, groups + j);
			if (rc) {
				return rc;
			}
		}
		if (capacity - used < groups[j].bytes + 16) {
			return -EINVAL;
		}
		memset((char *)output + used + groups[j].bytes, 0, 16);
		used += groups[j].bytes + 16;
	}
	return used;
}

static struct {
	int positive[8], negative[8], mask[8];
	unsigned negatives;
} split_routes[256];

__attribute__((constructor)) static void
initialize_split_routes(void)
{
	for (unsigned mask = 0; mask < 256; mask++) {
		unsigned positive = 0, negative = 0;
		for (unsigned i = 0; i < 8; i++) {
			unsigned sign = (mask >> i) & 1;
			split_routes[mask].positive[i] = positive;
			split_routes[mask].negative[i] = negative;
			split_routes[mask].mask[i] = -(int)sign;
			positive += !sign;
			negative += sign;
		}
		split_routes[mask].negatives = negative;
	}
}

__attribute__((target("avx2"))) static void
merge_split_avx2(const unsigned char *mask, float *output, unsigned words,
		const float *positive, const float *negative)
{
	unsigned count[2] = {0, 0}, i = 0;
	for (; i + 8 <= words; i += 8) {
		unsigned bits = mask[i / 8];
		__m256i p = _mm256_loadu_si256((const __m256i *)(positive + count[0]));
		__m256i n = _mm256_loadu_si256((const __m256i *)(negative + count[1]));
		p = _mm256_permutevar8x32_epi32(p,
				_mm256_loadu_si256((const __m256i *)split_routes[bits].positive));
		n = _mm256_permutevar8x32_epi32(n,
				_mm256_loadu_si256((const __m256i *)split_routes[bits].negative));
		__m256i selector = _mm256_loadu_si256((const __m256i *)split_routes[bits].mask);
		_mm256_storeu_si256((__m256i *)(output + i), _mm256_blendv_epi8(p, n, selector));
		count[1] += split_routes[bits].negatives;
		count[0] += 8 - split_routes[bits].negatives;
	}
	for (; i < words; i++) {
		unsigned sign = (mask[i / 8] >> (i % 8)) & 1;
		const float *source = sign ? negative + count[1]++ : positive + count[0]++;
		memcpy(output + i, source, sizeof(float));
	}
}

static int
split_decode(const void *input, size_t bytes, float *output, unsigned words,
	     const struct moe_codec_block *groups, float *positive, float *negative, bool fast)
{
	unsigned mask_bytes = (words + 7) / 8;
	const unsigned char *mask = input;
	float *temp[2] = {positive, negative};
	size_t expected = (size_t)mask_bytes + groups[0].bytes + groups[1].bytes + 32;
	if (bytes != expected || (uint64_t)groups[0].words + groups[1].words != words) {
		return -EINVAL;
	}
	size_t used = mask_bytes;
	for (unsigned j = 0; j < 2; j++) {
		if (groups[j].words && moe_codec_decode((char *)input + used, groups[j].bytes + 16,
						      temp[j], groups + j)) {
			return -EINVAL;
		}
		used += groups[j].bytes + 16;
	}
	if (fast && __builtin_cpu_supports("avx2")) {
		unsigned negatives = 0;
		for (unsigned i = 0; i < words / 8; i++) {
			negatives += split_routes[mask[i]].negatives;
		}
		if (words % 8) {
			negatives += split_routes[mask[words / 8] & ((1u << (words % 8)) - 1)].negatives;
		}
		if (negatives != groups[1].words) {
			return -EINVAL;
		}
		merge_split_avx2(mask, output, words, positive, negative);
		return 0;
	}
	unsigned count[2] = {0, 0};
	for (unsigned i = 0; i < words; i++) {
		unsigned sign = (mask[i / 8] >> (i % 8)) & 1;
		if (count[sign] >= groups[sign].words) {
			return -EINVAL;
		}
		memcpy(output + i, temp[sign] + count[sign]++, sizeof(float));
	}
	return 0;
}

int
research_split_decode(const void *input, size_t bytes, float *output, unsigned words,
		      const struct moe_codec_block *groups, float *positive, float *negative)
{
	return split_decode(input, bytes, output, words, groups, positive, negative, false);
}

int
research_split_decode_fast(const void *input, size_t bytes, float *output, unsigned words,
			   const struct moe_codec_block *groups, float *positive, float *negative)
{
	return split_decode(input, bytes, output, words, groups, positive, negative, true);
}

int64_t
research_sub_encode(const float *input, unsigned words, unsigned subwords,
		    void *output, size_t capacity)
{
	size_t used = 0;
	for (unsigned i = 0; i < words; i += subwords) {
		unsigned count = words - i < subwords ? words - i : subwords;
		struct moe_codec_block b;
		if (capacity - used < sizeof(b)) {
			return -EINVAL;
		}
		int rc = moe_codec_encode(input + i, count, (char *)output + used + sizeof(b),
					  capacity - used - sizeof(b), &b);
		if (rc) {
			return rc;
		}
		memcpy((char *)output + used, &b, sizeof(b));
		used += sizeof(b) + b.bytes + 16;
	}
	return used;
}

int
research_sub_decode(const void *input, size_t bytes, float *output,
		    unsigned words, unsigned subwords)
{
	size_t used = 0;
	for (unsigned i = 0; i < words; i += subwords) {
		struct moe_codec_block b;
		unsigned count = words - i < subwords ? words - i : subwords;
		if (bytes - used < sizeof(b)) {
			return -EINVAL;
		}
		memcpy(&b, (char *)input + used, sizeof(b));
		used += sizeof(b);
		if (b.words != count || bytes - used < (size_t)b.bytes + 16 ||
		    moe_codec_decode((char *)input + used, b.bytes + 16, output + i, &b)) {
			return -EINVAL;
		}
		used += b.bytes + 16;
	}
	return used == bytes ? 0 : -EINVAL;
}
