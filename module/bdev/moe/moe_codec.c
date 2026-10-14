/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include <immintrin.h>
#include "moe_codec.h"

__attribute__((target("sse4.2"))) static uint32_t
checksum_hw(const unsigned char *data, size_t bytes)
{
	uint64_t c[4] = {0, 1, 2, 3};
	while (bytes >= 32) {
		for (unsigned j = 0; j < 4; j++) {
			uint64_t word;
			memcpy(&word, data + 8 * j, 8);
			c[j] = _mm_crc32_u64(c[j], word);
		}
		data += 32;
		bytes -= 32;
	}
	uint32_t result = 0;
	for (unsigned j = 0; j < 4; j++) {
		result = _mm_crc32_u32(result, c[j]);
	}
	while (bytes--) {
		result = _mm_crc32_u8(result, *data++);
	}
	return result;
}

static uint32_t
crc_byte(uint32_t crc, unsigned char value)
{
	crc ^= value;
	for (unsigned i = 0; i < 8; i++) {
		crc = (crc >> 1) ^ ((0u - (crc & 1)) & 0x82f63b78u);
	}
	return crc;
}

static uint32_t
checksum(const unsigned char *data, size_t bytes)
{
	if (__builtin_cpu_supports("sse4.2")) {
		return checksum_hw(data, bytes);
	}
	uint32_t c[4] = {0, 1, 2, 3};
	while (bytes >= 32) {
		for (unsigned j = 0; j < 4; j++) {
			for (unsigned k = 0; k < 8; k++) {
				c[j] = crc_byte(c[j], data[8 * j + k]);
			}
		}
		data += 32;
		bytes -= 32;
	}
	uint32_t result = 0;
	for (unsigned j = 0; j < 4; j++) {
		for (unsigned k = 0; k < 4; k++) {
			result = crc_byte(result, (c[j] >> (8 * k)) & 255);
		}
	}
	while (bytes--) {
		result = crc_byte(result, *data++);
	}
	return result;
}

static uint32_t
float_bits(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}

int
moe_codec_encode(const float *input, uint32_t words, void *output, size_t capacity,
		 struct moe_codec_block *block)
{
	int scale = -30;
	int32_t low = INT32_MAX, high = INT32_MIN;
	uint32_t width;

	if (!input || !output || !block || !words || words > UINT32_MAX / 4 || capacity < (uint64_t)words * 4 + 16) {
		return -EINVAL;
	}
	*block = (struct moe_codec_block){.words = words, .bytes = words * 4, .bits = 32};
	for (uint32_t i = 0; i < words; i++) {
		uint32_t bits = float_bits(input[i]);
		unsigned exponent = (bits >> 23) & 255;
		unsigned mantissa = bits & 0x7fffff;
		if (exponent == 255 || bits == 0x80000000) {
			goto raw;
		}
		if (!exponent && !mantissa) {
			continue;
		}
		unsigned significand = mantissa | (exponent ? 0x800000 : 0);
		int required = (exponent ? 150 - (int)exponent : 149) -
			       __builtin_ctz(significand);
		scale = scale > required ? scale : required;
	}
	if (scale > 30) {
		goto raw;
	}
	double factor = ldexp(1.0, scale);
	float inverse = ldexpf(1.0f, -scale);
	for (uint32_t i = 0; i < words; i++) {
		double scaled = (double)input[i] * factor;
		if (scaled < INT32_MIN || scaled > INT32_MAX) {
			goto raw;
		}
		int32_t value = (int32_t)scaled;
		if (scaled != value || float_bits((float)value * inverse) != float_bits(input[i])) {
			goto raw;
		}
		low = value < low ? value : low;
		high = value > high ? value : high;
	}
	uint64_t span = (int64_t)high - low;
	width = span ? 64 - __builtin_clzll(span) : 0;
	if (width >= 32) {
		goto raw;
	}
	block->bytes = ((uint64_t)words * width + 7) / 8;
	block->bits = width;
	block->base = low;
	block->scale = scale;
	uint64_t accumulator = 0;
	unsigned occupied = 0;
	unsigned char *dest = output;
	for (uint32_t i = 0; i < words; i++) {
		int32_t value = (int32_t)((double)input[i] * factor);
		uint32_t delta = (uint32_t)((int64_t)value - low);
		accumulator |= (uint64_t)delta << occupied;
		occupied += width;
		while (occupied >= 8) {
			*dest++ = accumulator;
			accumulator >>= 8;
			occupied -= 8;
		}
	}
	if (occupied) {
		*dest++ = accumulator;
	}
	memset(dest, 0, 16);
	block->checksum = checksum(output, block->bytes);
	return 0;
raw:
	memcpy(output, input, (size_t)words * 4);
	memset((char *)output + (size_t)words * 4, 0, 16);
	block->checksum = checksum(output, block->bytes);
	return 0;
}

__attribute__((target("avx2"))) static uint32_t
decode_avx2(const unsigned char *source, float *dest, const struct moe_codec_block *b)
{
	if (b->bits > 25) {
		return 0;
	}
	unsigned char shuffle0[16], shuffle1[16];
	int shift[8];
	unsigned second = b->bits / 2;
	for (unsigned i = 0; i < 8; i++) {
		unsigned position = i * b->bits;
		shift[i] = position % 8;
		unsigned relative = position / 8 - (i >= 4 ? second : 0);
		unsigned char *shuffle = i < 4 ? shuffle0 : shuffle1;
		for (unsigned j = 0; j < 4; j++) {
			shuffle[(i % 4) * 4 + j] = relative + j;
		}
	}
	__m128i sh0 = _mm_loadu_si128((const __m128i *)shuffle0);
	__m128i sh1 = _mm_loadu_si128((const __m128i *)shuffle1);
	__m256i shifts = _mm256_loadu_si256((const __m256i *)shift);
	__m256i mask = _mm256_set1_epi32((1u << b->bits) - 1);
	__m256i base = _mm256_set1_epi32(b->base);
	__m256 factor = _mm256_set1_ps(ldexpf(1.0f, -b->scale));
	uint32_t i = 0;
	for (; i + 8 <= b->words; i += 8, source += b->bits) {
		__m128i a = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)source), sh0);
		__m128i c = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(source + second)), sh1);
		__m256i value = _mm256_inserti128_si256(_mm256_castsi128_si256(a), c, 1);
		value = _mm256_and_si256(_mm256_srlv_epi32(value, shifts), mask);
		value = _mm256_add_epi32(value, base);
		_mm256_storeu_ps(dest + i, _mm256_mul_ps(_mm256_cvtepi32_ps(value), factor));
	}
	return i;
}

int
moe_codec_decode(const void *input, size_t bytes, float *output,
		 const struct moe_codec_block *b)
{
	if (!input || !output || !b || bytes < (size_t)b->bytes + 16 ||
	    !b->words || b->bits > 32 || b->scale < -30 || b->scale > 30) {
		return -EINVAL;
	}
	uint64_t required = b->bits < 32 ? ((uint64_t)b->words * b->bits + 7) / 8 :
			    (uint64_t)b->words * 4;
	if (required != b->bytes) {
		return -EINVAL;
	}
	if (b->bits == 32) {
		memcpy(output, input, b->bytes);
		return 0;
	}
	const unsigned char *source = input;
	uint32_t first = __builtin_cpu_supports("avx2") ? decode_avx2(source, output, b) : 0;
	uint64_t mask = (UINT64_C(1) << b->bits) - 1;
	float factor = ldexpf(1.0f, -b->scale);
	for (uint32_t i = first; i < b->words; i++) {
		uint64_t position = (uint64_t)i * b->bits, value;
		memcpy(&value, source + position / 8, 8);
		int64_t restored = (int64_t)b->base + (int64_t)((value >> (position % 8)) & mask);
		if (restored > INT32_MAX) {
			return -EINVAL;
		}
		output[i] = (float)(int32_t)restored * factor;
	}
	return 0;
}
