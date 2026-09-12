#include "moe_cache.h"
#include "swiglu_moe.h"
#include "swiglu_ffn.h"
#include "matvec.h"
#include "topk.h"
#include "softmax.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
	char directory[] = "/tmp/moe-cache-test.XXXXXX";
	char path[512];
	const char *names[] = {"W_gate", "W_up", "W_down"};
	float data[3][3][8];
	const float *gate[3], *up[3], *down[3];
	float router[6] = {1, 0, -1, 0, 1, -1};
	float inputs[3][2] = {{1, 2}, {-2, 1}, {1, 2}};
	int capacity, expert, matrix, element, request, selected;
	assert(mkdtemp(directory) != NULL);
	for (expert = 0; expert < 3; expert++) {
		for (matrix = 0; matrix < 3; matrix++) {
			FILE *file;
			for (element = 0; element < 8; element++) {
				data[expert][matrix][element] = (expert + matrix + element - 5) * 0.1f;
			}
			snprintf(path, sizeof(path), "%s/%s_%d_2x4.bin", directory, names[matrix], expert);
			file = fopen(path, "wb");
			assert(file != NULL);
			assert(fwrite(data[expert][matrix], sizeof(float), 8, file) == 8);
			assert(fclose(file) == 0);
		}
		gate[expert] = data[expert][0];
		up[expert] = data[expert][1];
		down[expert] = data[expert][2];
	}
	for (capacity = 1; capacity <= 3; capacity++) {
		struct moe_cache cache;
		const float *cached_gate, *cached_up, *cached_down;
		assert(moe_cache_init(&cache, capacity, 2, 4, directory) == 0);
		for (request = 0; request < 3; request++) {
			float expected[2], actual[2] = {0}, logits[3], weights[2], output[2];
			int indices[2];
			swiglu_moe(inputs[request], router, gate, up, down, 3, 2, 2, 4, expected);
			matvec_mul(inputs[request], router, 2, 3, logits);
			topk_select(logits, 3, 2, indices, weights);
			softmax(weights, 2);
			for (selected = 0; selected < 2; selected++) {
				assert(moe_cache_get(&cache, indices[selected], &cached_gate, &cached_up, &cached_down) == 0);
				swiglu_ffn(inputs[request], cached_gate, cached_up, cached_down, 2, 4, output);
				for (element = 0; element < 2; element++) {
					actual[element] += weights[selected] * output[element];
				}
			}
			for (element = 0; element < 2; element++) {
				assert(isfinite(actual[element]) && fabsf(actual[element] - expected[element]) < 1e-5f);
			}
		}
		assert(moe_cache_get(&cache, 99, &cached_gate, &cached_up, &cached_down) != 0);
		assert(moe_cache_get(&cache, 0, &cached_gate, &cached_up, &cached_down) == 0);
		assert(moe_cache_get(&cache, 0, &cached_gate, &cached_up, &cached_down) == 0);
		moe_cache_destroy(&cache);
	}
	for (expert = 0; expert < 3; expert++) {
		for (matrix = 0; matrix < 3; matrix++) {
			snprintf(path, sizeof(path), "%s/%s_%d_2x4.bin", directory, names[matrix], expert);
			assert(unlink(path) == 0);
		}
	}
	assert(rmdir(directory) == 0);
	puts("cache regression PASS");
	return 0;
}
