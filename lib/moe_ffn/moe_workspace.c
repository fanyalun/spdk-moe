/* SPDX-License-Identifier: BSD-3-Clause */

#include "moe_workspace.h"
#include "silu.h"
#include "softmax.h"
#include "topk.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int moe_workspace_init(struct moe_workspace *ws, int d_model, int d_ff,
                       int num_experts, int top_k, enum moe_kernel kernel)
{
    if (!ws || d_model <= 0 || d_ff <= 0 || num_experts <= 0 ||
        top_k <= 0 || top_k > num_experts || !moe_kernel_supported(kernel)) {
        return -EINVAL;
    }
    memset(ws, 0, sizeof(*ws));
    uint64_t count = (uint64_t)d_model * ((uint64_t)top_k + 1) +
                     2 * (uint64_t)d_ff + num_experts + top_k;
    if (count > SIZE_MAX / sizeof(float) || (size_t)top_k > SIZE_MAX / sizeof(int)) {
        return -EOVERFLOW;
    }
    ws->storage = malloc((size_t)count * sizeof(float));
    ws->indices = malloc((size_t)top_k * sizeof(int));
    if (!ws->storage || !ws->indices) {
        moe_workspace_destroy(ws);
        return -ENOMEM;
    }
    /* Touch every allocated page before accepting requests. */
    memset(ws->storage, 0, (size_t)count * sizeof(float));
    memset(ws->indices, 0, (size_t)top_k * sizeof(int));
    ws->d_model = d_model;
    ws->d_ff = d_ff;
    ws->num_experts = num_experts;
    ws->top_k = top_k;
    ws->kernel = kernel;
    ws->input = ws->storage;
    ws->gate = ws->input + d_model;
    ws->up = ws->gate + d_ff;
    ws->logits = ws->up + d_ff;
    ws->weights = ws->logits + num_experts;
    ws->expert_outputs = ws->weights + top_k;
    return 0;
}

void moe_workspace_destroy(struct moe_workspace *ws)
{
    if (ws) {
        free(ws->storage);
        free(ws->indices);
        memset(ws, 0, sizeof(*ws));
    }
}

static int finite_vector(const float *values, int n)
{
    for (int i = 0; i < n; i++) {
        if (!isfinite(values[i])) {
            return -ERANGE;
        }
    }
    return 0;
}

int moe_route(struct moe_workspace *ws, const float *input, const float *router)
{
    if (finite_vector(input, ws->d_model)) {
        return -ERANGE;
    }
    memmove(ws->input, input, (size_t)ws->d_model * sizeof(float));
    matvec_ordered(ws->input, router, ws->d_model, ws->num_experts, ws->logits, ws->kernel, 0);
    if (finite_vector(ws->logits, ws->num_experts)) {
        return -ERANGE;
    }
    topk_select(ws->logits, ws->num_experts, ws->top_k, ws->indices, ws->weights);
    softmax(ws->weights, ws->top_k);
    return finite_vector(ws->weights, ws->top_k);
}

int moe_expert_stage(struct moe_workspace *ws, enum moe_stage stage,
                     const float *weight, float *gate, float *up, float *output, int packed)
{
    switch (stage) {
    case MOE_STAGE_GATE:
        matvec_ordered(ws->input, weight, ws->d_model, ws->d_ff, gate, ws->kernel, packed);
        return 0;
    case MOE_STAGE_UP:
        matvec_ordered(ws->input, weight, ws->d_model, ws->d_ff, up, ws->kernel, packed);
        for (int i = 0; i < ws->d_ff; i++) {
            gate[i] = silu(gate[i]) * up[i];
        }
        return 0;
    case MOE_STAGE_DOWN:
        matvec_ordered(gate, weight, ws->d_ff, ws->d_model, output, ws->kernel, packed);
        return finite_vector(output, ws->d_model);
    default:
        return -EINVAL;
    }
}

int moe_expert(struct moe_workspace *ws, const float *gate, const float *up,
               const float *down, int selected, int packed)
{
    if (selected < 0 || selected >= ws->top_k) {
        return -EINVAL;
    }
    float *output = ws->expert_outputs + (size_t)selected * ws->d_model;
    moe_expert_stage(ws, MOE_STAGE_GATE, gate, ws->gate, ws->up, output, packed);
    moe_expert_stage(ws, MOE_STAGE_UP, up, ws->gate, ws->up, output, packed);
    return moe_expert_stage(ws, MOE_STAGE_DOWN, down, ws->gate, ws->up, output, packed);
}

int moe_combine(struct moe_workspace *ws, float *output)
{
    memset(output, 0, (size_t)ws->d_model * sizeof(float));
    for (int k = 0; k < ws->top_k; k++) {
        for (int j = 0; j < ws->d_model; j++) {
            output[j] = fmaf(ws->weights[k], ws->expert_outputs[(size_t)k * ws->d_model + j],
                             output[j]);
        }
    }
    return finite_vector(output, ws->d_model);
}

int swiglu_moe_workspace(struct moe_workspace *ws, const float *input, const float *router,
                         const float *gate[], const float *up[], const float *down[],
                         int packed, float *output)
{
    int rc = moe_route(ws, input, router);
    if (rc) {
        return rc;
    }
    for (int k = 0; k < ws->top_k; k++) {
        int e = ws->indices[k];
        rc = moe_expert(ws, gate[e], up[e], down[e], k, packed);
        if (rc) {
            return rc;
        }
    }
    return moe_combine(ws, output);
}
