/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_WORKSPACE_H
#define MOE_WORKSPACE_H

#include "matvec_packed.h"

struct moe_workspace {
    int d_model, d_ff, num_experts, top_k;
    enum moe_kernel kernel;
    float *storage, *input, *gate, *up, *logits, *weights, *expert_outputs;
    int *indices;
};

int moe_workspace_init(struct moe_workspace *ws, int d_model, int d_ff,
                       int num_experts, int top_k, enum moe_kernel kernel);
void moe_workspace_destroy(struct moe_workspace *ws);
int moe_route(struct moe_workspace *ws, const float *input, const float *router);
int moe_expert(struct moe_workspace *ws, const float *gate, const float *up,
               const float *down, int selected, int packed);
int moe_combine(struct moe_workspace *ws, float *output);
int swiglu_moe_workspace(struct moe_workspace *ws, const float *input, const float *router,
                         const float *gate[], const float *up[], const float *down[],
                         int packed, float *output);

#endif
