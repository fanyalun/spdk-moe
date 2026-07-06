/*
 * swiglu_moe.c — Sparse Mixture of Experts（稀疏 MoE）实现
 *
 * 实现 Top-k Sparse MoE with Linear Router（对标 Mixtral 8×7B）：
 *
 *   logits      = input @ W_router
 *   indices, top_logits = topk(logits, top_k)
 *   weights     = softmax(top_logits)
 *   expert_out[i] = swiglu_ffn(input, W_gate[indices[i]], ...)
 *   output = Σ weights[i] × expert_out[i]
 *
 * 维度变化：d_model → num_experts（路由）→ d_model × top_k（专家计算）→ d_model（加权求和）
 *
 * 无共享专家、无辅助损失、无 capacity factor、无 expert drop。
 * 路由器无 bias。
 */

#include "swiglu_moe.h"
#include "swiglu_ffn.h"
#include "matvec.h"
#include "softmax.h"
#include "topk.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void swiglu_moe(const float* input, const float* W_router,
                const float* W_gate[], const float* W_up[], const float* W_down[],
                int num_experts, int top_k, int d_model, int d_ff, float* output) {

    /*
     * ====================================================================
     * 第 1 步：路由 logits 计算
     * logits = input @ W_router
     * 维度：(d_model,) → (num_experts,)
     *
     * W_router 形状 (d_model, num_experts)，行主序
     * 每个 logits[j] 代表路由器"认为"第 j 个专家对当前 token 的相关程度
     * ====================================================================
     */
    float* logits = (float*)malloc((size_t)num_experts * sizeof(float));
    assert(logits != NULL);
    matvec_mul(input, W_router, d_model, num_experts, logits);

    /*
     * ====================================================================
     * 第 2 步：Top-k 选择
     * 从 num_experts 个 logits 中选出值最大的 top_k 个
     * indices[]：选中的专家索引
     * top_logits[]：对应的原始 logits 值
     *
     * 只激活最相关的 top_k 个专家，实现稀疏计算
     * ====================================================================
     */
    int* indices = (int*)malloc((size_t)top_k * sizeof(int));
    assert(indices != NULL);
    float* top_logits = (float*)malloc((size_t)top_k * sizeof(float));
    assert(top_logits != NULL);

    topk_select(logits, num_experts, top_k, indices, top_logits);

    // logits 已完成使命，立即释放
    free(logits);

    /*
     * ====================================================================
     * 第 3 步：Softmax 归一化
     * weights = softmax(top_logits)
     *
     * 对选中的 top_k 个 logits 做 softmax 归一化，
     * 使得 weights 的和为 1，代表路由器对各选中专家的"信任权重"。
     * 原地修改 top_logits，其内容变为归一化权重。
     * ====================================================================
     */
    softmax(top_logits, top_k);
    float* weights = top_logits;  // softmax 后的结果即为权重

    /*
     * ====================================================================
     * 第 4 步：专家计算
     * 对每个选中的专家，执行完整的 swiglu_ffn 前向计算：
     *   expert_out[i] = swiglu_ffn(input,
     *                               W_gate[indices[i]],
     *                               W_up[indices[i]],
     *                               W_down[indices[i]],
     *                               d_model, d_ff)
     *
     * 每个专家是独立的 swiglu_ffn 实例，
     * 各自拥有独立的 W_gate、W_up、W_down 权重。
     * 各专家计算在概念上是并行的（实际按顺序执行）。
     * ====================================================================
     */
    float** expert_out = (float**)malloc((size_t)top_k * sizeof(float*));
    assert(expert_out != NULL);

    for (int i = 0; i < top_k; i++) {
        expert_out[i] = (float*)malloc((size_t)d_model * sizeof(float));
        assert(expert_out[i] != NULL);

        int expert_id = indices[i];

        swiglu_ffn(input,
                   W_gate[expert_id],
                   W_up[expert_id],
                   W_down[expert_id],
                   d_model, d_ff,
                   expert_out[i]);
    }

    /*
     * ====================================================================
     * 第 5 步：加权求和
     * output = Σ_{i=0}^{top_k-1} weights[i] × expert_out[i]
     *
     * 将 top_k 个专家的输出按 softmax 权重进行加权累加，
     * 生成最终的输出向量。
     *
     * 注意：output 需要先初始化为零，因为采用累加模式写入。
     * 如果 input 和 output 指向同一地址（别名），
     * 此时对 output 的写操作是安全的——input 在所有专家计算中已被完整使用。
     * ====================================================================
     */

    // 输出向量清零，为加权累加做准备
    memset(output, 0, (size_t)d_model * sizeof(float));

    // 逐元素加权累加
    for (int i = 0; i < top_k; i++) {
        float w = weights[i];
        for (int j = 0; j < d_model; j++) {
            output[j] += w * expert_out[i][j];
        }
    }

    /*
     * ====================================================================
     * 资源释放
     *
     * 各专家输出在加权求和完成后统一释放（展示 MoE "并行专家计算"语义：
     * 所有专家同时在计算完成后统一合成）。
     * weights（即 top_logits 的别名）与 indices 一并释放。
     * ====================================================================
     */
    for (int i = 0; i < top_k; i++) {
        free(expert_out[i]);
    }
    free(expert_out);
    free(weights);    // 即 top_logits
    free(indices);
}
