/*
 * swiglu_moe.h — Sparse Mixture of Experts（稀疏 MoE）模块
 *
 * 实现 Top-k Sparse MoE with Linear Router（对标 Mixtral 8×7B）：
 *
 *   logits      = input @ W_router                        — (d_model,) → (num_experts,)
 *   indices, top_logits = topk(logits, top_k)             — 选出值最大的 top_k 个
 *   weights     = softmax(top_logits)                      — 对选中的 top_k 个 logits 归一化
 *   对每个选中专家执行 swiglu_ffn(input, W_gate[e], W_up[e], W_down[e], ...)
 *   output = Σ weights[i] × expert_out[i]                  — 加权求和，(d_model,)
 *
 * MoE 层通过 swiglu_ffn 调用每个专家，作为独立的上层编排模块存在。
 */

#ifndef SWIGLU_MOE_H
#define SWIGLU_MOE_H

/*
 * 执行 Sparse MoE 层的前向计算
 *
 * 参数：
 *   input        — 输入向量，形状 (d_model,)，只读
 *   W_router     — 路由器权重矩阵，形状 (d_model, num_experts)，行主序，只读
 *   W_gate[]     — 长度为 num_experts 的指针数组，每个指向 (d_model, d_ff)，只读
 *   W_up[]       — 长度为 num_experts 的指针数组，每个指向 (d_model, d_ff)，只读
 *   W_down[]     — 长度为 num_experts 的指针数组，每个指向 (d_ff, d_model)，只读
 *   num_experts  — 专家总数，运行时指定
 *   top_k        — 每个 token 激活的专家数，运行时指定（必须 1 ≤ top_k ≤ num_experts）
 *   d_model      — 模型维度（输入/输出维度）
 *   d_ff         — 专家前馈网络中间维度
 *   output       — 输出向量，形状 (d_model,)，只写，由调用者分配
 *
 * 计算步骤：
 *   1. logits = input @ W_router                                    — 路由 logits
 *   2. indices, top_logits = topk_select(logits, num_experts, top_k) — Top-k 选择
 *   3. softmax(top_logits)                                           — 权重归一化
 *   4. 对每个选中专家执行 swiglu_ffn(input, ..., expert_out[i])      — 专家计算
 *   5. output = Σ weights[i] × expert_out[i]                         — 加权合成
 *
 * 内存管理：
 *   - logits 使用后立即释放
 *   - 各专家输出 expert_out[i] 在加权求和后统一释放
 *   - input 和 output 允许指向同一地址（别名安全）
 */
void swiglu_moe(const float* input, const float* W_router,
                const float* W_gate[], const float* W_up[], const float* W_down[],
                int num_experts, int top_k, int d_model, int d_ff, float* output);

#endif /* SWIGLU_MOE_H */
