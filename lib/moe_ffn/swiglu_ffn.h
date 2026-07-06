/*
 * swiglu_ffn.h — SwiGLU 前馈网络（Feed-Forward Network）模块
 *
 * 实现 LLaMA 风格的无 bias SwiGLU FFN：
 *   output = silu(input @ W_gate) ⊙ (input @ W_up) @ W_down
 *
 * 维度变化：d_model → d_ff → d_ff → d_model
 */

#ifndef SWIGLU_FFN_H
#define SWIGLU_FFN_H

/*
 * 执行 SwiGLU FFN 的前向计算
 *
 * 参数：
 *   input   — 输入向量，形状 (d_model,)，只读
 *   W_gate  — 门控投影权重，形状 (d_model, d_ff)，行主序，只读
 *   W_up    — 上投影权重，形状 (d_model, d_ff)，行主序，只读
 *   W_down  — 下投影权重，形状 (d_ff, d_model)，行主序，只读
 *   d_model — 模型维度（输入/输出维度）
 *   d_ff    — 前馈网络中间维度
 *   output  — 输出向量，形状 (d_model,)，只写，由调用者分配
 *
 * 计算步骤：
 *   1. gate_proj = input @ W_gate           — 升维投影，(d_model,) → (d_ff,)
 *   2. up_proj   = input @ W_up             — 升维投影，(d_model,) → (d_ff,)
 *   3. gate_activated[i] = silu(gate_proj[i]) — 逐元素 SiLU 门控激活
 *   4. gated[i] = gate_activated[i] * up_proj[i] — 逐元素门控乘法（Hadamard 积）
 *   5. output = gated @ W_down              — 降维投影，(d_ff,) → (d_model,)
 *
 * 内存管理：
 *   - input 和 output 允许指向同一地址（别名安全）
 *   - 函数内部在堆上分配 4 个临时向量（gate_proj, up_proj, gate_activated, gated）
 *   - 每个临时向量在不再需要时立即释放
 *   - 函数不修改任何输入数据
 */
void swiglu_ffn(const float* input, const float* W_gate, const float* W_up,
                const float* W_down, int d_model, int d_ff, float* output);

#endif /* SWIGLU_FFN_H */
