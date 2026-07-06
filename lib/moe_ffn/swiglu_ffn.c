/*
 * swiglu_ffn.c — SwiGLU 前馈网络实现
 *
 * 实现 LLaMA 风格的无 bias SwiGLU FFN：
 *   output = silu(input @ W_gate) ⊙ (input @ W_up) @ W_down
 *
 * 维度变化：d_model → d_ff → d_ff → d_model
 *
 * 每个临时向量在不再需要时立即释放，展示局部资源管理。
 */

#include "swiglu_ffn.h"
#include "silu.h"
#include "matvec.h"
#include <stdlib.h>
#include <assert.h>

void swiglu_ffn(const float* input, const float* W_gate, const float* W_up,
                const float* W_down, int d_model, int d_ff, float* output) {

    /*
     * ====================================================================
     * 第 1 步：升维投影 — gate_proj = input @ W_gate
     * 将输入从 d_model 维投影到 d_ff 维（门控分支）
     * ====================================================================
     */
    float* gate_proj = (float*)malloc((size_t)d_ff * sizeof(float));
    assert(gate_proj != NULL);
    matvec_mul(input, W_gate, d_model, d_ff, gate_proj);

    /*
     * ====================================================================
     * 第 2 步：升维投影 — up_proj = input @ W_up
     * 将输入从 d_model 维投影到 d_ff 维（信息分支）
     * ====================================================================
     */
    float* up_proj = (float*)malloc((size_t)d_ff * sizeof(float));
    assert(up_proj != NULL);
    matvec_mul(input, W_up, d_model, d_ff, up_proj);

    /*
     * ====================================================================
     * 第 3 步：逐元素 SiLU 门控激活
     * gate_activated[i] = silu(gate_proj[i])
     *
     * SiLU 对门控输出进行光滑门控，允许小的负值通过（软门控），
     * 这有助于训练时的梯度流动。
     * ====================================================================
     */
    float* gate_activated = (float*)malloc((size_t)d_ff * sizeof(float));
    assert(gate_activated != NULL);

    for (int i = 0; i < d_ff; i++) {
        gate_activated[i] = silu(gate_proj[i]);
    }

    // gate_proj 已完成使命，立即释放
    free(gate_proj);

    /*
     * ====================================================================
     * 第 4 步：逐元素门控乘法（Hadamard 积）
     * gated[i] = gate_activated[i] * up_proj[i]
     *
     * 将激活后的门控信号与信息分支逐元素相乘，
     * 实现"门控线性单元"的核心操作：信息选择性通过。
     * ====================================================================
     */
    float* gated = (float*)malloc((size_t)d_ff * sizeof(float));
    assert(gated != NULL);

    for (int i = 0; i < d_ff; i++) {
        gated[i] = gate_activated[i] * up_proj[i];
    }

    // gate_activated 和 up_proj 已完成使命，立即释放
    free(gate_activated);
    free(up_proj);

    /*
     * ====================================================================
     * 第 5 步：降维投影 — output = gated @ W_down
     * 将门控结果从 d_ff 维投影回 d_model 维
     *
     * 注意：如果 input 和 output 指向同一地址（别名），
     * 此处对 output 的写入会覆盖 input。
     * 由于此时已不再需要读取 input，别名是安全的。
     * ====================================================================
     */
    matvec_mul(gated, W_down, d_ff, d_model, output);

    // gated 已完成使命，立即释放
    free(gated);
}
