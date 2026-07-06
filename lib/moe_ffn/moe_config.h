#ifndef MOE_CONFIG_H
#define MOE_CONFIG_H

/* ===== 维度参数（须与 ffn 侧 moe_config.h 同名宏值一致）===== */
#define MOE_D_MODEL         2048
#define MOE_D_FF            7168
#define MOE_NUM_EXPERTS     8
#define MOE_TOP_K           2

/* ===== 种子 ===== */
#define MOE_SEED_DEFAULT    42

/* ===== NVMe-oF 连接参数 ===== */
#define MOE_TARGET_ADDR     "127.0.0.1"
#define MOE_TARGET_PORT     4420
#define MOE_TRANSPORT       "tcp"
#define MOE_NQN             "nqn.2024-07.io.spdk:moe_target"

/* ===== Vendor-specific NVMe 命令 ===== */
#define MOE_VENDOR_OPCODE   0xC1

/* ===== 权重文件目录 ===== */
#define MOE_WEIGHT_DIR      "/tmp/moe_weights"

/* ===== 校验容差 ===== */
#define MOE_MAX_ABS_ERROR   1e-5f

/* ===== 派生值 ===== */
#define MOE_ROUTER_COLS     MOE_NUM_EXPERTS
#define MOE_INPUT_BYTES     (MOE_D_MODEL * (int)sizeof(float))
#define MOE_ROUTER_ELEMS    (MOE_D_MODEL * MOE_ROUTER_COLS)
#define MOE_EXPERT_ELEMS    (MOE_D_MODEL * MOE_D_FF)
#define MOE_DOWN_ELEMS      (MOE_D_FF * MOE_D_MODEL)
#define MOE_TOTAL_FILES     (1 + 3 * MOE_NUM_EXPERTS)

#endif /* MOE_CONFIG_H */
