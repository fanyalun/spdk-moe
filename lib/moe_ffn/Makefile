#  SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk

SO_VER := 1
SO_MINOR := 0

C_SRCS = silu.c matvec.c swiglu_ffn.c softmax.c topk.c swiglu_moe.c test_utils.c
C_SRCS += matvec_packed.c moe_workspace.c moe_affinity.c
LIBNAME = moe_ffn

CFLAGS += -I$(SPDK_ROOT_DIR)/lib/moe_ffn
CFLAGS += -fno-fast-math -ffp-contract=fast
SYS_LIBS += -lm

SPDK_MAP_FILE = $(SPDK_ROOT_DIR)/mk/spdk_blank.map

include $(SPDK_ROOT_DIR)/mk/spdk.lib.mk
