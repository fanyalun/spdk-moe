# SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk

APP = moe_tgt
C_SRCS := ../moe_tgt.c
SPDK_LIB_LIST = $(ALL_MODULES_LIST) event event_nvmf
CFLAGS += -I$(SPDK_ROOT_DIR)/lib
SYS_LIBS += -lm

include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
