# SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk

APP = moe_bench
C_SRCS := ../moe_bench.c
SPDK_LIB_LIST = $(SOCK_MODULES_LIST) nvme vmd
CFLAGS += -I$(SPDK_ROOT_DIR)/lib
SYS_LIBS += -lm

include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
