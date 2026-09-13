#  SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk
APP = moe_accuracy_initiator
C_SRCS = ../test_moe_accuracy_initiator.c
SPDK_LIB_LIST = moe_ffn $(SOCK_MODULES_LIST) nvme vmd
REFERENCE_LIB ?= $(SPDK_ROOT_DIR)/../MoE-compute/build_accuracy/libmoe_reference.a
SYS_LIBS += $(REFERENCE_LIB) -lm
include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
