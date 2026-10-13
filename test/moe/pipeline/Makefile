#  SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk
APP = moe_pipeline_test
C_SRCS = ../test_moe_pipeline.c
C_SRCS += startup_faults.c
LDFLAGS += -Wl,--wrap=spdk_dma_zmalloc_socket
LDFLAGS += -Wl,--wrap=spdk_bdev_read
SPDK_LIB_LIST = $(ALL_MODULES_LIST) event event_bdev
REFERENCE_LIB ?= $(SPDK_ROOT_DIR)/../MoE-compute/build_accuracy/libmoe_reference.a
SYS_LIBS += $(REFERENCE_LIB) -lm
include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
