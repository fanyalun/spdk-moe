# SPDX-License-Identifier: BSD-3-Clause

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk

SO_VER := 1
SO_MINOR := 0

C_SRCS = bdev_moe.c moe_subsystem.c moe_cache.c
C_SRCS += moe_store.c bdev_moe_rpc.c
LIBNAME = bdev_moe

CFLAGS += -I$(SPDK_ROOT_DIR)/lib -I$(SPDK_ROOT_DIR)/lib/moe_ffn
CFLAGS += -fno-fast-math
SPDK_MAP_FILE = $(SPDK_ROOT_DIR)/mk/spdk_blank.map

include $(SPDK_ROOT_DIR)/mk/spdk.lib.mk
