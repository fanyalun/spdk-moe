// SPDX-License-Identifier: BSD-3-Clause
#ifndef MOE_HUGEPAGES_H
#define MOE_HUGEPAGES_H

#include "spdk/event.h"

#define MOE_DEFAULT_MEM_MIB 256

// Returned strings must remain valid until spdk_app_fini() and are freed by the caller.
int moe_hugepages_prepare(struct spdk_app_opts *opts, char **env_context, char **hugedir);

#endif
