/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef MOE_AFFINITY_H
#define MOE_AFFINITY_H

#include "spdk/stdinc.h"

int moe_affinity_init(void);
bool moe_affinity_allowed(int cpu);
bool moe_affinity_reserved(int cpu);
int moe_affinity_default_reactor(void);

#endif
