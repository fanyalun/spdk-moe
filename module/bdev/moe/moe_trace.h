/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_TRACE_H
#define MOE_TRACE_H

#include "spdk/stdinc.h"

#ifdef MOE_NVTX
#include <nvtx3/nvToolsExt.h>

static inline uint64_t
moe_trace_start(const char *operation, int expert, int stage, uint64_t bytes)
{
	char label[128];
	nvtxEventAttributes_t event = {0};

	snprintf(label, sizeof(label), "%s expert=%d stage=%d bytes=%llu",
		 operation, expert, stage, (unsigned long long)bytes);
	event.version = NVTX_VERSION;
	event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	event.messageType = NVTX_MESSAGE_TYPE_ASCII;
	event.message.ascii = label;
	event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
	event.payload.ullValue = bytes;
	return nvtxRangeStartEx(&event);
}

static inline void
moe_trace_end(uint64_t range)
{
	nvtxRangeEnd(range);
}
#else
static inline uint64_t
moe_trace_start(const char *operation, int expert, int stage, uint64_t bytes)
{
	return 0;
}

static inline void
moe_trace_end(uint64_t range)
{
}
#endif

#endif
