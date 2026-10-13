/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef MOE_FFN_TRACE_H
#define MOE_FFN_TRACE_H

#include "spdk/stdinc.h"

#ifdef MOE_NVTX
#include <nvtx3/nvToolsExt.h>

extern _Thread_local int moe_trace_expert_id;

static inline void moe_trace_set_expert(int expert) { moe_trace_expert_id = expert; }
static inline int moe_trace_get_expert(void) { return moe_trace_expert_id; }

static inline void
moe_trace_event(const char *operation, int expert, int stage, uint64_t bytes,
		char *label, size_t capacity, nvtxEventAttributes_t *event)
{
	const char *matrix = stage == 0 ? "Gate" : stage == 1 ? "Up" :
			     stage == 2 ? "Down" : "Gate + Up + Down";
	const char *description = operation;
	uint32_t color = 0xff378a5d;
	if (!strcmp(operation, "WEIGHT_IO")) {
		description = "I/O outstanding (submit to callback)";
		color = 0xff258ab0;
	} else if (!strcmp(operation, "IO_SUBMIT")) {
		description = "CPU submit SPDK read";
		color = 0xff136084;
	} else if (!strcmp(operation, "DECODE")) {
		description = "CPU Decode + checksum + FP32 writeback";
		color = 0xffe49331;
	} else if (!strcmp(operation, "COMPUTE")) {
		description = "CPU expert compute";
	} else if (!strcmp(operation, "ROUTER")) {
		description = "CPU Router GEMV";
		color = 0xff9467bd;
	} else if (!strcmp(operation, "TOPK")) {
		description = "CPU Top-K selection";
		color = 0xffbe74bb;
	} else if (!strcmp(operation, "NORMALIZE")) {
		description = "CPU Normalize routing weights (Softmax)";
		color = 0xffdf91c6;
	} else if (!strcmp(operation, "ROUTE")) {
		description = "CPU routing (Router + Top-K + Normalize)";
		color = 0xff755097;
	} else if (!strcmp(operation, "COMBINE")) {
		description = "CPU Aggregate weighted expert outputs";
		color = 0xffb75d78;
	} else if (!strcmp(operation, "WAIT_MATRIX")) {
		description = "Pipeline wait: compute idle, matrix I/O + decode pending";
		color = 0xffb33939;
	} else if (!strcmp(operation, "WAIT_WORKER")) {
		description = "CPU worker condvar wait for owner dispatch";
		color = 0xffb33939;
	}
	if (!strcmp(operation, "REQUEST")) {
		snprintf(label, capacity, "Request %llu / CPU MoE Forward",
			 (unsigned long long)bytes);
		color = 0xff546e7a;
	} else if (expert >= 0) {
		snprintf(label, capacity, "Expert %d / %s / %s", expert, matrix, description);
	} else {
		snprintf(label, capacity, "%s", description);
	}
	size_t used = strlen(label);
	snprintf(label + used, capacity - used, " | meta:%s expert=%d stage=%d bytes=%llu",
		 operation, expert, stage, (unsigned long long)bytes);
	*event = (nvtxEventAttributes_t){0};
	event->version = NVTX_VERSION;
	event->size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	event->messageType = NVTX_MESSAGE_TYPE_ASCII;
	event->message.ascii = label;
	event->colorType = NVTX_COLOR_ARGB;
	event->color = color;
	event->payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
	event->payload.ullValue = bytes;
}

static inline uint64_t
moe_trace_start(const char *operation, int expert, int stage, uint64_t bytes)
{
	char label[320];
	nvtxEventAttributes_t event;
	moe_trace_event(operation, expert, stage, bytes, label, sizeof(label), &event);
	return nvtxRangeStartEx(&event);
}

static inline void
moe_trace_end(uint64_t range)
{
	nvtxRangeEnd(range);
}

static inline void
moe_trace_push(const char *operation, int expert, int stage, uint64_t bytes)
{
	char label[320];
	nvtxEventAttributes_t event;
	moe_trace_event(operation, expert, stage, bytes, label, sizeof(label), &event);
	nvtxRangePushEx(&event);
}

static inline void
moe_trace_pop(void)
{
	nvtxRangePop();
}

static inline void
moe_trace_name_thread(const char *name)
{
	nvtxNameOsThreadA((uint32_t)gettid(), name);
}

static inline void
moe_trace_capture(uint64_t request, bool begin)
{
	const char *first = getenv("MOE_PROFILE_FIRST");
	const char *count = getenv("MOE_PROFILE_COUNT");
	if (!first || !count) {
		return;
	}
	uint64_t start = strtoull(first, NULL, 10), length = strtoull(count, NULL, 10);
	if (begin && request == start) {
		nvtxEventAttributes_t event = {0};
		event.version = NVTX_VERSION;
		event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
		event.message.registered = nvtxDomainRegisterStringA(NULL, "PROFILE_WINDOW");
		nvtxRangePushEx(&event);
	} else if (!begin && length && request == start + length - 1) {
		nvtxRangePop();
	}
}
#else
static inline uint64_t
moe_trace_start(const char *operation, int expert, int stage, uint64_t bytes)
{
	return 0;
}

static inline void moe_trace_end(uint64_t range) {}
static inline void moe_trace_push(const char *operation, int expert, int stage, uint64_t bytes) {}
static inline void moe_trace_pop(void) {}
static inline void moe_trace_name_thread(const char *name) {}
static inline void moe_trace_capture(uint64_t request, bool begin) {}
static inline void moe_trace_set_expert(int expert) {}
static inline int moe_trace_get_expert(void) { return -1; }
#endif

#endif
