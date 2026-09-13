/* SPDX-License-Identifier: BSD-3-Clause */

#include "spdk/stdinc.h"
#include "spdk/rpc.h"
#include "spdk/util.h"
#include "moe_ffn/moe_config.h"
#include "bdev_moe.h"

struct create_rpc {
	struct moe_create_opts opts;
	struct spdk_jsonrpc_request *request;
};

#define DECODER(field, decoder, optional) \
	{#field, offsetof(struct moe_create_opts, field), decoder, optional}

static const struct spdk_json_object_decoder create_decoders[] = {
	DECODER(name, spdk_json_decode_string, false),
	DECODER(backend, spdk_json_decode_string, false),
	DECODER(base_bdev, spdk_json_decode_string, true),
	DECODER(weight_dir, spdk_json_decode_string, true),
	DECODER(kernel, spdk_json_decode_string, true),
	DECODER(diagnostics, spdk_json_decode_string, true),
	DECODER(d_model, spdk_json_decode_int32, true),
	DECODER(d_ff, spdk_json_decode_int32, true),
	DECODER(num_experts, spdk_json_decode_int32, true),
	DECODER(top_k, spdk_json_decode_int32, true),
	DECODER(cache_slots, spdk_json_decode_int32, true),
	DECODER(io_size, spdk_json_decode_uint32, true),
	DECODER(io_depth, spdk_json_decode_uint32, true),
	DECODER(prefetch, spdk_json_decode_uint32, true),
	DECODER(compute_cpu, spdk_json_decode_int32, true),
	DECODER(compute_threads, spdk_json_decode_int32, true),
};

static void
free_create(struct create_rpc *r)
{
	free((void *)r->opts.name);
	free((void *)r->opts.backend);
	free((void *)r->opts.base_bdev);
	free((void *)r->opts.weight_dir);
	free((void *)r->opts.kernel);
	free((void *)r->opts.diagnostics);
	free(r);
}

static void
create_done(void *arg, int status)
{
	struct create_rpc *r = arg;

	if (status) {
		spdk_jsonrpc_send_error_response(r->request, status, strerror(-status));
	} else {
		struct spdk_json_write_ctx *w = spdk_jsonrpc_begin_result(r->request);
		spdk_json_write_string(w, r->opts.name);
		spdk_jsonrpc_end_result(r->request, w);
	}
	free_create(r);
}

static void
rpc_bdev_moe_create(struct spdk_jsonrpc_request *request, const struct spdk_json_val *params)
{
	struct create_rpc *r = calloc(1, sizeof(*r));

	if (!r) {
		spdk_jsonrpc_send_error_response(request, -ENOMEM, "allocation failed");
		return;
	}
	r->request = request;
	r->opts = (struct moe_create_opts){
		.d_model = MOE_D_MODEL, .d_ff = MOE_D_FF, .num_experts = MOE_NUM_EXPERTS,
		.top_k = MOE_TOP_K, .cache_slots = 7, .io_size = 1024 * 1024, .io_depth = 4, .prefetch = 2,
		.compute_cpu = -1,
		.compute_threads = 1,
	};
	if (!params || spdk_json_decode_object(params, create_decoders, SPDK_COUNTOF(create_decoders), &r->opts)) {
		create_done(r, -EINVAL);
		return;
	}
	if (!r->opts.weight_dir) {
		r->opts.weight_dir = strdup(MOE_WEIGHT_DIR);
	}
	int rc = bdev_moe_create_async(&r->opts, create_done, r);
	if (rc) {
		create_done(r, rc);
	}
}
SPDK_RPC_REGISTER("bdev_moe_create", rpc_bdev_moe_create, SPDK_RPC_RUNTIME)

struct delete_rpc {
	char *name;
};

static const struct spdk_json_object_decoder delete_decoders[] = {
	{"name", offsetof(struct delete_rpc, name), spdk_json_decode_string},
};

static void
rpc_delete_done(void *arg, int status)
{
	struct spdk_jsonrpc_request *request = arg;
	if (status) {
		spdk_jsonrpc_send_error_response(request, status, strerror(-status));
	} else {
		spdk_jsonrpc_send_bool_response(request, true);
	}
}

static void
rpc_bdev_moe_delete(struct spdk_jsonrpc_request *request, const struct spdk_json_val *params)
{
	struct delete_rpc r = {0};
	if (!params || spdk_json_decode_object(params, delete_decoders, SPDK_COUNTOF(delete_decoders), &r)) {
		spdk_jsonrpc_send_error_response(request, -EINVAL, "expected name");
	} else {
		bdev_moe_delete(r.name, rpc_delete_done, request);
	}
	free(r.name);
}
SPDK_RPC_REGISTER("bdev_moe_delete", rpc_bdev_moe_delete, SPDK_RPC_RUNTIME)
