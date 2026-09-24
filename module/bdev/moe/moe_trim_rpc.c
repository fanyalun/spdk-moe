// SPDX-License-Identifier: BSD-3-Clause
#include "spdk/stdinc.h"
#include "spdk/bdev.h"
#include "spdk/rpc.h"
#include "spdk/util.h"
#include "spdk/thread.h"

struct trim_request {
	char *name;
	struct spdk_jsonrpc_request *rpc;
	struct spdk_bdev_desc *desc;
	struct spdk_io_channel *channel;
	struct spdk_bdev_io_wait_entry wait;
	uint64_t offset, count, total;
	bool removed;
};

static void trim_submit(void *arg);

static void
trim_finish(struct trim_request *r, int status)
{
	if (status) {
		spdk_jsonrpc_send_error_response(r->rpc, status, strerror(-status));
	} else {
		spdk_jsonrpc_send_bool_response(r->rpc, true);
	}
	if (r->channel) {
		spdk_put_io_channel(r->channel);
	}
	if (r->desc) {
		spdk_bdev_close(r->desc);
	}
	free(r->name);
	free(r);
}

static void
trim_done(struct spdk_bdev_io *io, bool success, void *arg)
{
	struct trim_request *r = arg;
	spdk_bdev_free_io(io);
	if (!success) {
		trim_finish(r, -EIO);
		return;
	}
	r->offset += r->count;
	trim_submit(r);
}

static void
trim_submit(void *arg)
{
	struct trim_request *r = arg;
	struct spdk_bdev *bdev = spdk_bdev_desc_get_bdev(r->desc);
	if (r->removed || r->offset == r->total) {
		trim_finish(r, r->removed ? -ENODEV : 0);
		return;
	}
	// Bound each operation to 1GiB; SPDK splits further for device limits.
	r->count = spdk_min(r->total - r->offset,
			    UINT64_C(1073741824) / spdk_bdev_get_block_size(bdev));
	int rc = spdk_bdev_unmap_blocks(r->desc, r->channel, r->offset, r->count, trim_done, r);
	if (rc == -ENOMEM) {
		r->wait.bdev = bdev;
		r->wait.cb_fn = trim_submit;
		r->wait.cb_arg = r;
		rc = spdk_bdev_queue_io_wait(bdev, r->channel, &r->wait);
	}
	if (rc) {
		trim_finish(r, rc);
	}
}

static void
trim_event(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *arg)
{
	struct trim_request *r = arg;
	if (type == SPDK_BDEV_EVENT_REMOVE) {
		r->removed = true;
	}
}

static void
rpc_moe_trim(struct spdk_jsonrpc_request *request, const struct spdk_json_val *params)
{
	static const struct spdk_json_object_decoder decoders[] = {
		{"name", offsetof(struct trim_request, name), spdk_json_decode_string},
	};
	struct trim_request *r = calloc(1, sizeof(*r));
	if (!r) {
		spdk_jsonrpc_send_error_response(request, -ENOMEM, "allocation failed");
		return;
	}
	r->rpc = request;
	if (!params || spdk_json_decode_object(params, decoders, SPDK_COUNTOF(decoders), r)) {
		trim_finish(r, -EINVAL);
		return;
	}
	int rc = spdk_bdev_open_ext(r->name, true, trim_event, r, &r->desc);
	if (rc) {
		trim_finish(r, rc);
		return;
	}
	struct spdk_bdev *bdev = spdk_bdev_desc_get_bdev(r->desc);
	if (!spdk_bdev_io_type_supported(bdev, SPDK_BDEV_IO_TYPE_UNMAP)) {
		trim_finish(r, -ENOTSUP);
		return;
	}
	r->channel = spdk_bdev_get_io_channel(r->desc);
	if (!r->channel) {
		trim_finish(r, -ENOMEM);
		return;
	}
	r->total = spdk_bdev_get_num_blocks(bdev);
	trim_submit(r);
}
SPDK_RPC_REGISTER("bdev_moe_trim", rpc_moe_trim, SPDK_RPC_RUNTIME)
