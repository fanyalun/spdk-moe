#  SPDX-License-Identifier: BSD-3-Clause


def bdev_moe_create(client, name, backend, **options):
    """Import weights, warm up inference, then publish the MoE bdev."""
    params = dict(name=name, backend=backend)
    params.update({key: value for key, value in options.items() if value is not None})
    return client.call('bdev_moe_create', params)


def bdev_moe_delete(client, name):
    """Drain and delete a MoE bdev."""
    return client.call('bdev_moe_delete', dict(name=name))
