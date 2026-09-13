/* SPDX-License-Identifier: BSD-3-Clause
 *
 * moe_tgt.c — MoE FFN NVMe-oF target 入口。
 *
 * 进程启动 SPDK event 框架并加载 moe_tgt.json。bdev_moe 模块在 bdev
 * 子系统初始化阶段加载权重并创建 moe_bdev_0，JSON RPC 配置负责创建
 * TCP transport、NVMe-oF subsystem、listener 和 namespace。
 */

#include "spdk/stdinc.h"

#include "spdk/env.h"
#include "spdk/event.h"

#include "moe_ffn/moe_config.h"

static void
moe_tgt_started(void *arg1)
{
	printf("MoE target started: %s:%d, nqn=%s\n",
	       MOE_TARGET_ADDR, MOE_TARGET_PORT, MOE_NQN);
}

int
main(int argc, char **argv)
{
	struct spdk_app_opts opts = {};
	int rc;

	spdk_app_opts_init(&opts, sizeof(opts));
	opts.name = "moe_tgt";
	opts.mem_size = 1536;

	if (argc == 2 && argv[1][0] != '-') {
		opts.json_config_file = argv[1];
	} else {
		rc = spdk_app_parse_args(argc, argv, &opts, NULL, NULL, NULL, NULL);
		if (rc != SPDK_APP_PARSE_ARGS_SUCCESS) {
			return rc == SPDK_APP_PARSE_ARGS_HELP ? 0 : 1;
		}
	}

	rc = spdk_app_start(&opts, moe_tgt_started, NULL);
	spdk_app_fini();
	return rc;
}
