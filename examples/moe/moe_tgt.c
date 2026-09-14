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
#include "spdk/cpuset.h"

#include "moe_ffn/moe_config.h"
#include "moe_ffn/moe_affinity.h"

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
	char default_mask[32];
	int rc;

	rc = moe_affinity_init();
	if (rc) {
		fprintf(stderr, "Cannot capture CPU affinity or invalid MOE_INITIATOR_CPU: %d\n", rc);
		return 1;
	}
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

	if (!opts.reactor_mask && !opts.lcore_map) {
		int cpu = opts.main_core >= 0 ? opts.main_core : moe_affinity_default_reactor();
		if (cpu < 0 || !moe_affinity_allowed(cpu)) {
			fprintf(stderr, "No default reactor: need two allowed target physical cores "
				"apart from MOE_INITIATOR_CPU (default 0); specify -m to override\n");
			return 1;
		}
		snprintf(default_mask, sizeof(default_mask), "[%d]", cpu);
		opts.reactor_mask = default_mask;
		fprintf(stderr, "MoE default reactor_cpu=%d (startup affinity preserved)\n", cpu);
	}
	if (opts.reactor_mask) {
		struct spdk_cpuset set = {};
		if (spdk_cpuset_parse(&set, opts.reactor_mask)) {
			fprintf(stderr, "Invalid reactor CPU mask\n");
			return 1;
		}
		for (uint32_t cpu = 0; cpu < SPDK_CPUSET_SIZE; cpu++) {
			if (spdk_cpuset_get_cpu(&set, cpu) && !moe_affinity_allowed(cpu)) {
				fprintf(stderr, "Reactor CPU %u is outside startup affinity\n", cpu);
				return 1;
			}
		}
	}
	rc = spdk_app_start(&opts, moe_tgt_started, NULL);
	spdk_app_fini();
	return rc;
}
