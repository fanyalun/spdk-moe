// SPDX-License-Identifier: BSD-3-Clause
#include "spdk/stdinc.h"

#include <mntent.h>
#include <sys/vfs.h>
#include <linux/magic.h>

#include "moe_hugepages.h"

static bool
usable_mount(const char *dir)
{
	struct statfs fs;

	return statfs(dir, &fs) == 0 && fs.f_type == HUGETLBFS_MAGIC &&
	       fs.f_bsize == 2 * 1024 * 1024 && access(dir, R_OK | W_OK | X_OK) == 0;
}

static bool
overrides_memory(const char *extra)
{
	while (*extra) {
		extra += strspn(extra, " \t");
		if (strncmp(extra, "-m", 2) == 0) {
			return true;
		}
		extra += strcspn(extra, " \t");
	}
	return false;
}

int
moe_hugepages_prepare(struct spdk_app_opts *opts, char **env_context, char **hugedir)
{
	const char *extra = opts->env_context ? opts->env_context : "";
	struct mntent *entry;
	FILE *mounts;
	size_t length;

	*env_context = NULL;
	*hugedir = NULL;
	if (opts->no_huge || strstr(extra, "--no-huge")) {
		fprintf(stderr, "MoE memory mode=ordinary (explicit --no-huge), pool_mib=%d\n",
			opts->mem_size);
		return 0;
	}
	if (opts->mem_size <= 0 || opts->mem_size % 2 != 0) {
		fprintf(stderr, "MoE 2 MiB hugepages require a positive, even -s memory size\n");
		return -EINVAL;
	}
	if (overrides_memory(extra) || strstr(extra, "--socket-mem") || strstr(extra, "--in-memory") ||
	    strstr(extra, "--huge-dir")) {
		fprintf(stderr, "MoE fixed hugepage pool conflicts with env-context -m, "
			"--socket-mem, --in-memory or --huge-dir; use -s and --huge-dir directly\n");
		return -EINVAL;
	}
	if (opts->hugedir) {
		if (!usable_mount(opts->hugedir)) {
			fprintf(stderr, "MoE --huge-dir must be a writable 2 MiB hugetlbfs mount: %s\n",
				opts->hugedir);
			return -EINVAL;
		}
	} else {
		mounts = setmntent("/proc/self/mounts", "r");
		if (!mounts) {
			return -errno;
		}
		while ((entry = getmntent(mounts)) != NULL) {
			if (strcmp(entry->mnt_type, "hugetlbfs") == 0 && usable_mount(entry->mnt_dir)) {
				*hugedir = strdup(entry->mnt_dir);
				endmntent(mounts);
				if (!*hugedir) {
					return -ENOMEM;
				}
				opts->hugedir = *hugedir;
				goto selected;
			}
		}
		endmntent(mounts);
		fprintf(stderr, "MoE needs a writable 2 MiB hugetlbfs mount and %d free pages; "
			"see docs/moe/reversible_weights.md. Local debugging can use --no-huge.\n",
			opts->mem_size / 2);
		return -ENOENT;
	}
selected:
	length = strlen(extra) + sizeof(" --legacy-mem");
	*env_context = malloc(length);
	if (!*env_context) {
		return -ENOMEM;
	}
	snprintf(*env_context, length, "%s --legacy-mem", extra);
	if (setenv("MOE_FIXED_HUGEPAGES", "1", 1) != 0) {
		return -errno;
	}
	opts->env_context = *env_context;
	fprintf(stderr, "MoE memory mode=hugetlb, page_kib=2048, pool_mib=%d, pages=%d, "
		"hugedir=%s; SPDK/DPDK preallocates and faults pages before bdev warmup\n",
		opts->mem_size, opts->mem_size / 2, opts->hugedir);
	return 0;
}
