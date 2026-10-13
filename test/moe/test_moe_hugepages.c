// SPDX-License-Identifier: BSD-3-Clause
#include "spdk/stdinc.h"
#include "spdk/util.h"

#include <mntent.h>
#include <sys/vfs.h>
#include <linux/magic.h>

static unsigned mount_index;
static bool deny_access;
static struct mntent mounts[] = {
	{.mnt_dir = "/ordinary", .mnt_type = "tmpfs"},
	{.mnt_dir = "/huge1g", .mnt_type = "hugetlbfs"},
	{.mnt_dir = "/huge2m", .mnt_type = "hugetlbfs"},
};

static int
fake_statfs(const char *dir, struct statfs *fs)
{
	memset(fs, 0, sizeof(*fs));
	fs->f_type = strcmp(dir, "/ordinary") ? HUGETLBFS_MAGIC : TMPFS_MAGIC;
	fs->f_bsize = strcmp(dir, "/huge1g") ? 2097152 : 1073741824;
	return 0;
}

static int
fake_access(const char *dir, int mode)
{
	assert(mode == (R_OK | W_OK | X_OK));
	return deny_access ? -1 : 0;
}

static FILE *
fake_setmntent(const char *path, const char *mode)
{
	assert(strcmp(path, "/proc/self/mounts") == 0 && strcmp(mode, "r") == 0);
	mount_index = 0;
	return stdin;
}

static struct mntent *
fake_getmntent(FILE *stream)
{
	assert(stream == stdin);
	return mount_index < SPDK_COUNTOF(mounts) ? &mounts[mount_index++] : NULL;
}

static int
fake_endmntent(FILE *stream)
{
	assert(stream == stdin);
	return 1;
}

#define statfs(...) fake_statfs(__VA_ARGS__)
#define access fake_access
#define setmntent fake_setmntent
#define getmntent fake_getmntent
#define endmntent fake_endmntent
#include "../../examples/moe/moe_hugepages.c"

int
main(void)
{
	struct spdk_app_opts opts = {.mem_size = MOE_DEFAULT_MEM_MIB};
	char *extra, *dir;

	opts.env_context = "--no-telemetry";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == 0);
	assert(strcmp(opts.hugedir, "/huge2m") == 0);
	assert(strcmp(opts.env_context, "--no-telemetry --legacy-mem") == 0);
	assert(strcmp(getenv("MOE_FIXED_HUGEPAGES"), "1") == 0);
	free(extra);
	free(dir);

	opts = (struct spdk_app_opts) {.mem_size = 512, .hugedir = "/huge2m"};
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == 0 && dir == NULL);
	free(extra);
	opts.hugedir = "/huge1g";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.hugedir = "/ordinary";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.hugedir = NULL;
	deny_access = true;
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -ENOENT);
	deny_access = false;
	opts.mem_size = 513;
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.mem_size = 0;
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.mem_size = MOE_DEFAULT_MEM_MIB;
	opts.env_context = "--socket-mem=1664";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.env_context = "-m4096";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.env_context = "--no-telemetry\t-m 4096";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.env_context = "--in-memory";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.env_context = "--huge-dir=/huge1g";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == -EINVAL);
	opts.env_context = "--no-huge";
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == 0 && !extra && !dir);
	opts.env_context = NULL;
	opts.no_huge = true;
	assert(moe_hugepages_prepare(&opts, &extra, &dir) == 0 && !extra && !dir);
	return 0;
}
