/* SPDX-License-Identifier: BSD-3-Clause */
#include "moe_affinity.h"

static cpu_set_t g_allowed;
static bool g_initialized;
static int g_initiator;

static int
topology(int cpu, const char *name)
{
	char path[160];
	int value = -1;
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, name);
	FILE *file = fopen(path, "r");
	if (file) {
		if (fscanf(file, "%d", &value) != 1) {
			value = -1;
		}
		fclose(file);
	}
	return value;
}

static bool
same_core(int a, int b)
{
	int package = topology(a, "physical_package_id");
	int core = topology(a, "core_id");
	return a == b || (package >= 0 && core >= 0 &&
			 package == topology(b, "physical_package_id") &&
			 core == topology(b, "core_id"));
}

static int
node(int cpu)
{
	char path[128];
	int value = -1;
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
	DIR *dir = opendir(path);
	if (dir) {
		struct dirent *entry;
		while ((entry = readdir(dir))) {
			if (!strncmp(entry->d_name, "node", 4)) {
				char *end;
				long n = strtol(entry->d_name + 4, &end, 10);
				if (end != entry->d_name + 4 && !*end && n >= 0 && n <= INT_MAX) {
					value = n;
					break;
				}
			}
		}
		closedir(dir);
	}
	return value;
}

int
moe_affinity_init(void)
{
	const char *text = getenv("MOE_INITIATOR_CPU");
	g_initialized = false;
	g_initiator = 0;
	if (text) {
		char *end;
		errno = 0;
		long cpu = strtol(text, &end, 10);
		if (errno || end == text || *end || cpu < 0 || cpu >= CPU_SETSIZE) {
			return -EINVAL;
		}
		g_initiator = cpu;
	}
	if (sched_getaffinity(0, sizeof(g_allowed), &g_allowed)) {
		return -errno;
	}
	g_initialized = true;
	return 0;
}

bool
moe_affinity_allowed(int cpu)
{
	return cpu >= 0 && cpu < CPU_SETSIZE && (!g_initialized || CPU_ISSET(cpu, &g_allowed));
}

bool
moe_affinity_reserved(int cpu)
{
	return same_core(cpu, g_initiator);
}

int
moe_affinity_default_reactor(void)
{
	for (int worker = 0; worker < CPU_SETSIZE; worker++) {
		if (!moe_affinity_allowed(worker) || moe_affinity_reserved(worker)) {
			continue;
		}
		if (node(worker) < 0 || topology(worker, "physical_package_id") < 0 ||
		    topology(worker, "core_id") < 0) {
			continue;
		}
		for (int reactor = worker + 1; reactor < CPU_SETSIZE; reactor++) {
			if (moe_affinity_allowed(reactor) && !moe_affinity_reserved(reactor) &&
			    !same_core(worker, reactor) && node(worker) == node(reactor) &&
			    topology(worker, "physical_package_id") == topology(reactor, "physical_package_id")) {
				return reactor;
			}
		}
	}
	return -ENOSPC;
}
