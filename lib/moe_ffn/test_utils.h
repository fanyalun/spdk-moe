/*
 * test_utils.h — 测试工具模块
 *
 * 提供向量打印和统计摘要功能，用于测试程序的输出展示。
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

/*
 * 打印向量内容
 *
 * 格式化输出向量，支持指定打印前 N 个和后 N 个元素。
 * 当向量总长度 <= head + tail 时，打印全部元素。
 * 当向量总长度 >  head + tail 时，中间用 "..." 省略。
 *
 * 参数：
 *   vec   — 向量数据指针
 *   len   — 向量总长度
 *   head  — 打印前 head 个元素
 *   tail  — 打印后 tail 个元素
 *   name  — 向量名称（用于输出标识）
 */
void print_vector(const float* vec, int len, int head, int tail, const char* name);

/*
 * 计算并打印向量统计摘要
 *
 * 计算向量的最小值、最大值和均值，并以格式化方式打印。
 *
 * 参数：
 *   vec  — 向量数据指针
 *   len  — 向量长度
 *   name — 向量名称（用于输出标识）
 */
void print_stats(const float* vec, int len, const char* name);

#endif /* TEST_UTILS_H */
