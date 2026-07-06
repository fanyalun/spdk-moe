/*
 * topk.h — Top-k 选择模块
 *
 * 从数组中选出值最大的 k 个元素，返回其索引和值。
 * 使用朴素的全排序方法（构造 (value, index) 对，降序排序，取前 k 个）。
 * 本模块不依赖项目中其他任何模块。
 */

#ifndef TOPK_H
#define TOPK_H

/*
 * 从数组中选出值最大的 k 个元素
 *
 * 参数：
 *   values    — 输入数组，长度 n，只读
 *   n         — 输入数组长度
 *   k         — 要选出的元素个数
 *   indices   — 输出参数，存放选中的 k 个索引，由调用者分配
 *   out_values — 输出参数，存放选中的 k 个值，由调用者分配
 *
 * 特殊情况：
 *   当 k >= n 时，返回全部 n 个元素（按值降序排列）
 *
 * 实现说明：
 *   1. 在堆上构造 n 个 (value, index) 对
 *   2. 使用插入排序按 value 降序排列
 *   3. 取前 k 个的 index 和 value，写入输出参数
 *   4. 释放临时数组
 *
 * 使用插入排序（O(n²)）而非 qsort()，有意保留性能优化空间。
 */
void topk_select(const float* values, int n, int k, int* indices, float* out_values);

#endif /* TOPK_H */
