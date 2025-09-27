#pragma once
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

// 我需要对两个结果矩阵做元素级误差或整体误差检查

// 元素级近似比较
static inline bool almost_equalf(float x, float y, float abs_tol, float rel_tol) {
    if (x == y) return true;                 // 含 +0/-0
    float ax = fabsf(x), ay = fabsf(y);
    float diff = fabsf(x - y);

    // 处理 NaN/Inf：只要有一个非有限就按不相等（也可按需特殊处理）
    if (!isfinite(x) || !isfinite(y)) return false;

    float tol = fmaxf(abs_tol, rel_tol * fmaxf(ax, ay));
    return diff <= tol;
}

// 元素级全矩阵比较：逐元素都通过则返回 true
bool compare_mats_elementwise(const float *C1, const float *C2,
                              size_t rows, size_t cols,
                              float abs_tol, float rel_tol,
                              size_t *out_bad_i, size_t *out_bad_j,
                              float *out_diff) {
    size_t n = rows * cols;
    for (size_t idx = 0; idx < n; ++idx) {
        if (!almost_equalf(C1[idx], C2[idx], abs_tol, rel_tol)) {
            if (out_bad_i) *out_bad_i = idx / cols;
            if (out_bad_j) *out_bad_j = idx % cols;
            if (out_diff)  *out_diff  = fabsf(C1[idx] - C2[idx]);
            return false;
        }
    }
    return true;
}


// 计算整体误差：最大绝对差、Frobenius 归一化误差
void mat_diff_metrics(const float *C1, const float *C2,
                      size_t rows, size_t cols,
                      float *out_max_abs_diff,
                      float *out_rel_frob_err) {
    size_t n = rows * cols;
    double sum_sq_diff = 0.0, sum_sq_c1 = 0.0, sum_sq_c2 = 0.0;
    float max_abs_diff = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float d = C1[i] - C2[i];
        float ad = fabsf(d);
        if (ad > max_abs_diff) max_abs_diff = ad;
        sum_sq_diff += (double)d * (double)d;
        sum_sq_c1   += (double)C1[i] * (double)C1[i];
        sum_sq_c2   += (double)C2[i] * (double)C2[i];
    }

    double frob_diff = sqrt(sum_sq_diff);
    double scale = fmax(1.0, fmax(sqrt(sum_sq_c1), sqrt(sum_sq_c2)));
    if (out_max_abs_diff) *out_max_abs_diff = max_abs_diff;
    if (out_rel_frob_err) *out_rel_frob_err = (float)(frob_diff / scale);
}



bool compare(size_t m, size_t n, const float *C1, const float *C2) {
	float abs_tol = 1e-6f;
	float rel_tol = 1e-5f;

	size_t bi, bj; float bd;
	bool ok = compare_mats_elementwise(C1, C2, m, n, abs_tol, rel_tol, &bi, &bj, &bd);

	float max_abs, rel_frob;
	mat_diff_metrics(C1, C2, m, n, &max_abs, &rel_frob);

	if(ok) {
		printf("元素级全部通过\n");
        return true;
	}
	else {
		// 第一个不通过的位置 (bi, bj)，差值 bd
		printf("第一个不通过的位置 (%zu, %zu)，差值 %0.5f\n", bi, bj, bd);
        return false;
	}
}