#include <cuda.h>

// Update the include path to the correct location
#include "native.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <iostream>
#include "common.h"
#include "compare.hpp"
#include "Timer.cuh"

// ============ 小工具：随机填充、计时 ============

static void fill_uniform(std::vector<float>& buf, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : buf) x = dist(rng);
}

struct WallTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void tic() { t0 = clock::now(); }
    double toc_ms() const {
        auto t1 = clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    }
};

// ============ 你需要接入的两个内核 ============
// 说明：下面两个函数是“占位符”，请在 TODO 区域替换为你实际的内核调用。
// 若你的内核是 CUDA kernel launch / OpenCL / 其它库调用，也在此处完成。
// 要点：在相同输入下，分别生成输出矩阵 C1 和 C2（大小 m x n，行优先存储）。

// 你可以按需调整 block 尺寸
static inline dim3 pickBlock() { return dim3(16, 16, 1); }

static inline dim3 pickGrid(size_t M, size_t N, dim3 block) {
    return dim3( (unsigned)((N + block.x - 1) / block.x),
                 (unsigned)((M + block.y - 1) / block.y),
                 1 );
}

float run_kernel_A(size_t m, size_t n,
                  const float* A_h, const float* B_h, size_t k,
                  float* C_out)
{
    const size_t M = m, N = n, K = k;
    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CHECK(cudaMalloc(&A_d, sizeA));
    CHECK(cudaMalloc(&B_d, sizeB));
    CHECK(cudaMalloc(&C_d, sizeC));

    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    // 若内核以累加形式写 C，可先清零
    CHECK(cudaMemset(C_d, 0, sizeC));

    dim3 block = pickBlock();
    dim3 grid  = pickGrid(M, N, block);
	// --------- 核函数计时 ---------
    startTimer();
    native_matmul<<<grid, block>>>(static_cast<int>(M),
                                   static_cast<int>(N),
                                   static_cast<int>(K),
                                   A_d, B_d, C_d);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
	float kernel_ms = stopTimer();
    // --------- 计时结束 ---------

    CHECK(cudaMemcpy(C_out, C_d, sizeC, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(C_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(A_d));

	return kernel_ms;  // 返回纯核函数耗时
}

float run_kernel_B(size_t m, size_t n,
                  const float* A_h, const float* B_h, size_t k,
                  float* C_out)
{
    const size_t M = m, N = n, K = k;
    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CHECK(cudaMalloc(&A_d, sizeA));
    CHECK(cudaMalloc(&B_d, sizeB));
    CHECK(cudaMalloc(&C_d, sizeC));

    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(C_d, 0, sizeC)); // 视内核实现选择是否需要

    dim3 block = pickBlock();
    dim3 grid  = pickGrid(M, N, block);

	// --------- 核函数计时 ---------
    startTimer();
    native_matmul_version2<<<grid, block>>>(static_cast<int>(M),
                                            static_cast<int>(N),
                                            static_cast<int>(K),
                                            A_d, B_d, C_d);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
	float kernel_ms = stopTimer();
    // --------- 计时结束 ---------

    CHECK(cudaMemcpy(C_out, C_d, sizeC, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(C_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(A_d));

	return kernel_ms;  // 返回纯核函数耗时
}

// ============ 单个测试用例 ============
// 支持可选的 k 维度与输入 A(m x k)、B(k x n)。
// 如果你的内核不需要 A/B/k，可忽略这些参数，或在 run_kernel_* 中不使用。

struct TestConfig {
    size_t m{256}, n{256}, k{256};   // 需要就用，不需要就忽略
    float lo{-1.0f}, hi{1.0f};       // 随机输入范围
    uint32_t seed{42};               // 固定随机种子便于复现
    bool warmup{true};
};

bool run_one_case(const TestConfig& cfg,
                  double& e2eA_ms, double& e2eB_ms,   // 端到端时间
                  float& kA_ms,    float& kB_ms)      // 核函数时间
{
    const size_t M = cfg.m, N = cfg.n, K = cfg.k;

    // 如需 A、B 输入，这里准备；如果不需要，可删去
    std::vector<float> A(M * K), B(K * N);
    fill_uniform(A, cfg.lo, cfg.hi, cfg.seed);
    fill_uniform(B, cfg.lo, cfg.hi, cfg.seed + 1);

    std::vector<float> C1(M * N), C2(M * N);

	// 可选热身（避免第一次显存分配、JIT 带来的抖动）
    if (cfg.warmup) {
        run_kernel_A(M, N, A.data(), B.data(), K, C1.data());
        run_kernel_B(M, N, A.data(), B.data(), K, C2.data());
    }

	// A：端到端计时（外层 WallTimer），核函数计时由 run_kernel_A 返回
	WallTimer t;

    t.tic();
    kA_ms = run_kernel_A(M, N, A.data(), B.data(), K, C1.data());
    e2eA_ms = t.toc_ms();

	// B：同理
    t.tic();
    kB_ms = run_kernel_B(M, N, A.data(), B.data(), K, C2.data());
    e2eB_ms = t.toc_ms();

	// 正确性校验
    bool ok = compare(M, N, C1.data(), C2.data());
    return ok;
}

// ============ 批量测试入口 ============

int main(int argc, char** argv)
{
    // 可通过命令行传入：m n k iters
    size_t m = 256, n = 256, k = 256;
    int iters = 10;
    if (argc >= 3) {
        m = std::stoul(argv[1]);
        n = std::stoul(argv[2]);
    }
    if (argc >= 4) {
        k = std::stoul(argv[3]);
    }
    if (argc >= 5) {
        iters = std::stoi(argv[4]);
    }

    std::cout << "Matrix sizes: m=" << m << " n=" << n << " k=" << k
              << " | iters=" << iters << "\n";

    int pass = 0;
    // double sumA = 0.0, sumB = 0.0;
	double sum_e2eA = 0.0, sum_e2eB = 0.0;
    double sum_kA   = 0.0, sum_kB   = 0.0; // 用 double 聚合，打印时无损

    for (int i = 0; i < iters; ++i) {
        TestConfig cfg;
        cfg.m = m; cfg.n = n; cfg.k = k;
        cfg.seed = 1234u + i;      // 每次换种子，覆盖更多输入分布
        cfg.warmup = (i == 0);     // 首次热身, 之后关闭

        // double msA = 0.0, msB = 0.0;
		double e2eA_ms = 0.0, e2eB_ms = 0.0;
        float  kA_ms = 0.0f,  kB_ms = 0.0f;

        bool ok = run_one_case(cfg, e2eA_ms, e2eB_ms, kA_ms, kB_ms);
        if (ok) ++pass;
        // sumA += msA; sumB += msB;
		sum_e2eA += e2eA_ms;  sum_e2eB += e2eB_ms;
        sum_kA   += kA_ms;    sum_kB   += kB_ms;

		std::cout << "[Iter " << i << "] "
                  << (ok ? "OK" : "MISMATCH")
                  << " | e2eA=" << e2eA_ms << " ms"
                  << " | e2eB=" << e2eB_ms << " ms"
                  << " | kA="   << kA_ms    << " ms"
                  << " | kB="   << kB_ms    << " ms"
                  << "\n";
    }

    std::cout << "---- Summary ----\n";
    std::cout << "Passed " << pass << "/" << iters << "\n";
    std::cout << "Avg e2eA = " << (sum_e2eA / iters) << " ms\n";
    std::cout << "Avg e2eB = " << (sum_e2eB / iters) << " ms\n";
    std::cout << "Avg kA   = " << (sum_kA   / iters) << " ms\n";
    std::cout << "Avg kB   = " << (sum_kB   / iters) << " ms\n";

    return (pass == iters) ? 0 : 1;
}