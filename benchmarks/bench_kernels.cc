#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "tinyinfer/matmul.hpp"
#include "tinyinfer/quant.hpp"

int main() {
    constexpr std::size_t m = 1;
    constexpr std::size_t n = 2048;
    constexpr std::size_t k = 2048;
    constexpr int warmup = 5;
    constexpr int iters = 40;

    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> x(m * k);
    std::vector<float> w(n * k);
    std::vector<float> y(m * n, 0.0f);
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);

    const auto w_q8 = tinyinfer::quantize_q8_0(w.data(), n, k);
    const auto info = tinyinfer::get_build_info();

    for (int i = 0; i < warmup; ++i) {
        tinyinfer::matmul_fp32_nt(x.data(), w.data(), y.data(), m, n, k);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        tinyinfer::matmul_fp32_nt(x.data(), w.data(), y.data(), m, n, k);
    }
    const auto t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < warmup; ++i) {
        tinyinfer::matmul_q8_0_nt(x.data(), w_q8, y.data(), m);
    }
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        tinyinfer::matmul_q8_0_nt(x.data(), w_q8, y.data(), m);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double q8_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    const double flops = 2.0 * static_cast<double>(m) * n * k;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Build: AVX2+FMA=" << (info.avx2_fma ? "ON" : "OFF")
              << " OpenMP=" << (info.openmp ? "ON" : "OFF")
              << " Threads=" << info.max_threads << "\n";
    std::cout << "GEMV [1x2048 @ 2048x2048^T] FP32 : " << fp32_ms << " ms ("
              << (flops / (fp32_ms * 1e6)) << " GFLOP/s, "
              << (w.size() * sizeof(float) / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "GEMV [1x2048 @ 2048x2048^T] Q8_0 : " << q8_ms << " ms ("
              << (flops / (q8_ms * 1e6)) << " GFLOP/s, "
              << (w_q8.memory_bytes() / (1024.0 * 1024.0)) << " MB, "
              << w_q8.compression_ratio() << "x compression)\n";
    return 0;
}
