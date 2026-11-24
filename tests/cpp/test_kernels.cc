#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "tinyinfer/matmul.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/quant.hpp"

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

void test_matmul_fp32_and_q8_non_multiple_tail() {
    constexpr std::size_t m = 5;
    constexpr std::size_t n = 19;
    constexpr std::size_t k = 77;  // Non-multiple of 32 tests scalar tail safety.

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(m * k);
    std::vector<float> w(n * k);
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);

    std::vector<float> ref(m * n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<double>(x[i * k + p]) * static_cast<double>(w[j * k + p]);
            }
            ref[i * n + j] = static_cast<float>(acc);
        }
    }

    std::vector<float> y_fp32(m * n, 0.0f);
    tinyinfer::matmul_fp32_nt(x.data(), w.data(), y_fp32.data(), m, n, k);

    float max_fp32_err = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        max_fp32_err = std::max(max_fp32_err, std::fabs(ref[i] - y_fp32[i]));
    }
    require(max_fp32_err < 1e-4f, "FP32 matmul drift exceeded 1e-4");

    const auto w_q8 = tinyinfer::quantize_q8_0(w.data(), n, k);
    require(w_q8.compression_ratio() > 2.8, "Q8_0 compression ratio lower than expected");

    std::vector<float> y_q8(m * n, 0.0f);
    tinyinfer::matmul_q8_0_nt(x.data(), w_q8, y_q8.data(), m);

    const double cos_sim = cosine_similarity(ref, y_q8);
    require(cos_sim > 0.999, "Q8_0 matmul cosine similarity dropped below 0.999");
}

void test_transformer_ops() {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);

    tinyinfer::rmsnorm(out.data(), x.data(), w.data(), x.size(), 1e-5f);
    // rms = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5)
    const float expected0 = 1.0f / std::sqrt(7.5f + 1e-5f);
    require(std::fabs(out[0] - expected0) < 1e-5f, "RMSNorm value mismatch");

    tinyinfer::softmax(out.data(), x.data(), x.size());
    float prob_sum = 0.0f;
    for (float p : out) prob_sum += p;
    require(std::fabs(prob_sum - 1.0f) < 1e-5f, "Softmax probabilities do not sum to 1");

    tinyinfer::silu(out.data(), x.data(), x.size());
    const float expected_silu = 2.0f / (1.0f + std::exp(-2.0f));
    require(std::fabs(out[1] - expected_silu) < 1e-5f, "SiLU value mismatch");

    std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
    tinyinfer::rope_inplace(q.data(), 1, 4, 3, 10000.0f);
    const float norm_after = std::sqrt(q[0] * q[0] + q[1] * q[1]);
    require(std::fabs(norm_after - 1.0f) < 1e-5f, "RoPE failed to preserve vector norm");
}

}  // namespace

int main() {
    const auto info = tinyinfer::get_build_info();
    std::cout << "BuildInfo: compiler=" << info.compiler
              << " avx2_fma=" << (info.avx2_fma ? "ON" : "OFF")
              << " openmp=" << (info.openmp ? "ON" : "OFF")
              << " threads=" << info.max_threads << "\n";

    test_matmul_fp32_and_q8_non_multiple_tail();
    test_transformer_ops();
    std::cout << "[PASS] All C++20 kernel tests passed.\n";
    return 0;
}
