#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "tinyinfer/attention.hpp"
#include "tinyinfer/kv_cache.hpp"

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

void test_gqa_prefill_matches_step_by_step_decode() {
    constexpr std::size_t n_layers = 2;
    constexpr std::size_t max_seq = 16;
    constexpr std::size_t n_q_heads = 8;
    constexpr std::size_t n_kv_heads = 2;
    constexpr std::size_t head_dim = 36;  // Non-multiple of 16 tests SIMD tail.
    constexpr std::size_t seq_len = 7;

    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 0.7f);

    const std::size_t q_dim = n_q_heads * head_dim;
    const std::size_t kv_dim = n_kv_heads * head_dim;

    std::vector<float> q(seq_len * q_dim);
    std::vector<float> k(seq_len * kv_dim);
    std::vector<float> v(seq_len * kv_dim);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);

    tinyinfer::KVCache cache_prefill(n_layers, max_seq, n_kv_heads, head_dim);
    std::vector<float> out_prefill(seq_len * q_dim, 0.0f);
    tinyinfer::gqa_prefill(
        q.data(), k.data(), v.data(), cache_prefill, 1, 0, seq_len, n_q_heads, out_prefill.data()
    );
    require(cache_prefill.seq_len(1) == seq_len, "Prefill did not update layer seq_len");

    tinyinfer::KVCache cache_decode(n_layers, max_seq, n_kv_heads, head_dim);
    std::vector<float> out_decode(seq_len * q_dim, 0.0f);
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        tinyinfer::gqa_decode_step(
            q.data() + pos * q_dim,
            k.data() + pos * kv_dim,
            v.data() + pos * kv_dim,
            cache_decode,
            1,
            pos,
            n_q_heads,
            out_decode.data() + pos * q_dim
        );
    }

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < out_prefill.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(out_prefill[i] - out_decode[i]));
    }
    require(max_diff < 1e-6f, "Prefill and step-by-step decode outputs diverged");

    cache_decode.reset();
    require(cache_decode.seq_len(1) == 0, "KVCache::reset did not clear seq_len");
}

void test_gqa_matches_exact_scalar_reference() {
    constexpr std::size_t n_q_heads = 4;
    constexpr std::size_t n_kv_heads = 2;
    constexpr std::size_t head_dim = 8;
    constexpr std::size_t seq_len = 5;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const std::size_t q_dim = n_q_heads * head_dim;
    const std::size_t kv_dim = n_kv_heads * head_dim;
    std::vector<float> q(seq_len * q_dim);
    std::vector<float> k(seq_len * kv_dim);
    std::vector<float> v(seq_len * kv_dim);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);

    tinyinfer::KVCache cache(1, 16, n_kv_heads, head_dim);
    std::vector<float> out(seq_len * q_dim, 0.0f);
    tinyinfer::gqa_prefill(
        q.data(), k.data(), v.data(), cache, 0, 0, seq_len, n_q_heads, out.data()
    );

    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const std::size_t group = n_q_heads / n_kv_heads;
    double max_err = 0.0;

    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        for (std::size_t h = 0; h < n_q_heads; ++h) {
            const std::size_t kvh = h / group;
            std::vector<double> scores(pos + 1, 0.0);
            double max_s = -1e30;
            for (std::size_t t = 0; t <= pos; ++t) {
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q[pos * q_dim + h * head_dim + d]) *
                           static_cast<double>(k[t * kv_dim + kvh * head_dim + d]);
                }
                scores[t] = dot * inv_sqrt_d;
                if (scores[t] > max_s) max_s = scores[t];
            }
            double sum = 0.0;
            for (double& s : scores) {
                s = std::exp(s - max_s);
                sum += s;
            }
            for (std::size_t d = 0; d < head_dim; ++d) {
                double expected = 0.0;
                for (std::size_t t = 0; t <= pos; ++t) {
                    expected += (scores[t] / sum) *
                                static_cast<double>(v[t * kv_dim + kvh * head_dim + d]);
                }
                const double actual = static_cast<double>(out[pos * q_dim + h * head_dim + d]);
                max_err = std::max(max_err, std::fabs(actual - expected));
            }
        }
    }
    require(max_err < 1e-5, "GQA output drifted from FP64 scalar reference");
}

}  // namespace

int main() {
    test_gqa_prefill_matches_step_by_step_decode();
    test_gqa_matches_exact_scalar_reference();
    std::cout << "[PASS] All C++20 KV-Cache and GQA attention tests passed.\n";
    return 0;
}
