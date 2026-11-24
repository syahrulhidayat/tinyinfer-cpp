#include "tinyinfer/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tinyinfer {

namespace {

#if defined(__AVX2__) && defined(__FMA__)
inline float hsum256_ps(__m256 v) noexcept {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    const __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

inline float dot_head(const float* a, const float* b, std::size_t dim) noexcept {
    std::size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= dim; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float total = hsum256_ps(acc0);
#else
    float total = 0.0f;
#endif
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

inline void axpy_head(float* acc, const float* v, float weight, std::size_t dim) noexcept {
    std::size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 wv = _mm256_set1_ps(weight);
    for (; i + 16 <= dim; i += 16) {
        __m256 y0 = _mm256_loadu_ps(acc + i);
        __m256 y1 = _mm256_loadu_ps(acc + i + 8);
        y0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + i), y0);
        y1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + i + 8), y1);
        _mm256_storeu_ps(acc + i, y0);
        _mm256_storeu_ps(acc + i + 8, y1);
    }
    for (; i + 8 <= dim; i += 8) {
        __m256 y0 = _mm256_loadu_ps(acc + i);
        y0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + i), y0);
        _mm256_storeu_ps(acc + i, y0);
    }
#endif
    for (; i < dim; ++i) {
        acc[i] += weight * v[i];
    }
}

void attend_single_position(
    const float* q,
    const KVCache& cache,
    std::size_t layer,
    std::size_t pos,
    std::size_t n_q_heads,
    float* out
) {
    const std::size_t n_kv_heads = cache.n_kv_heads();
    const std::size_t head_dim = cache.head_dim();
    const std::size_t kv_group_size = n_q_heads / n_kv_heads;
    const std::size_t active_len = pos + 1;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const auto total_heads = static_cast<std::ptrdiff_t>(n_q_heads);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_q_heads * active_len >= 512)
#endif
    for (std::ptrdiff_t h_idx = 0; h_idx < total_heads; ++h_idx) {
        // Thread-local score buffer stays allocated across decode steps (zero hot-path allocs).
        thread_local std::vector<float> scores;
        if (scores.size() < active_len) {
            scores.resize(active_len);
        }

        const auto q_head = static_cast<std::size_t>(h_idx);
        const std::size_t kv_head = q_head / kv_group_size;
        const float* q_ptr = q + q_head * head_dim;
        float* o_ptr = out + q_head * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < active_len; ++t) {
            const float* k_ptr = cache.key_ptr(layer, t, kv_head);
            const float s = dot_head(q_ptr, k_ptr, head_dim) * inv_sqrt_d;
            scores[t] = s;
            if (s > max_score) {
                max_score = s;
            }
        }

        double denom = 0.0;
        for (std::size_t t = 0; t < active_len; ++t) {
            const float p = std::exp(scores[t] - max_score);
            scores[t] = p;
            denom += static_cast<double>(p);
        }
        const float inv_denom = static_cast<float>(1.0 / denom);

        std::fill(o_ptr, o_ptr + head_dim, 0.0f);
        for (std::size_t t = 0; t < active_len; ++t) {
            const float prob = scores[t] * inv_denom;
            const float* v_ptr = cache.value_ptr(layer, t, kv_head);
            axpy_head(o_ptr, v_ptr, prob, head_dim);
        }
    }
}

void validate_gqa_args(const KVCache& cache, std::size_t layer, std::size_t n_q_heads) {
    if (layer >= cache.n_layers()) {
        throw std::out_of_range("GQA: layer index out of range.");
    }
    if (n_q_heads == 0 || n_q_heads % cache.n_kv_heads() != 0) {
        throw std::invalid_argument("GQA: n_q_heads must be a positive multiple of n_kv_heads.");
    }
}

}  // namespace

void gqa_decode_step(
    const float* q,
    const float* k_new,
    const float* v_new,
    KVCache& cache,
    std::size_t layer,
    std::size_t pos,
    std::size_t n_q_heads,
    float* out
) {
    validate_gqa_args(cache, layer, n_q_heads);
    if (q == nullptr || k_new == nullptr || v_new == nullptr || out == nullptr) {
        throw std::invalid_argument("gqa_decode_step: null tensor pointer.");
    }

    cache.update(layer, pos, k_new, v_new);
    attend_single_position(q, cache, layer, pos, n_q_heads, out);
}

void gqa_prefill(
    const float* q,
    const float* k,
    const float* v,
    KVCache& cache,
    std::size_t layer,
    std::size_t start_pos,
    std::size_t seq_len,
    std::size_t n_q_heads,
    float* out
) {
    validate_gqa_args(cache, layer, n_q_heads);
    if (seq_len == 0) {
        return;
    }
    if (start_pos + seq_len > cache.max_seq_len()) {
        throw std::out_of_range("gqa_prefill: start_pos + seq_len exceeds cache max_seq_len.");
    }
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        throw std::invalid_argument("gqa_prefill: null tensor pointer.");
    }

    const std::size_t q_dim = n_q_heads * cache.head_dim();
    const std::size_t kv_dim = cache.kv_dim();

    for (std::size_t s = 0; s < seq_len; ++s) {
        cache.update(layer, start_pos + s, k + s * kv_dim, v + s * kv_dim);
    }

    for (std::size_t s = 0; s < seq_len; ++s) {
        attend_single_position(q + s * q_dim, cache, layer, start_pos + s, n_q_heads, out + s * q_dim);
    }
}

}  // namespace tinyinfer
