#include "tinyinfer/matmul.hpp"

#include <cstddef>
#include <stdexcept>

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

inline float dot_fp32_row(const float* a, const float* b, std::size_t k) noexcept {
    std::size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= k; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= k; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float total = hsum256_ps(acc0);
#else
    float total = 0.0f;
#endif
    for (; i < k; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

inline float dot_q8_fp32_row(
    const float* x,
    const int8_t* q_row,
    const float* s_row,
    std::size_t cols,
    std::size_t blocks_per_row
) noexcept {
    const std::size_t full_blocks = cols / kQ8BlockSize;
    float total = 0.0f;

#if defined(__AVX2__) && defined(__FMA__)
    for (std::size_t b = 0; b < full_blocks; ++b) {
        const float* xb = x + b * kQ8BlockSize;
        const int8_t* qb = q_row + b * kQ8BlockSize;

        const __m256 q0 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb)))
        );
        const __m256 q1 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb + 8)))
        );
        const __m256 q2 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb + 16)))
        );
        const __m256 q3 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb + 24)))
        );

        __m256 acc0 = _mm256_mul_ps(q0, _mm256_loadu_ps(xb));
        __m256 acc1 = _mm256_mul_ps(q1, _mm256_loadu_ps(xb + 8));
        acc0 = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xb + 16), acc0);
        acc1 = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xb + 24), acc1);

        total += hsum256_ps(_mm256_add_ps(acc0, acc1)) * s_row[b];
    }
#else
    for (std::size_t b = 0; b < full_blocks; ++b) {
        const float* xb = x + b * kQ8BlockSize;
        const int8_t* qb = q_row + b * kQ8BlockSize;
        float block_sum = 0.0f;
        for (std::size_t i = 0; i < kQ8BlockSize; ++i) {
            block_sum += static_cast<float>(qb[i]) * xb[i];
        }
        total += block_sum * s_row[b];
    }
#endif

    // Safe scalar tail when cols is not a multiple of kQ8BlockSize.
    if (full_blocks < blocks_per_row) {
        const std::size_t col_start = full_blocks * kQ8BlockSize;
        const std::size_t tail = cols - col_start;
        const float* xb = x + col_start;
        const int8_t* qb = q_row + col_start;
        float block_sum = 0.0f;
        for (std::size_t i = 0; i < tail; ++i) {
            block_sum += static_cast<float>(qb[i]) * xb[i];
        }
        total += block_sum * s_row[full_blocks];
    }

    return total;
}

}  // namespace

void matmul_fp32_nt(
    const float* x,
    const float* w,
    float* y,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (x == nullptr || w == nullptr || y == nullptr) {
        throw std::invalid_argument("matmul_fp32_nt: null tensor pointer.");
    }

    const auto total_out = static_cast<std::ptrdiff_t>(m * n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m * n * k >= 65536)
#endif
    for (std::ptrdiff_t idx = 0; idx < total_out; ++idx) {
        const auto flat = static_cast<std::size_t>(idx);
        const std::size_t row_m = flat / n;
        const std::size_t row_n = flat % n;
        y[flat] = dot_fp32_row(x + row_m * k, w + row_n * k, k);
    }
}

void matmul_q8_0_nt_raw(
    const float* x,
    const int8_t* q_weights,
    const float* q_scales,
    float* y,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t blocks_per_row
) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (x == nullptr || q_weights == nullptr || q_scales == nullptr || y == nullptr) {
        throw std::invalid_argument("matmul_q8_0_nt_raw: null tensor pointer.");
    }

    const std::size_t q_stride = blocks_per_row * kQ8BlockSize;
    const auto total_out = static_cast<std::ptrdiff_t>(m * n);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m * n * k >= 65536)
#endif
    for (std::ptrdiff_t idx = 0; idx < total_out; ++idx) {
        const auto flat = static_cast<std::size_t>(idx);
        const std::size_t row_m = flat / n;
        const std::size_t row_n = flat % n;
        y[flat] = dot_q8_fp32_row(
            x + row_m * k,
            q_weights + row_n * q_stride,
            q_scales + row_n * blocks_per_row,
            k,
            blocks_per_row
        );
    }
}

void matmul_q8_0_nt(
    const float* x,
    const Q8Matrix& w_q8,
    float* y,
    std::size_t m
) {
    matmul_q8_0_nt_raw(
        x,
        w_q8.weights.data(),
        w_q8.scales.data(),
        y,
        m,
        w_q8.rows,
        w_q8.cols,
        w_q8.blocks_per_row
    );
}

}  // namespace tinyinfer
