#include "tinyinfer/quant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tinyinfer {

BuildInfo get_build_info() noexcept {
    BuildInfo info;
#if defined(__AVX2__) && defined(__FMA__)
    info.avx2_fma = true;
#else
    info.avx2_fma = false;
#endif
#ifdef _OPENMP
    info.openmp = true;
    info.max_threads = omp_get_max_threads();
#else
    info.openmp = false;
    info.max_threads = 1;
#endif
#if defined(__clang__)
    info.compiler = "Clang";
#elif defined(__GNUC__)
    info.compiler = "GCC";
#else
    info.compiler = "Unknown";
#endif
    return info;
}

Q8Matrix quantize_q8_0(const float* data, std::size_t rows, std::size_t cols) {
    if (data == nullptr && rows > 0 && cols > 0) {
        throw std::invalid_argument("quantize_q8_0: null input pointer.");
    }

    Q8Matrix out;
    out.rows = rows;
    out.cols = cols;
    out.blocks_per_row = (cols + kQ8BlockSize - 1) / kQ8BlockSize;
    out.weights.assign(rows * out.blocks_per_row * kQ8BlockSize, 0);
    out.scales.assign(rows * out.blocks_per_row, 0.0f);

    const auto total_rows = static_cast<std::ptrdiff_t>(rows);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows >= 64)
#endif
    for (std::ptrdiff_t r_idx = 0; r_idx < total_rows; ++r_idx) {
        const auto r = static_cast<std::size_t>(r_idx);
        const float* src_row = data + r * cols;
        int8_t* dst_row = out.weights.data() + r * out.blocks_per_row * kQ8BlockSize;
        float* scale_row = out.scales.data() + r * out.blocks_per_row;

        for (std::size_t b = 0; b < out.blocks_per_row; ++b) {
            const std::size_t col_start = b * kQ8BlockSize;
            const std::size_t valid = std::min(kQ8BlockSize, cols - col_start);

            float max_abs = 0.0f;
            for (std::size_t i = 0; i < valid; ++i) {
                const float v = std::fabs(src_row[col_start + i]);
                if (v > max_abs) {
                    max_abs = v;
                }
            }

            const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 0.0f;
            const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            scale_row[b] = scale;

            int8_t* q_block = dst_row + b * kQ8BlockSize;
            for (std::size_t i = 0; i < valid; ++i) {
                const long rounded = std::lround(src_row[col_start + i] * inv_scale);
                const long clamped = std::clamp(rounded, -127L, 127L);
                q_block[i] = static_cast<int8_t>(clamped);
            }
        }
    }

    return out;
}

std::vector<float> dequantize_q8_0(const Q8Matrix& qmat) {
    std::vector<float> out(qmat.rows * qmat.cols, 0.0f);
    for (std::size_t r = 0; r < qmat.rows; ++r) {
        const int8_t* q_row = qmat.weights.data() + r * qmat.blocks_per_row * kQ8BlockSize;
        const float* s_row = qmat.scales.data() + r * qmat.blocks_per_row;
        float* dst_row = out.data() + r * qmat.cols;

        for (std::size_t b = 0; b < qmat.blocks_per_row; ++b) {
            const std::size_t col_start = b * kQ8BlockSize;
            const std::size_t valid = std::min(kQ8BlockSize, qmat.cols - col_start);
            const float scale = s_row[b];
            const int8_t* q_block = q_row + b * kQ8BlockSize;
            for (std::size_t i = 0; i < valid; ++i) {
                dst_row[col_start + i] = static_cast<float>(q_block[i]) * scale;
            }
        }
    }
    return out;
}

}  // namespace tinyinfer
