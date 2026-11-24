#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinyinfer {

inline constexpr std::size_t kQ8BlockSize = 32;

struct Q8Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t blocks_per_row = 0;
    // Contiguous row-major quantized weights: rows * blocks_per_row * kQ8BlockSize
    std::vector<int8_t> weights;
    // Per-block symmetric scale factors: rows * blocks_per_row
    std::vector<float> scales;

    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return weights.size() * sizeof(int8_t) + scales.size() * sizeof(float);
    }

    [[nodiscard]] double compression_ratio() const noexcept {
        const double fp32_bytes = static_cast<double>(rows * cols * sizeof(float));
        const double q8_bytes = static_cast<double>(memory_bytes());
        return q8_bytes > 0.0 ? fp32_bytes / q8_bytes : 0.0;
    }
};

struct BuildInfo {
    bool avx2_fma = false;
    bool openmp = false;
    bool cuda = false;
    int max_threads = 1;
    std::string compiler;
};

[[nodiscard]] BuildInfo get_build_info() noexcept;

}  // namespace tinyinfer
