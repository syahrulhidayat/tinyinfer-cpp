#pragma once

#include <cstddef>
#include <vector>

#include "tinyinfer/types.hpp"

namespace tinyinfer {

// Quantizes a row-major FP32 matrix [rows, cols] into block-32 symmetric INT8 (Q8_0).
[[nodiscard]] Q8Matrix quantize_q8_0(const float* data, std::size_t rows, std::size_t cols);

// Reconstructs a row-major FP32 matrix [rows, cols] from Q8_0 for inspection or drift testing.
[[nodiscard]] std::vector<float> dequantize_q8_0(const Q8Matrix& qmat);

}  // namespace tinyinfer
