#pragma once

#include <cstddef>

#include "tinyinfer/types.hpp"

namespace tinyinfer {

// Computes Y = X @ W^T where:
//   X is row-major [m, k]
//   W is row-major [n, k]
//   Y is row-major [m, n]
void matmul_fp32_nt(
    const float* x,
    const float* w,
    float* y,
    std::size_t m,
    std::size_t n,
    std::size_t k
);

// Computes Y = X @ W_q8^T where:
//   X    is row-major FP32 [m, k]
//   W_q8 is Q8_0 quantized [n, k]
//   Y    is row-major FP32 [m, n]
void matmul_q8_0_nt(
    const float* x,
    const Q8Matrix& w_q8,
    float* y,
    std::size_t m
);

}  // namespace tinyinfer
