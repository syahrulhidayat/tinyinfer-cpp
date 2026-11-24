#pragma once

#include <cstddef>
#include <cstdint>

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

// Zero-copy raw pointer overload for mmap-backed Q8_0 matrices.
//   q_weights: [n, blocks_per_row * kQ8BlockSize]
//   q_scales:  [n, blocks_per_row]
void matmul_q8_0_nt_raw(
    const float* x,
    const int8_t* q_weights,
    const float* q_scales,
    float* y,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t blocks_per_row
);

// Computes Y = X @ W_q8^T using an owning Q8Matrix.
void matmul_q8_0_nt(
    const float* x,
    const Q8Matrix& w_q8,
    float* y,
    std::size_t m
);

}  // namespace tinyinfer
