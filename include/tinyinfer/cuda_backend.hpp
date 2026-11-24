#pragma once

#include <cstddef>
#include <string>

#include "tinyinfer/types.hpp"

namespace tinyinfer {

// Returns true if compiled with TINYINFER_ENABLE_CUDA=ON and a CUDA device is accessible.
[[nodiscard]] bool cuda_available() noexcept;

// Returns active CUDA device name or "none (CPU-only build)".
[[nodiscard]] std::string cuda_device_name();

// Computes Y = X @ W^T on CUDA device (warp-reduced FP32 kernel).
void cuda_matmul_fp32_nt(
    const float* x,
    const float* w,
    float* y,
    std::size_t m,
    std::size_t n,
    std::size_t k
);

// Computes Y = X @ W_q8^T on CUDA device (warp-reduced Q8_0 block-32 kernel).
void cuda_matmul_q8_0_nt(
    const float* x,
    const Q8Matrix& w_q8,
    float* y,
    std::size_t m
);

}  // namespace tinyinfer
