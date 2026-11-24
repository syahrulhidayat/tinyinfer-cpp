#pragma once

#include <cstddef>

namespace tinyinfer {

// RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
void rmsnorm(
    float* out,
    const float* x,
    const float* weight,
    std::size_t n,
    float eps = 1e-5f
);

// Numerically stable Softmax over length n.
void softmax(float* out, const float* x, std::size_t n);

// SiLU (Swish): out[i] = x[i] / (1 + exp(-x[i]))
void silu(float* out, const float* x, std::size_t n);

// Rotary Positional Embedding (RoPE) applied in-place across [n_heads, head_dim].
// head_dim must be even.
void rope_inplace(
    float* vec,
    std::size_t n_heads,
    std::size_t head_dim,
    std::size_t pos,
    float theta_base = 10000.0f
);

}  // namespace tinyinfer
