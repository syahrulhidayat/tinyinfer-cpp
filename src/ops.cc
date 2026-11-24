#include "tinyinfer/ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tinyinfer {

void rmsnorm(float* out, const float* x, const float* weight, std::size_t n, float eps) {
    if (n == 0) {
        return;
    }
    if (out == nullptr || x == nullptr || weight == nullptr) {
        throw std::invalid_argument("rmsnorm: null pointer.");
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i]);
        sum_sq += v * v;
    }
    const float inv_rms = static_cast<float>(
        1.0 / std::sqrt(sum_sq / static_cast<double>(n) + static_cast<double>(eps))
    );
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void softmax(float* out, const float* x, std::size_t n) {
    if (n == 0) {
        return;
    }
    if (out == nullptr || x == nullptr) {
        throw std::invalid_argument("softmax: null pointer.");
    }

    float max_val = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = std::exp(x[i] - max_val);
        out[i] = e;
        sum += static_cast<double>(e);
    }

    const float inv_sum = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

void silu(float* out, const float* x, std::size_t n) {
    if (n == 0) {
        return;
    }
    if (out == nullptr || x == nullptr) {
        throw std::invalid_argument("silu: null pointer.");
    }

    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
}

void rope_inplace(
    float* vec,
    std::size_t n_heads,
    std::size_t head_dim,
    std::size_t pos,
    float theta_base
) {
    if (vec == nullptr) {
        throw std::invalid_argument("rope_inplace: null pointer.");
    }
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("rope_inplace: head_dim must be even.");
    }

    const std::size_t half = head_dim / 2;
    for (std::size_t h = 0; h < n_heads; ++h) {
        float* head_ptr = vec + h * head_dim;
        for (std::size_t i = 0; i < half; ++i) {
            const double exponent = static_cast<double>(2 * i) / static_cast<double>(head_dim);
            const double freq = 1.0 / std::pow(static_cast<double>(theta_base), exponent);
            const double angle = static_cast<double>(pos) * freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));

            const float v0 = head_ptr[2 * i];
            const float v1 = head_ptr[2 * i + 1];
            head_ptr[2 * i] = v0 * c - v1 * s;
            head_ptr[2 * i + 1] = v0 * s + v1 * c;
        }
    }
}

}  // namespace tinyinfer
