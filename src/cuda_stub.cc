#include "tinyinfer/cuda_backend.hpp"

#include <stdexcept>

namespace tinyinfer {

bool cuda_available() noexcept {
    return false;
}

std::string cuda_device_name() {
    return "none (CPU-only build)";
}

void cuda_matmul_fp32_nt(
    const float*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t
) {
    throw std::runtime_error(
        "CUDA backend unavailable: rebuild with -DTINYINFER_ENABLE_CUDA=ON."
    );
}

void cuda_matmul_q8_0_nt(
    const float*,
    const Q8Matrix&,
    float*,
    std::size_t
) {
    throw std::runtime_error(
        "CUDA backend unavailable: rebuild with -DTINYINFER_ENABLE_CUDA=ON."
    );
}

}  // namespace tinyinfer
