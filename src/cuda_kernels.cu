#include "tinyinfer/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tinyinfer {

namespace {

void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error (") + context + "): " + cudaGetErrorString(err)
        );
    }
}

template <typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
        }
    }

    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void copy_from_host(const T* src) {
        if (count_ > 0) {
            check_cuda(
                cudaMemcpy(ptr_, src, count_ * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy H2D"
            );
        }
    }

    void copy_to_host(T* dst) const {
        if (count_ > 0) {
            check_cuda(
                cudaMemcpy(dst, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H"
            );
        }
    }

    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void kernel_matmul_fp32_nt(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    std::size_t n,
    std::size_t k
) {
    const std::size_t row_n = blockIdx.x;
    const std::size_t row_m = blockIdx.y;
    const unsigned int lane = threadIdx.x;

    const float* x_row = x + row_m * k;
    const float* w_row = w + row_n * k;

    float acc = 0.0f;
    for (std::size_t i = lane; i < k; i += 32) {
        acc = fmaf(x_row[i], w_row[i], acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row_m * n + row_n] = acc;
    }
}

__global__ void kernel_matmul_q8_0_nt(
    const float* __restrict__ x,
    const int8_t* __restrict__ q_weights,
    const float* __restrict__ q_scales,
    float* __restrict__ y,
    std::size_t n,
    std::size_t k,
    std::size_t blocks_per_row
) {
    const std::size_t row_n = blockIdx.x;
    const std::size_t row_m = blockIdx.y;
    const unsigned int lane = threadIdx.x;

    const float* x_row = x + row_m * k;
    const int8_t* q_row = q_weights + row_n * blocks_per_row * kQ8BlockSize;
    const float* s_row = q_scales + row_n * blocks_per_row;

    float acc = 0.0f;
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
        const std::size_t col = b * kQ8BlockSize + lane;
        if (col < k) {
            const float w_val = static_cast<float>(q_row[b * kQ8BlockSize + lane]) * s_row[b];
            acc = fmaf(x_row[col], w_val, acc);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row_m * n + row_n] = acc;
    }
}

}  // namespace

bool cuda_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_name() {
    if (!cuda_available()) {
        return "none (no CUDA device detected)";
    }
    cudaDeviceProp prop {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "CUDA device 0";
    }
    return std::string(prop.name);
}

void cuda_matmul_fp32_nt(
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
    if (!cuda_available()) {
        throw std::runtime_error("cuda_matmul_fp32_nt: no CUDA device detected.");
    }

    CudaBuffer<float> d_x(m * k);
    CudaBuffer<float> d_w(n * k);
    CudaBuffer<float> d_y(m * n);
    d_x.copy_from_host(x);
    d_w.copy_from_host(w);

    const dim3 grid(static_cast<unsigned int>(n), static_cast<unsigned int>(m), 1);
    kernel_matmul_fp32_nt<<<grid, 32>>>(d_x.get(), d_w.get(), d_y.get(), n, k);
    check_cuda(cudaGetLastError(), "kernel_matmul_fp32_nt launch");
    check_cuda(cudaDeviceSynchronize(), "kernel_matmul_fp32_nt sync");
    d_y.copy_to_host(y);
}

void cuda_matmul_q8_0_nt(
    const float* x,
    const Q8Matrix& w_q8,
    float* y,
    std::size_t m
) {
    const std::size_t n = w_q8.rows;
    const std::size_t k = w_q8.cols;
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (!cuda_available()) {
        throw std::runtime_error("cuda_matmul_q8_0_nt: no CUDA device detected.");
    }

    CudaBuffer<float> d_x(m * k);
    CudaBuffer<int8_t> d_q(w_q8.weights.size());
    CudaBuffer<float> d_s(w_q8.scales.size());
    CudaBuffer<float> d_y(m * n);
    d_x.copy_from_host(x);
    d_q.copy_from_host(w_q8.weights.data());
    d_s.copy_from_host(w_q8.scales.data());

    const dim3 grid(static_cast<unsigned int>(n), static_cast<unsigned int>(m), 1);
    kernel_matmul_q8_0_nt<<<grid, 32>>>(
        d_x.get(), d_q.get(), d_s.get(), d_y.get(), n, k, w_q8.blocks_per_row
    );
    check_cuda(cudaGetLastError(), "kernel_matmul_q8_0_nt launch");
    check_cuda(cudaDeviceSynchronize(), "kernel_matmul_q8_0_nt sync");
    d_y.copy_to_host(y);
}

}  // namespace tinyinfer
