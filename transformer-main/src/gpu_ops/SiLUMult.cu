#include "SiLUMult.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace silu_mult_detail {

constexpr int SILU_MULT_THREADS = 128;
constexpr int SILU_MAX_BLOCKS = 1024;

template<typename compute_t>
__device__ inline compute_t sigmoid(compute_t x) {
    return static_cast<compute_t>(1) / (static_cast<compute_t>(1) + exp(-x));
}

template<typename x_t, typename y_t, typename compute_t>
__global__ void siluKernel(x_t *x, const y_t *y, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        compute_t xval = gpu_ops::read_as<compute_t>(x[i]);
        compute_t yval = gpu_ops::read_as<compute_t>(y[i]);
        compute_t result = xval * yval * sigmoid(xval);
        x[i] = gpu_ops::write_from<x_t>(result);
    }
}

} // namespace silu_mult_detail

template<typename x_t, typename y_t, typename compute_t>
void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, int32_t n, cudaStream_t stream) {
    int threads = silu_mult_detail::SILU_MULT_THREADS;
    int blocks = min((n + threads - 1) / threads, silu_mult_detail::SILU_MAX_BLOCKS);

    x_t *x_ptr = static_cast<x_t*>(x->data);
    const y_t *y_ptr = static_cast<const y_t*>(y->data);

    silu_mult_detail::siluKernel<x_t, y_t, compute_t><<<blocks, threads, 0, stream>>>(x_ptr, y_ptr, n);
    checkCuda(cudaGetLastError());
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    int n = x->size / sizeof(__nv_bfloat16);
    silu_mult_in_place<__nv_bfloat16, __nv_bfloat16, float>(x, y, n, stream);
}

template void SiLUMult::silu_mult_in_place<__nv_bfloat16, __nv_bfloat16, float>(const std::shared_ptr<CudaBuffer>&, const std::shared_ptr<CudaBuffer>&, int32_t, cudaStream_t);
template void SiLUMult::silu_mult_in_place<float, float, float>(const std::shared_ptr<CudaBuffer>&, const std::shared_ptr<CudaBuffer>&, int32_t, cudaStream_t);
