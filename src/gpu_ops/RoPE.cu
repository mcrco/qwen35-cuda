#include "RoPE.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace rope_detail {

constexpr int ROPE_THREADS = 128;
constexpr int ROPE_MAX_BLOCKS = 1024;

template<typename x_t, typename compute_t>
__global__ void ropeKernel(x_t *x, int32_t num_heads, int32_t head_dim, int32_t rotary_dim, int32_t position_idx, compute_t theta_base) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_heads * rotary_dim / 2; i += stride) {
        int h = i / (rotary_dim / 2);
        int m = i % (rotary_dim / 2);
        int j = h * head_dim + m;

        compute_t theta = pow(theta_base, static_cast<compute_t>(-2.0) * m / rotary_dim);
        compute_t cosval = cos(position_idx * theta);
        compute_t sinval = sin(position_idx * theta);

        compute_t first = gpu_ops::read_as<compute_t>(x[j]);
        compute_t second = gpu_ops::read_as<compute_t>(x[j + (rotary_dim / 2)]);
        x[j] = gpu_ops::write_from<x_t>(cosval * first - sinval * second);
        x[j + (rotary_dim / 2)] = gpu_ops::write_from<x_t>(sinval * first + cosval * second);
    }
}

} // namespace rope_detail

template<typename x_t, typename compute_t>
void RoPE::apply_rope_to_qk(x_t *x, int32_t num_heads, int32_t head_dim, int32_t rotary_dim,
        int32_t position_idx, compute_t theta_base, cudaStream_t stream) {
    int threads = rope_detail::ROPE_THREADS;
    int blocks = min((num_heads * rotary_dim + threads - 1) / threads, rope_detail::ROPE_MAX_BLOCKS);

    rope_detail::ropeKernel<x_t, compute_t><<<blocks, threads, 0, stream>>>(x, num_heads, head_dim, rotary_dim, position_idx, theta_base);
    checkCuda(cudaGetLastError());
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t rotary_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    apply_rope_to_qk<__nv_bfloat16, float>(x, num_heads, head_dim, rotary_dim, position_idx, theta_base, stream);
}

template void RoPE::apply_rope_to_qk<__nv_bfloat16, float>(__nv_bfloat16*, int32_t, int32_t, int32_t, int32_t, float, cudaStream_t);
template void RoPE::apply_rope_to_qk<float, float>(float*, int32_t, int32_t, int32_t, int32_t, float, cudaStream_t);
