#include "RoPE.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

const int ROPE_THREADS = 128;
const int ROPE_MAX_BLOCKS = 1024;

__global__ void ropeKernel(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t position_idx, float theta_base) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_heads * head_dim / 2; i += stride) {
        // Get true lesser index for rotation.
        int h = i / (head_dim / 2);
        int m = i % (head_dim / 2);
        int j = h * head_dim + m;

        // Trigonometry stuff.
        float theta = powf(theta_base, -2.0 * m / head_dim);
        float cosval = cosf(position_idx * theta);
        float sinval = sinf(position_idx * theta);

        // Rotate x[j] with x[j + d/2].
        float first = __bfloat162float(x[j]);
        float second = __bfloat162float(x[j + (head_dim / 2)]);
        x[j] = __float2bfloat16(cosval * first - sinval * second);
        x[j + (head_dim / 2)] = __float2bfloat16(sinval * first + cosval * second);
    }
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    int threads = ROPE_THREADS;
    int blocks = min((num_heads * head_dim + threads - 1) / threads, ROPE_MAX_BLOCKS);

    ropeKernel<<<blocks, threads, 0, stream>>>(x, num_heads, head_dim, position_idx, theta_base);
    checkCuda(cudaGetLastError());
}
