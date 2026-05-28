#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include <fcntl.h>
#include "../ErrorCheck.h"

const int LAYERNORM_THREADS = 128;
const int LAYERNORM_MAX_BLOCKS = 1024;

LayerNorm::LayerNorm(int32_t len) {}

__global__ void sumOfSquaresKernel(__nv_bfloat16 *data, float *output, int n) {
    extern __shared__ float partial_sums[];

    int tidx = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0;
    for (int i = idx; i < n; i += stride) {
        float input = __bfloat162float(data[i]);
        local_sum += input * input;
    }
    partial_sums[tidx] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            partial_sums[tidx] += partial_sums[tidx + s];
        }
        __syncthreads();
    }

    if (tidx == 0) {
        atomicAdd(output, partial_sums[tidx]);
    }
}

__global__ void layerNormKernel(__nv_bfloat16 *hidden_state, __nv_bfloat16 *weights, float rms, __nv_bfloat16 *output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float result = __bfloat162float(hidden_state[i]) * __bfloat162float(weights[i]) / rms;
        output[i] = __float2bfloat16(result);
    }
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    int n = hidden_state->size / sizeof(__nv_bfloat16);

    int threads = LAYERNORM_THREADS;
    int blocks = min((n + threads - 1) / threads, LAYERNORM_MAX_BLOCKS);

    __nv_bfloat16 *hidden_state_ptr = static_cast<__nv_bfloat16*>(hidden_state->data);
    __nv_bfloat16 *output_ptr = static_cast<__nv_bfloat16*>(output->data);
    __nv_bfloat16 *weights_ptr = static_cast<__nv_bfloat16*>(weights->data);

    // Compute RMS norm.
    auto sum_of_squares_buffer = std::make_shared<CudaBuffer>(sizeof(float));
    float *sum_of_squares = static_cast<float*>(sum_of_squares_buffer->data);
    cudaMemset(sum_of_squares, 0, sizeof(float));
    int shared_mem_size = threads * sizeof(float);
    sumOfSquaresKernel<<<blocks, threads, shared_mem_size, stream>>>(hidden_state_ptr, sum_of_squares, n);
    checkCuda(cudaGetLastError());
    cudaStreamSynchronize(stream);
    float rms = sqrtf(*sum_of_squares / n + LayerNorm::EPS);

    // Perform layer norm.
    layerNormKernel<<<blocks, threads, 0, stream>>>(hidden_state_ptr, weights_ptr, rms, output_ptr, n);
    checkCuda(cudaGetLastError());
}
