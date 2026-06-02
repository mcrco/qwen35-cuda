#include "LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace layer_norm_detail {

constexpr int LAYERNORM_THREADS = 128;
constexpr int LAYERNORM_MAX_BLOCKS = 1024;

template<typename hidden_t, typename compute_t>
__global__ void sumOfSquaresKernel(const hidden_t *data, compute_t *output, int n) {
    extern __shared__ unsigned char shared[];
    compute_t *partial_sums = reinterpret_cast<compute_t*>(shared);

    int tidx = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    compute_t local_sum = static_cast<compute_t>(0);
    for (int i = idx; i < n; i += stride) {
        compute_t input = gpu_ops::read_as<compute_t>(data[i]);
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

template<typename hidden_t, typename weight_t, typename output_t, typename compute_t>
__global__ void layerNormKernel(const hidden_t *hidden_state, const weight_t *weights, compute_t rms, output_t *output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        compute_t result = gpu_ops::read_as<compute_t>(hidden_state[i]) * gpu_ops::read_as<compute_t>(weights[i]) / rms;
        output[i] = gpu_ops::write_from<output_t>(result);
    }
}

} // namespace layer_norm_detail

LayerNorm::LayerNorm(int32_t len) {}

template<typename hidden_t, typename weight_t, typename output_t, typename compute_t>
void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, int32_t n, cudaStream_t stream) {
    int threads = layer_norm_detail::LAYERNORM_THREADS;
    int blocks = min((n + threads - 1) / threads, layer_norm_detail::LAYERNORM_MAX_BLOCKS);

    const hidden_t *hidden_state_ptr = static_cast<const hidden_t*>(hidden_state->data);
    output_t *output_ptr = static_cast<output_t*>(output->data);
    const weight_t *weights_ptr = static_cast<const weight_t*>(weights->data);

    auto sum_of_squares_buffer = std::make_shared<CudaBuffer>(sizeof(compute_t));
    compute_t *sum_of_squares = static_cast<compute_t*>(sum_of_squares_buffer->data);
    checkCuda(cudaMemsetAsync(sum_of_squares, 0, sizeof(compute_t), stream));
    int shared_mem_size = threads * sizeof(compute_t);
    layer_norm_detail::sumOfSquaresKernel<hidden_t, compute_t><<<blocks, threads, shared_mem_size, stream>>>(hidden_state_ptr, sum_of_squares, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));
    compute_t rms = sqrt(static_cast<double>(*sum_of_squares) / n + static_cast<double>(LayerNorm::EPS));

    layer_norm_detail::layerNormKernel<hidden_t, weight_t, output_t, compute_t><<<blocks, threads, 0, stream>>>(hidden_state_ptr, weights_ptr, rms, output_ptr, n);
    checkCuda(cudaGetLastError());
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    int n = hidden_state->size / sizeof(__nv_bfloat16);
    normalize_hidden_state<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(hidden_state, output, n, stream);
}

template void LayerNorm::normalize_hidden_state<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    cudaStream_t stream);

template void LayerNorm::normalize_hidden_state<float, __nv_bfloat16, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    cudaStream_t stream);
