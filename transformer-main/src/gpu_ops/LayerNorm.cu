#include "LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

#include <cmath>

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

template<typename hidden_t, typename weight_t, typename output_t, typename compute_t>
__global__ void zeroCenteredLayerNormKernel(const hidden_t *hidden_state, const weight_t *weights, compute_t rms, output_t *output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        compute_t result = gpu_ops::read_as<compute_t>(hidden_state[i]) *
            (static_cast<compute_t>(1) + gpu_ops::read_as<compute_t>(weights[i])) / rms;
        output[i] = gpu_ops::write_from<output_t>(result);
    }
}

template<typename compute_t>
__device__ compute_t silu(compute_t x) {
    float xf = static_cast<float>(x);
    return static_cast<compute_t>(xf / (1.0f + expf(-xf)));
}

template<typename hidden_t, typename gate_t, typename weight_t, typename output_t, typename compute_t>
__global__ void gatedRmsNormRowsKernel(const hidden_t *hidden_state, const gate_t *gate, const weight_t *weights, output_t *output, int rows, int cols, float eps) {
    extern __shared__ unsigned char shared[];
    compute_t *partial_sums = reinterpret_cast<compute_t*>(shared);

    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int tidx = threadIdx.x;
    const hidden_t *hidden_row = hidden_state + row * cols;
    const gate_t *gate_row = gate + row * cols;
    output_t *output_row = output + row * cols;

    compute_t local_sum = static_cast<compute_t>(0);
    for (int col = tidx; col < cols; col += blockDim.x) {
        compute_t input = gpu_ops::read_as<compute_t>(hidden_row[col]);
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

    compute_t inv_rms = static_cast<compute_t>(1) /
        sqrt(static_cast<double>(partial_sums[0]) / static_cast<double>(cols) + static_cast<double>(eps));
    for (int col = tidx; col < cols; col += blockDim.x) {
        compute_t result = gpu_ops::read_as<compute_t>(hidden_row[col]) *
            inv_rms *
            gpu_ops::read_as<compute_t>(weights[col]) *
            silu(gpu_ops::read_as<compute_t>(gate_row[col]));
        output_row[col] = gpu_ops::write_from<output_t>(result);
    }
}

template<typename value_t, typename compute_t>
__global__ void l2NormRowsKernel(value_t *values, int rows, int cols, float scale, float eps) {
    extern __shared__ unsigned char shared[];
    compute_t *partial_sums = reinterpret_cast<compute_t*>(shared);

    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int tidx = threadIdx.x;
    value_t *values_row = values + row * cols;

    compute_t local_sum = static_cast<compute_t>(0);
    for (int col = tidx; col < cols; col += blockDim.x) {
        compute_t input = gpu_ops::read_as<compute_t>(values_row[col]);
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

    compute_t coeff = static_cast<compute_t>(scale) /
        sqrt(static_cast<double>(partial_sums[0]) + static_cast<double>(eps));
    for (int col = tidx; col < cols; col += blockDim.x) {
        compute_t result = gpu_ops::read_as<compute_t>(values_row[col]) * coeff;
        values_row[col] = gpu_ops::write_from<value_t>(result);
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

template<typename hidden_t, typename weight_t, typename output_t, typename compute_t>
void LayerNorm::zero_centered_rms_norm(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, int32_t n, float eps, cudaStream_t stream) {
    zero_centered_rms_norm<hidden_t, weight_t, output_t, compute_t>(
        static_cast<const hidden_t*>(hidden_state->data),
        static_cast<output_t*>(output->data),
        n,
        eps,
        stream);
}

template<typename hidden_t, typename weight_t, typename output_t, typename compute_t>
void LayerNorm::zero_centered_rms_norm(const hidden_t *hidden_state_ptr, output_t *output_ptr, int32_t n, float eps, cudaStream_t stream) {
    int threads = layer_norm_detail::LAYERNORM_THREADS;
    int blocks = min((n + threads - 1) / threads, layer_norm_detail::LAYERNORM_MAX_BLOCKS);

    const weight_t *weights_ptr = static_cast<const weight_t*>(weights->data);

    auto sum_of_squares_buffer = std::make_shared<CudaBuffer>(sizeof(compute_t));
    compute_t *sum_of_squares = static_cast<compute_t*>(sum_of_squares_buffer->data);
    checkCuda(cudaMemsetAsync(sum_of_squares, 0, sizeof(compute_t), stream));
    int shared_mem_size = threads * sizeof(compute_t);
    layer_norm_detail::sumOfSquaresKernel<hidden_t, compute_t><<<blocks, threads, shared_mem_size, stream>>>(hidden_state_ptr, sum_of_squares, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));
    compute_t rms = sqrt(static_cast<double>(*sum_of_squares) / n + static_cast<double>(eps));

    layer_norm_detail::zeroCenteredLayerNormKernel<hidden_t, weight_t, output_t, compute_t><<<blocks, threads, 0, stream>>>(hidden_state_ptr, weights_ptr, rms, output_ptr, n);
    checkCuda(cudaGetLastError());
}

template<typename hidden_t, typename gate_t, typename weight_t, typename output_t, typename compute_t>
void LayerNorm::normalize_gated_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &gate, const std::shared_ptr<CudaBuffer> &output, int32_t rows, int32_t cols, float eps, cudaStream_t stream) {
    int threads = layer_norm_detail::LAYERNORM_THREADS;
    int shared_mem_size = threads * sizeof(compute_t);

    const hidden_t *hidden_state_ptr = static_cast<const hidden_t*>(hidden_state->data);
    const gate_t *gate_ptr = static_cast<const gate_t*>(gate->data);
    output_t *output_ptr = static_cast<output_t*>(output->data);
    const weight_t *weights_ptr = static_cast<const weight_t*>(weights->data);

    layer_norm_detail::gatedRmsNormRowsKernel<hidden_t, gate_t, weight_t, output_t, compute_t><<<rows, threads, shared_mem_size, stream>>>(
        hidden_state_ptr, gate_ptr, weights_ptr, output_ptr, rows, cols, eps);
    checkCuda(cudaGetLastError());
}

template<typename value_t, typename compute_t>
void LayerNorm::l2_norm_rows(const std::shared_ptr<CudaBuffer> &values, int32_t rows, int32_t cols, float scale, float eps, cudaStream_t stream) {
    int threads = layer_norm_detail::LAYERNORM_THREADS;
    int shared_mem_size = threads * sizeof(compute_t);
    value_t *values_ptr = static_cast<value_t*>(values->data);

    layer_norm_detail::l2NormRowsKernel<value_t, compute_t><<<rows, threads, shared_mem_size, stream>>>(values_ptr, rows, cols, scale, eps);
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

template void LayerNorm::zero_centered_rms_norm<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    float eps,
    cudaStream_t stream);

template void LayerNorm::zero_centered_rms_norm<float, float, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    float eps,
    cudaStream_t stream);

template void LayerNorm::zero_centered_rms_norm<float, float, float, float>(
    const float *hidden_state,
    float *output,
    int32_t n,
    float eps,
    cudaStream_t stream);

template void LayerNorm::normalize_gated_hidden_state<float, float, float, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &gate,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t rows,
    int32_t cols,
    float eps,
    cudaStream_t stream);

template void LayerNorm::l2_norm_rows<float, float>(
    const std::shared_ptr<CudaBuffer> &values,
    int32_t rows,
    int32_t cols,
    float scale,
    float eps,
    cudaStream_t stream);
