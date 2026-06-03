#include "LinearAttention.cuh"

#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace linear_attention_detail {

constexpr int LINEAR_ATTENTION_THREADS = 128;
constexpr int LINEAR_ATTENTION_MAX_BLOCKS = 1024;

template<typename compute_t>
__device__ compute_t silu(compute_t x) {
    return x / (static_cast<compute_t>(1) + exp(-x));
}

template<typename compute_t>
__device__ compute_t sigmoid(compute_t x) {
    return static_cast<compute_t>(1) / (static_cast<compute_t>(1) + exp(-x));
}

template<typename compute_t>
__device__ compute_t softplus(compute_t x) {
    if (x > static_cast<compute_t>(20)) {
        return x;
    }
    if (x < static_cast<compute_t>(-20)) {
        return exp(x);
    }
    return log(static_cast<compute_t>(1) + exp(x));
}

__global__ void shiftInsertConvStateKernel(
        const input_float_t *qkv,
        input_float_t *conv_state,
        int conv_kernel_dim,
        int conv_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int c = idx; c < conv_size; c += stride) {
        for (int k = 0; k + 1 < conv_kernel_dim; k++) {
            conv_state[k * conv_size + c] = conv_state[(k + 1) * conv_size + c];
        }
        conv_state[(conv_kernel_dim - 1) * conv_size + c] = qkv[c];
    }
}

template<typename compute_t>
__global__ void conv1dSiluKernel(
        const input_float_t *conv_state,
        const input_float_t *conv_weight,
        const input_float_t *conv_bias,
        input_float_t *mixed_qkv,
        int conv_kernel_dim,
        int conv_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int c = idx; c < conv_size; c += stride) {
        compute_t sum = conv_bias == nullptr
            ? static_cast<compute_t>(0)
            : gpu_ops::read_as<compute_t>(conv_bias[c]);
        for (int k = 0; k < conv_kernel_dim; k++) {
            compute_t state_val = gpu_ops::read_as<compute_t>(conv_state[k * conv_size + c]);
            compute_t weight_val = gpu_ops::read_as<compute_t>(conv_weight[c * conv_kernel_dim + k]);
            sum += state_val * weight_val;
        }
        mixed_qkv[c] = gpu_ops::write_from<input_float_t>(silu(sum));
    }
}

template<typename compute_t>
__global__ void splitQkvKernel(
        const input_float_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        int num_key_heads,
        int num_value_heads,
        int key_head_dim,
        int value_head_dim) {
    int vh = threadIdx.x + blockIdx.x * blockDim.x;
    int group_size = num_value_heads / num_key_heads;
    int keys_size = num_key_heads * key_head_dim;

    if (vh >= num_value_heads) {
        return;
    }

    int kh = vh / group_size;
    for (int d = 0; d < key_head_dim; d++) {
        queries[vh * key_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[kh * key_head_dim + d]);
        keys[vh * key_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[keys_size + kh * key_head_dim + d]);
    }
    for (int d = 0; d < value_head_dim; d++) {
        values[vh * value_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[2 * keys_size + vh * value_head_dim + d]);
    }
}

template<typename compute_t>
__global__ void gatedDeltaUpdateKernel(
        input_float_t *state,
        const float *queries,
        const float *keys,
        const float *values,
        const input_float_t *beta_raw,
        const input_float_t *decay_raw,
        const input_float_t *dt_bias,
        const input_float_t *A_log,
        float *weighted_values,
        int num_value_heads,
        int key_head_dim,
        int value_head_dim) {
    int h = threadIdx.x + blockIdx.x * blockDim.x;
    if (h >= num_value_heads) {
        return;
    }

    compute_t beta = sigmoid(gpu_ops::read_as<compute_t>(beta_raw[h]));
    compute_t decay = -exp(gpu_ops::read_as<compute_t>(A_log[h])) *
        softplus(gpu_ops::read_as<compute_t>(decay_raw[h]) + gpu_ops::read_as<compute_t>(dt_bias[h]));
    compute_t decay_coeff = exp(decay);
    int state_head_base = h * key_head_dim * value_head_dim;
    int key_head_base = h * key_head_dim;
    int value_head_base = h * value_head_dim;

    for (int k = 0; k < key_head_dim; k++) {
        for (int v = 0; v < value_head_dim; v++) {
            int idx = state_head_base + k * value_head_dim + v;
            compute_t decayed = gpu_ops::read_as<compute_t>(state[idx]) * decay_coeff;
            state[idx] = gpu_ops::write_from<input_float_t>(decayed);
        }
    }

    for (int v = 0; v < value_head_dim; v++) {
        compute_t predicted = static_cast<compute_t>(0);
        for (int k = 0; k < key_head_dim; k++) {
            int idx = state_head_base + k * value_head_dim + v;
            predicted += gpu_ops::read_as<compute_t>(state[idx]) * static_cast<compute_t>(keys[key_head_base + k]);
        }
        compute_t delta = (static_cast<compute_t>(values[value_head_base + v]) - predicted) * beta;
        for (int k = 0; k < key_head_dim; k++) {
            int idx = state_head_base + k * value_head_dim + v;
            compute_t updated = gpu_ops::read_as<compute_t>(state[idx]) +
                static_cast<compute_t>(keys[key_head_base + k]) * delta;
            state[idx] = gpu_ops::write_from<input_float_t>(updated);
        }
    }

    for (int v = 0; v < value_head_dim; v++) {
        compute_t out = static_cast<compute_t>(0);
        for (int k = 0; k < key_head_dim; k++) {
            int idx = state_head_base + k * value_head_dim + v;
            out += gpu_ops::read_as<compute_t>(state[idx]) * static_cast<compute_t>(queries[key_head_base + k]);
        }
        weighted_values[value_head_base + v] = static_cast<float>(out);
    }
}

inline int blocks_for(size_t n) {
    return min(static_cast<int>((n + LINEAR_ATTENTION_THREADS - 1) / LINEAR_ATTENTION_THREADS), LINEAR_ATTENTION_MAX_BLOCKS);
}

} // namespace linear_attention_detail

void LinearAttention::conv1d_silu(
        const input_float_t *qkv,
        input_float_t *conv_state,
        const input_float_t *conv_weight,
        const input_float_t *conv_bias,
        input_float_t *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = linear_attention_detail::blocks_for(conv_kernel_dim * conv_size);
    linear_attention_detail::shiftInsertConvStateKernel<<<blocks, threads, 0, stream>>>(
        qkv, conv_state, static_cast<int>(conv_kernel_dim), static_cast<int>(conv_size));
    checkCuda(cudaGetLastError());

    blocks = linear_attention_detail::blocks_for(conv_size);
    linear_attention_detail::conv1dSiluKernel<float><<<blocks, threads, 0, stream>>>(
        conv_state, conv_weight, conv_bias, mixed_qkv,
        static_cast<int>(conv_kernel_dim), static_cast<int>(conv_size));
    checkCuda(cudaGetLastError());
}

void LinearAttention::split_qkv(
        const input_float_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = linear_attention_detail::blocks_for(num_value_heads);
    linear_attention_detail::splitQkvKernel<float><<<blocks, threads, 0, stream>>>(
        mixed_qkv, queries, keys, values,
        static_cast<int>(num_key_heads), static_cast<int>(num_value_heads),
        static_cast<int>(key_head_dim), static_cast<int>(value_head_dim));
    checkCuda(cudaGetLastError());
}

void LinearAttention::gated_delta_update(
        input_float_t *state,
        const float *queries,
        const float *keys,
        const float *values,
        const input_float_t *beta_raw,
        const input_float_t *decay_raw,
        const input_float_t *dt_bias,
        const input_float_t *A_log,
        float *weighted_values,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = linear_attention_detail::blocks_for(num_value_heads);
    linear_attention_detail::gatedDeltaUpdateKernel<float><<<blocks, threads, 0, stream>>>(
        state, queries, keys, values, beta_raw, decay_raw, dt_bias, A_log,
        weighted_values, static_cast<int>(num_value_heads),
        static_cast<int>(key_head_dim), static_cast<int>(value_head_dim));
    checkCuda(cudaGetLastError());
}
