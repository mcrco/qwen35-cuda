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

template<typename hidden_t>
__global__ void shiftInsertConvStateKernel(
        const hidden_t *qkv,
        hidden_t *conv_state,
        int conv_kernel_dim,
        int conv_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Assign each thread to shift and insert one or more convolution channels.
    for (int c = idx; c < conv_size; c += stride) {
        for (int k = 0; k + 1 < conv_kernel_dim; k++) {
            conv_state[k * conv_size + c] = conv_state[(k + 1) * conv_size + c];
        }
        conv_state[(conv_kernel_dim - 1) * conv_size + c] = qkv[c];
    }
}

template<typename hidden_t, typename weight_t, typename compute_t>
__global__ void conv1dSiluKernel(
        const hidden_t *conv_state,
        const weight_t *conv_weight,
        const weight_t *conv_bias,
        hidden_t *mixed_qkv,
        int conv_kernel_dim,
        int conv_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Assign each thread to compute one or more depthwise conv outputs.
    for (int c = idx; c < conv_size; c += stride) {
        compute_t sum = conv_bias == nullptr
            ? static_cast<compute_t>(0)
            : gpu_ops::read_as<compute_t>(conv_bias[c]);
        // Sum over the rolling convolution window for this channel.
        for (int k = 0; k < conv_kernel_dim; k++) {
            compute_t state_val = gpu_ops::read_as<compute_t>(conv_state[k * conv_size + c]);
            compute_t weight_val = gpu_ops::read_as<compute_t>(conv_weight[c * conv_kernel_dim + k]);
            sum += state_val * weight_val;
        }
        mixed_qkv[c] = gpu_ops::write_from<hidden_t>(silu(sum));
    }
}

template<typename hidden_t, typename compute_t>
__global__ void splitQkvKernel(
        const hidden_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        int num_key_heads,
        int num_value_heads,
        int key_head_dim,
        int value_head_dim) {
    // Assign each thread to split one value head.
    int vh = threadIdx.x + blockIdx.x * blockDim.x;
    int group_size = num_value_heads / num_key_heads;
    int keys_size = num_key_heads * key_head_dim;

    if (vh >= num_value_heads) {
        return;
    }

    int kh = vh / group_size;
    // Copy the grouped query/key head into the per-value-head layout.
    for (int d = 0; d < key_head_dim; d++) {
        queries[vh * key_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[kh * key_head_dim + d]);
        keys[vh * key_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[keys_size + kh * key_head_dim + d]);
    }
    // Copy the value head.
    for (int d = 0; d < value_head_dim; d++) {
        values[vh * value_head_dim + d] =
            gpu_ops::read_as<compute_t>(mixed_qkv[2 * keys_size + vh * value_head_dim + d]);
    }
}

template<typename hidden_t, typename compute_t>
__global__ void normalizeMixedQkKernel(
        hidden_t *mixed_qkv,
        int num_key_heads,
        int key_head_dim) {
    extern __shared__ compute_t partial_sums[];

    // Assign each block to one query or key row in mixed_qkv.
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int keys_size = num_key_heads * key_head_dim;
    bool is_key = row >= num_key_heads;
    int head = is_key ? row - num_key_heads : row;
    int base = (is_key ? keys_size : 0) + head * key_head_dim;

    compute_t local_sum = static_cast<compute_t>(0);
    for (int d = tid; d < key_head_dim; d += blockDim.x) {
        compute_t value = gpu_ops::read_as<compute_t>(mixed_qkv[base + d]);
        local_sum += value * value;
    }
    partial_sums[tid] = local_sum;
    __syncthreads();

    // Reduce row sum of squares.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    compute_t scale = is_key
        ? static_cast<compute_t>(1)
        : rsqrt(static_cast<compute_t>(key_head_dim));
    compute_t coeff = scale / sqrt(static_cast<double>(partial_sums[0]) + 1.0e-6);
    for (int d = tid; d < key_head_dim; d += blockDim.x) {
        compute_t result = gpu_ops::read_as<compute_t>(mixed_qkv[base + d]) * coeff;
        mixed_qkv[base + d] = gpu_ops::write_from<hidden_t>(result);
    }
}

template<typename hidden_t, typename weight_t, typename compute_t>
__global__ void gatedDeltaUpdateKernel(
        hidden_t *state,
        const hidden_t *mixed_qkv,
        const hidden_t *beta_raw,
        const hidden_t *decay_raw,
        const weight_t *dt_bias,
        const weight_t *A_log,
        float *weighted_values,
        int num_key_heads,
        int num_value_heads,
        int key_head_dim,
        int value_head_dim) {
    extern __shared__ compute_t scratch[];

    // Assign each block to one (value_head, value_dim) state column.
    int hv = blockIdx.x;
    int h = hv / value_head_dim;
    int v = hv - h * value_head_dim;
    if (h >= num_value_heads) {
        return;
    }

    compute_t beta = sigmoid(gpu_ops::read_as<compute_t>(beta_raw[h]));
    compute_t decay = -exp(gpu_ops::read_as<compute_t>(A_log[h])) *
        softplus(gpu_ops::read_as<compute_t>(decay_raw[h]) + gpu_ops::read_as<compute_t>(dt_bias[h]));
    compute_t decay_coeff = exp(decay);
    int state_head_base = h * key_head_dim * value_head_dim;
    int value_head_base = h * value_head_dim;
    int group_size = num_value_heads / num_key_heads;
    int kh = h / group_size;
    int keys_size = num_key_heads * key_head_dim;
    int query_base = kh * key_head_dim;
    int key_base = keys_size + kh * key_head_dim;
    int value_base = 2 * keys_size + h * value_head_dim;

    // Compute predicted[h, v] with a block-level reduction over key dimension.
    compute_t predicted_partial = static_cast<compute_t>(0);
    for (int k = threadIdx.x; k < key_head_dim; k += blockDim.x) {
        int idx = state_head_base + k * value_head_dim + v;
        compute_t decayed = gpu_ops::read_as<compute_t>(state[idx]) * decay_coeff;
        compute_t key = gpu_ops::read_as<compute_t>(mixed_qkv[key_base + k]);
        predicted_partial += decayed * key;
    }
    scratch[threadIdx.x] = predicted_partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    compute_t value = gpu_ops::read_as<compute_t>(mixed_qkv[value_base + v]);
    compute_t delta = (value - scratch[0]) * beta;
    compute_t out_partial = static_cast<compute_t>(0);
    // Update the state column and compute weighted_values partials in one pass over k.
    for (int k = threadIdx.x; k < key_head_dim; k += blockDim.x) {
        int idx = state_head_base + k * value_head_dim + v;
        compute_t key = gpu_ops::read_as<compute_t>(mixed_qkv[key_base + k]);
        compute_t query = gpu_ops::read_as<compute_t>(mixed_qkv[query_base + k]);
        compute_t decayed = gpu_ops::read_as<compute_t>(state[idx]) * decay_coeff;
        compute_t updated = decayed + key * delta;
        state[idx] = gpu_ops::write_from<hidden_t>(updated);
        out_partial += updated * query;
    }
    scratch[threadIdx.x] = out_partial;
    __syncthreads();

    // Reduce weighted_values partials across key dimension.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        weighted_values[value_head_base + v] = static_cast<float>(scratch[0]);
    }
}

inline int blocks_for(size_t n) {
    return min(static_cast<int>((n + LINEAR_ATTENTION_THREADS - 1) / LINEAR_ATTENTION_THREADS), LINEAR_ATTENTION_MAX_BLOCKS);
}

} // namespace linear_attention_detail

template<typename hidden_t, typename weight_t, typename compute_t>
void LinearAttention::conv1d_silu(
        const hidden_t *qkv,
        hidden_t *conv_state,
        const weight_t *conv_weight,
        const weight_t *conv_bias,
        hidden_t *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = linear_attention_detail::blocks_for(conv_kernel_dim * conv_size);
    // Shift the convolution state and insert the current qkv vector.
    linear_attention_detail::shiftInsertConvStateKernel<hidden_t><<<blocks, threads, 0, stream>>>(
        qkv, conv_state, static_cast<int>(conv_kernel_dim), static_cast<int>(conv_size));
    checkCuda(cudaGetLastError());

    // Apply depthwise conv and SiLU to produce mixed qkv.
    blocks = linear_attention_detail::blocks_for(conv_size);
    linear_attention_detail::conv1dSiluKernel<hidden_t, weight_t, compute_t><<<blocks, threads, 0, stream>>>(
        conv_state, conv_weight, conv_bias, mixed_qkv,
        static_cast<int>(conv_kernel_dim), static_cast<int>(conv_size));
    checkCuda(cudaGetLastError());
}

template<typename hidden_t, typename compute_t>
void LinearAttention::split_qkv(
        const hidden_t *mixed_qkv,
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
    // Expand grouped query/key heads and split values out of mixed qkv.
    linear_attention_detail::splitQkvKernel<hidden_t, compute_t><<<blocks, threads, 0, stream>>>(
        mixed_qkv, queries, keys, values,
        static_cast<int>(num_key_heads), static_cast<int>(num_value_heads),
        static_cast<int>(key_head_dim), static_cast<int>(value_head_dim));
    checkCuda(cudaGetLastError());
}

template<typename hidden_t, typename compute_t>
void LinearAttention::normalize_mixed_qk(
        hidden_t *mixed_qkv,
        size_t num_key_heads,
        size_t key_head_dim,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = static_cast<int>(2 * num_key_heads);
    size_t shared_bytes = static_cast<size_t>(threads) * sizeof(compute_t);
    // Normalize query rows with 1/sqrt(d) scaling and key rows without extra scaling.
    linear_attention_detail::normalizeMixedQkKernel<hidden_t, compute_t><<<blocks, threads, shared_bytes, stream>>>(
        mixed_qkv, static_cast<int>(num_key_heads), static_cast<int>(key_head_dim));
    checkCuda(cudaGetLastError());
}

template<typename hidden_t, typename weight_t, typename compute_t>
void LinearAttention::gated_delta_update(
        hidden_t *state,
        const hidden_t *mixed_qkv,
        const hidden_t *beta_raw,
        const hidden_t *decay_raw,
        const weight_t *dt_bias,
        const weight_t *A_log,
        float *weighted_values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim,
        cudaStream_t stream) {
    int threads = linear_attention_detail::LINEAR_ATTENTION_THREADS;
    int blocks = static_cast<int>(num_value_heads * value_head_dim);
    size_t shared_bytes = static_cast<size_t>(threads) * sizeof(compute_t);
    // Fuse decay, delta update, and output reduction per (value_head, value_dim).
    linear_attention_detail::gatedDeltaUpdateKernel<hidden_t, weight_t, compute_t><<<blocks, threads, shared_bytes, stream>>>(
        state, mixed_qkv, beta_raw, decay_raw, dt_bias, A_log,
        weighted_values, static_cast<int>(num_key_heads), static_cast<int>(num_value_heads),
        static_cast<int>(key_head_dim), static_cast<int>(value_head_dim));
    checkCuda(cudaGetLastError());
}

template void LinearAttention::conv1d_silu<float, float, float>(
    const float*, float*, const float*, const float*, float*, size_t, size_t, cudaStream_t);
template void LinearAttention::split_qkv<float, float>(
    const float*, float*, float*, float*, size_t, size_t, size_t, size_t, cudaStream_t);
template void LinearAttention::normalize_mixed_qk<float, float>(
    float*, size_t, size_t, cudaStream_t);
template void LinearAttention::gated_delta_update<float, float, float>(
    float*, const float*, const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t, size_t, cudaStream_t);
