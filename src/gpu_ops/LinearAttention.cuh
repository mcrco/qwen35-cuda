#pragma once

#include "../qwen35/Qwen35Types.cuh"

#include <cstddef>
#include <cuda_runtime.h>

class LinearAttention {
public:
    template<typename hidden_t, typename weight_t, typename compute_t = float>
    static void conv1d_silu(
        const hidden_t *qkv,
        hidden_t *conv_state,
        const weight_t *conv_weight,
        const weight_t *conv_bias,
        hidden_t *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size,
        cudaStream_t stream);

    template<typename hidden_t, typename compute_t = float>
    static void split_qkv(
        const hidden_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim,
        cudaStream_t stream);

    template<typename hidden_t, typename compute_t = float>
    static void normalize_mixed_qk(
        hidden_t *mixed_qkv,
        size_t num_key_heads,
        size_t key_head_dim,
        cudaStream_t stream);

    template<typename hidden_t, typename weight_t, typename compute_t = float>
    static void gated_delta_update(
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
        cudaStream_t stream);
};

extern template void LinearAttention::conv1d_silu<float, float, float>(
    const float*, float*, const float*, const float*, float*, size_t, size_t, cudaStream_t);
extern template void LinearAttention::split_qkv<float, float>(
    const float*, float*, float*, float*, size_t, size_t, size_t, size_t, cudaStream_t);
extern template void LinearAttention::normalize_mixed_qk<float, float>(
    float*, size_t, size_t, cudaStream_t);
extern template void LinearAttention::gated_delta_update<float, float, float>(
    float*, const float*, const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t, size_t, cudaStream_t);
