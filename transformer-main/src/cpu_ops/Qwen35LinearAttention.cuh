#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35LinearAttention {
public:
    static void conv1d_silu(
        const float *qkv,
        float *conv_state,
        const float *conv_weight,
        const float *conv_bias,
        float *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size);

    static void split_qkv(
        const float *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim);

    static void gated_delta_update(
        float *state,
        const float *queries,
        const float *keys,
        const float *values,
        const float *beta_raw,
        const float *decay_raw,
        const float *dt_bias,
        const float *A_log,
        float *weighted_values,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim);
};
