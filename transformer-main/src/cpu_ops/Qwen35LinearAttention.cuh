#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35LinearAttention {
public:
    static void conv1d_silu(
        const input_float_t *qkv,
        input_float_t *conv_state,
        const input_float_t *conv_weight,
        const input_float_t *conv_bias,
        input_float_t *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size);

    static void split_qkv(
        const input_float_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim);

    static void gated_delta_update(
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
        size_t value_head_dim);
};
