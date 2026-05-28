#include "Qwen35LinearAttention.cuh"
#include "BufferOps.cuh"
#include "Qwen35Math.cuh"

#include <cmath>

void Qwen35LinearAttention::conv1d_silu(
        const input_float_t *qkv,
        input_float_t *conv_state,
        const input_float_t *conv_weight,
        const input_float_t *conv_bias,
        input_float_t *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size) {
    for (size_t k = 0; k + 1 < conv_kernel_dim; k++) {
        BufferOps::copy(conv_state + (k + 1) * conv_size, conv_state + k * conv_size, conv_size);
    }
    BufferOps::copy(qkv, conv_state + (conv_kernel_dim - 1) * conv_size, conv_size);

    for (size_t c = 0; c < conv_size; c++) {
        float sum = conv_bias ? normalize_input_float(conv_bias[c]) : 0.0f;
        for (size_t k = 0; k < conv_kernel_dim; k++) {
            sum += normalize_input_float(conv_state[k * conv_size + c]) * normalize_input_float(conv_weight[c * conv_kernel_dim + k]);
        }
        mixed_qkv[c] = input_float_from_float(qwen35_silu(sum));
    }
}

void Qwen35LinearAttention::split_qkv(
        const input_float_t *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim) {
    size_t group_size = num_value_heads / num_key_heads;
    size_t keys_size = num_key_heads * key_head_dim;
    for (size_t vh = 0; vh < num_value_heads; vh++) {
        size_t kh = vh / group_size;
        for (size_t d = 0; d < key_head_dim; d++) {
            queries[vh * key_head_dim + d] = normalize_input_float(mixed_qkv[kh * key_head_dim + d]);
            keys[vh * key_head_dim + d] = normalize_input_float(mixed_qkv[keys_size + kh * key_head_dim + d]);
        }
        for (size_t d = 0; d < value_head_dim; d++) {
            values[vh * value_head_dim + d] = normalize_input_float(mixed_qkv[2 * keys_size + vh * value_head_dim + d]);
        }
    }
}

void Qwen35LinearAttention::gated_delta_update(
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
        size_t value_head_dim) {
    BufferOps::zero_float(weighted_values, num_value_heads * value_head_dim);

    for (size_t h = 0; h < num_value_heads; h++) {
        float beta = qwen35_sigmoid(normalize_input_float(beta_raw[h]));
        float decay = -std::exp(normalize_input_float(A_log[h])) * qwen35_softplus(normalize_input_float(decay_raw[h]) + normalize_input_float(dt_bias[h]));
        float decay_coeff = std::exp(decay);

        for (size_t k = 0; k < key_head_dim; k++) {
            for (size_t v = 0; v < value_head_dim; v++) {
                size_t idx = h * key_head_dim * value_head_dim + k * value_head_dim + v;
                state[idx] = input_float_from_float(normalize_input_float(state[idx]) * decay_coeff);
            }
        }

        for (size_t v = 0; v < value_head_dim; v++) {
            float predicted = 0.0f;
            for (size_t k = 0; k < key_head_dim; k++) {
                size_t idx = h * key_head_dim * value_head_dim + k * value_head_dim + v;
                predicted += normalize_input_float(state[idx]) * keys[h * key_head_dim + k];
            }
            float delta = (values[h * value_head_dim + v] - predicted) * beta;
            for (size_t k = 0; k < key_head_dim; k++) {
                size_t idx = h * key_head_dim * value_head_dim + k * value_head_dim + v;
                float updated = normalize_input_float(state[idx]) + keys[h * key_head_dim + k] * delta;
                state[idx] = input_float_from_float(updated);
            }
        }

        for (size_t v = 0; v < value_head_dim; v++) {
            float out = 0.0f;
            for (size_t k = 0; k < key_head_dim; k++) {
                size_t idx = h * key_head_dim * value_head_dim + k * value_head_dim + v;
                out += normalize_input_float(state[idx]) * queries[h * key_head_dim + k];
            }
            weighted_values[h * value_head_dim + v] = out;
        }
    }
}
