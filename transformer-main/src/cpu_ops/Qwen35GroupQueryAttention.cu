#include "Qwen35GroupQueryAttention.cuh"
#include "Qwen35Math.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

void Qwen35GroupQueryAttention::sdpa(
        input_float_t *queries,
        input_float_t *keys_cache,
        input_float_t *values_cache,
        input_float_t *weighted_values,
        input_float_t *gate,
        size_t layer_num,
        size_t seq_len,
        size_t num_layers,
        size_t num_query_heads,
        size_t num_kv_heads,
        size_t head_size,
        size_t keys_size,
        size_t values_size) {
    size_t group_size = num_query_heads / num_kv_heads;
    for (size_t qh = 0; qh < num_query_heads; qh++) {
        size_t kvh = qh / group_size;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t t = 0; t <= seq_len; t++) {
            auto key = keys_cache + t * num_layers * keys_size + layer_num * keys_size + kvh * head_size;
            float score = 0.0f;
            for (size_t d = 0; d < head_size; d++) {
                score += normalize_input_float(queries[qh * head_size + d]) * normalize_input_float(key[d]);
            }
            max_score = std::max(max_score, score / std::sqrt(static_cast<float>(head_size)));
        }

        float denom = 0.0f;
        for (size_t d = 0; d < head_size; d++) {
            weighted_values[qh * head_size + d] = input_float_from_float(0.0f);
        }
        for (size_t t = 0; t <= seq_len; t++) {
            auto key = keys_cache + t * num_layers * keys_size + layer_num * keys_size + kvh * head_size;
            auto value = values_cache + t * num_layers * values_size + layer_num * values_size + kvh * head_size;
            float score = 0.0f;
            for (size_t d = 0; d < head_size; d++) {
                score += normalize_input_float(queries[qh * head_size + d]) * normalize_input_float(key[d]);
            }
            float coeff = std::exp(score / std::sqrt(static_cast<float>(head_size)) - max_score);
            denom += coeff;
            for (size_t d = 0; d < head_size; d++) {
                float current = normalize_input_float(weighted_values[qh * head_size + d]);
                weighted_values[qh * head_size + d] = input_float_from_float(current + coeff * normalize_input_float(value[d]));
            }
        }
        for (size_t d = 0; d < head_size; d++) {
            size_t idx = qh * head_size + d;
            float v = normalize_input_float(weighted_values[idx]) / denom;
            v *= qwen35_sigmoid(normalize_input_float(gate[idx]));
            weighted_values[idx] = input_float_from_float(v);
        }
    }
}
