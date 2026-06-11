#include "GroupQueryAttention.cuh"
#include "Math.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

void CpuGroupQueryAttention::sdpa(
        float *queries,
        float *keys_cache,
        float *values_cache,
        float *weighted_values,
        float *gate,
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
                score += static_cast<float>(queries[qh * head_size + d]) * static_cast<float>(key[d]);
            }
            max_score = std::max(max_score, score / std::sqrt(static_cast<float>(head_size)));
        }

        float denom = 0.0f;
        for (size_t d = 0; d < head_size; d++) {
            weighted_values[qh * head_size + d] = static_cast<float>(0.0f);
        }
        for (size_t t = 0; t <= seq_len; t++) {
            auto key = keys_cache + t * num_layers * keys_size + layer_num * keys_size + kvh * head_size;
            auto value = values_cache + t * num_layers * values_size + layer_num * values_size + kvh * head_size;
            float score = 0.0f;
            for (size_t d = 0; d < head_size; d++) {
                score += static_cast<float>(queries[qh * head_size + d]) * static_cast<float>(key[d]);
            }
            float coeff = std::exp(score / std::sqrt(static_cast<float>(head_size)) - max_score);
            denom += coeff;
            for (size_t d = 0; d < head_size; d++) {
                float current = static_cast<float>(weighted_values[qh * head_size + d]);
                weighted_values[qh * head_size + d] = static_cast<float>(current + coeff * static_cast<float>(value[d]));
            }
        }
        for (size_t d = 0; d < head_size; d++) {
            size_t idx = qh * head_size + d;
            float v = static_cast<float>(weighted_values[idx]) / denom;
            v *= cpu_sigmoid(static_cast<float>(gate[idx]));
            weighted_values[idx] = static_cast<float>(v);
        }
    }
}
