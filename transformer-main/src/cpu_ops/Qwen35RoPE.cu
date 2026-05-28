#include "Qwen35RoPE.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

static void apply_partial_rope_to_row(float *row, size_t dim, size_t rotary_dim, size_t position_idx, float theta_base) {
    rotary_dim = std::min(rotary_dim, dim);
    size_t half = rotary_dim / 2;
    for (size_t i = 0; i < half; i++) {
        float theta = 1.0f / std::pow(theta_base, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        float angle = static_cast<float>(position_idx) * theta;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float x1 = row[i];
        float x2 = row[i + half];
        row[i] = x1 * c - x2 * s;
        row[i + half] = x2 * c + x1 * s;
    }
}

void Qwen35RoPE::apply_partial_rope_to_qk(
        input_float_t *queries,
        size_t num_query_heads,
        input_float_t *keys,
        size_t num_kv_heads,
        size_t head_size,
        size_t rotary_dim,
        size_t position_idx,
        float theta_base) {
    auto apply = [&] (input_float_t *values, size_t num_heads) {
        for (size_t h = 0; h < num_heads; h++) {
            float row[256];
            if (head_size > 256) {
                throw std::runtime_error("Qwen35 CPU RoPE scratch is too small");
            }
            for (size_t d = 0; d < head_size; d++) {
                row[d] = normalize_input_float(values[h * head_size + d]);
            }
            apply_partial_rope_to_row(row, head_size, rotary_dim, position_idx, theta_base);
            for (size_t d = 0; d < head_size; d++) {
                values[h * head_size + d] = input_float_from_float(row[d]);
            }
        }
    };

    apply(queries, num_query_heads);
    apply(keys, num_kv_heads);
}
