#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35RoPE {
public:
    static void apply_partial_rope_to_qk(
        input_float_t *queries,
        size_t num_query_heads,
        input_float_t *keys,
        size_t num_kv_heads,
        size_t head_size,
        size_t rotary_dim,
        size_t position_idx,
        float theta_base);
};
