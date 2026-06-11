#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35GroupQueryAttention {
public:
    static void sdpa(
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
        size_t values_size);
};
