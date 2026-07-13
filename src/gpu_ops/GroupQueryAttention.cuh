#pragma once

#include "../qwen35/Qwen35Config.h"
#include "../qwen35/Qwen35Types.cuh"
#include "../CudaBuffer.cuh"
#include <cstdint>
#include <memory>

template<Qwen35Size QWEN35_SIZE>
class GroupQueryAttention {
public:
    using Qwen35Config = Qwen35Config<QWEN35_SIZE>;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len) {}

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V,
     * then applies the Qwen35 per-element output gate.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, head_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, head_size)
     * @param weighted_values (num_query_heads, head_size) outputs
     * @param gate (num_query_heads, head_size)
     * @param layer_num layer index, starting at 0
     * @param token_pos current token position in the cache
     * @param stream CUDA stream for asynchronous operation
     */
    template<typename query_t, typename key_t, typename value_t, typename output_t, typename gate_t, typename compute_t = float>
    static void sdpa(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, const gate_t *gate, int32_t layer_num, int32_t token_pos, cudaStream_t stream);
};

extern template void GroupQueryAttention<QWEN35_0_8B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
extern template void GroupQueryAttention<QWEN35_2B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
extern template void GroupQueryAttention<QWEN35_4B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
extern template void GroupQueryAttention<QWEN35_9B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
