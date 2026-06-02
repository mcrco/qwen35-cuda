#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <memory>

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len) {}

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V
     * - dot product each query with its target value throughout the sequence
     * - numerically stable softmax
     * - save a weighted sum of values
     * Does not perform the output projection.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, value_size)
     * @param weighted_values (num_query_heads, value_size) outputs
     * @param layer_num layer index, starting at 0
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    template<typename query_t, typename key_t, typename value_t, typename output_t, typename compute_t = float>
    static void sdpa(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream);

    static void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream);
};

extern template void GroupQueryAttention<QWEN2_0_5B>::sdpa<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, float>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float*, int32_t, int32_t, cudaStream_t);
extern template void GroupQueryAttention<QWEN2_0_5B>::sdpa<float, float, float, float, float>(
    const float*, const float*, const float*, float*, int32_t, int32_t, cudaStream_t);
