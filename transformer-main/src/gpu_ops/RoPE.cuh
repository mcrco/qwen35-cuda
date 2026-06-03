#pragma once
#include "../qwen35/Qwen35Types.cuh"
#include <memory>
#include <cuda_bf16.h>
#include <cstdint>

/**
 *  GPT-NeoX Style Rotary Positional Embeddings, see https://nn.labml.ai/transformers/rope/index.html
 */
class RoPE {
public:
    /**
     * Apply RoPE in-place
     * @param x queries or keys, of shape (num_heads, head_dim)
     * @param num_heads Number of query/key heads
     * @param head_dim Elements per query/key
     * @param rotary_dim Number of leading head elements to rotate
     * @param position_idx Current position in sequence
     * @param theta_base RoPE parameter
     * @param stream CUDA stream for asynchronous operation
     */
    template<typename x_t, typename compute_t = float>
    static void apply_rope_to_qk(x_t *x, int32_t num_heads, int32_t head_dim, int32_t rotary_dim,
                                 int32_t position_idx, compute_t theta_base, cudaStream_t stream);

    static void apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t rotary_dim,
                                 int32_t position_idx, float theta_base, cudaStream_t stream);
};

extern template void RoPE::apply_rope_to_qk<__nv_bfloat16, float>(__nv_bfloat16*, int32_t, int32_t, int32_t, int32_t, float, cudaStream_t);
extern template void RoPE::apply_rope_to_qk<float, float>(float*, int32_t, int32_t, int32_t, int32_t, float, cudaStream_t);
extern template void RoPE::apply_rope_to_qk<input_float_t, float>(input_float_t*, int32_t, int32_t, int32_t, int32_t, float, cudaStream_t);
