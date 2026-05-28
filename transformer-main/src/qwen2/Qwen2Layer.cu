#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "Qwen2Layer.cuh"
#include "../CudaBuffer.cuh"
#include <fcntl.h>
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

const int RES_ADD_THREADS = 128;
const int RES_ADD_MAX_BLOCKS = 1024;

__device__ inline float normalize_float(float x) {
    return x;
}

__device__ inline float normalize_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename value_t>
__global__ void residualAddKernel(__nv_bfloat16 *residual, value_t *values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float rval = normalize_float(residual[i]);
        float vval = __bfloat162float(values[i]);
        residual[i] = __float2bfloat16(rval + vval);
    }
}

template<Qwen2Size QWEN2_SIZE>
Qwen2Layer<QWEN2_SIZE>::Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()) {
    norm_hidden_state = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
    queries = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
    attention_output = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(float));
    attention_proj = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
    gate_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
    up_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
    down_proj = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
}

template<Qwen2Size QWEN2_SIZE>
void Qwen2Layer<QWEN2_SIZE>::forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
    // First layernorm
    input_layernorm.normalize_hidden_state(hidden_state, norm_hidden_state, stream);

    // QKV projections.
    // First get pointers from buffers.
    __nv_bfloat16 *q_proj_weight_ptr = static_cast<__nv_bfloat16 *>(q_proj_weight->data);
    __nv_bfloat16 *q_proj_bias_ptr = static_cast<__nv_bfloat16 *>(q_proj_bias->data);
    __nv_bfloat16 *k_proj_weight_ptr = static_cast<__nv_bfloat16 *>(k_proj_weight->data);
    __nv_bfloat16 *k_proj_bias_ptr = static_cast<__nv_bfloat16 *>(k_proj_bias->data);
    __nv_bfloat16 *v_proj_weight_ptr = static_cast<__nv_bfloat16 *>(v_proj_weight->data);
    __nv_bfloat16 *v_proj_bias_ptr = static_cast<__nv_bfloat16 *>(v_proj_bias->data);
    __nv_bfloat16 *k_cache_ptr = static_cast<__nv_bfloat16 *>(k_cache->data);
    __nv_bfloat16 *v_cache_ptr = static_cast<__nv_bfloat16 *>(v_cache->data);
    __nv_bfloat16 *hidden_state_ptr = static_cast<__nv_bfloat16 *>(hidden_state->data);
    __nv_bfloat16 *norm_hidden_state_ptr = static_cast<__nv_bfloat16 *>(norm_hidden_state->data);
    __nv_bfloat16 *queries_ptr = static_cast<__nv_bfloat16 *>(queries->data);

    // Query projection.
    // 14 query heads * 64 query dim = 896 query elements
    // => we should have 1 row/1 output value per query element
    // => m = queries_size = num_query_heads * query_size, k = hidden_size
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::queries_size(), Qwen2Config::hidden_size(), q_proj_weight_ptr, q_proj_bias_ptr, norm_hidden_state_ptr, queries_ptr, stream);

    // Compute where current token's KV vectors should be stored in KV cache as
    // a flattened index. We increment seq_len in main.cu before forward(), so
    // our current token pos should actually be seq_len - 1.
    int token_pos = seq_len - 1;
    __nv_bfloat16 *keys_ptr = k_cache_ptr
        + token_pos * Qwen2Config::num_layers() * Qwen2Config::keys_size() // offset from previous tokens
        + layer_num * Qwen2Config::keys_size(); // offset from layers for current token
    __nv_bfloat16 *vals_ptr = v_cache_ptr +
        token_pos * Qwen2Config::num_layers() * Qwen2Config::values_size() // offset from previous tokens
        + layer_num * Qwen2Config::values_size(); // offset from layers for current token

    // Compute and store KV projections in cache.
    // Like before, one row per kv element.
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::keys_size(), Qwen2Config::hidden_size(), k_proj_weight_ptr, k_proj_bias_ptr, norm_hidden_state_ptr, keys_ptr, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::values_size(), Qwen2Config::hidden_size(), v_proj_weight_ptr, v_proj_bias_ptr, norm_hidden_state_ptr, vals_ptr, stream);

    // Add RoPE embeddings in-place to queries and keys.
    RoPE::apply_rope_to_qk(queries_ptr, Qwen2Config::num_query_heads(), Qwen2Config::head_size(), token_pos, Qwen2Config::rope_theta_base(), stream);
    RoPE::apply_rope_to_qk(keys_ptr, Qwen2Config::num_kv_heads(), Qwen2Config::head_size(), token_pos, Qwen2Config::rope_theta_base(), stream);

    // Self-attention.
    // Get pointer from cuda buffer.
    float *attention_output_ptr = static_cast<float *>(attention_output->data);
    // Set output values to 0
    checkCuda(cudaMemsetAsync(
        attention_output->data,
        0,
        attention_output->size,
        stream
    ));
    GroupQueryAttention<QWEN2_SIZE>::sdpa(queries_ptr, k_cache_ptr, v_cache_ptr, attention_output_ptr, layer_num, seq_len, stream);

    // Output projection of attention before hidden state.
    __nv_bfloat16 *o_proj_weight_ptr = static_cast<__nv_bfloat16*>(o_proj_weight->data);
    __nv_bfloat16 *attention_proj_ptr = static_cast<__nv_bfloat16 *>(attention_proj->data);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::queries_size(), o_proj_weight_ptr, nullptr, attention_output_ptr, attention_proj_ptr, stream);

    // Residual connection
    int threads = RES_ADD_THREADS;
    int blocks = min(RES_ADD_MAX_BLOCKS, ((int)Qwen2Config::hidden_size() + threads - 1) / threads);
    residualAddKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(hidden_state_ptr, attention_proj_ptr, Qwen2Config::hidden_size());
    checkCuda(cudaGetLastError());

    // Layernorm after attention + residual connection.
    post_attention_layernorm.normalize_hidden_state(hidden_state, norm_hidden_state, stream);

    // Final MLP.
    // Get pointers from buffers.
    __nv_bfloat16 *gate_proj_weight_ptr = static_cast<__nv_bfloat16 *>(gate_proj_weight->data);
    __nv_bfloat16 *up_proj_weight_ptr = static_cast<__nv_bfloat16 *>(up_proj_weight->data);
    __nv_bfloat16 *down_proj_weight_ptr = static_cast<__nv_bfloat16 *>(down_proj_weight->data);
    __nv_bfloat16 *gate_proj_ptr = static_cast<__nv_bfloat16 *>(gate_proj->data);
    __nv_bfloat16 *up_proj_ptr = static_cast<__nv_bfloat16 *>(up_proj->data);
    __nv_bfloat16 *down_proj_ptr = static_cast<__nv_bfloat16 *>(down_proj->data);

    // Up/gate projections should result in a higher dimension (intermediate_size, 1) vector
    // => m = intermediate_size, k = hidden_size
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::intermediate_size(),
    Qwen2Config::hidden_size(), gate_proj_weight_ptr, nullptr, norm_hidden_state_ptr,
    gate_proj_ptr, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(), up_proj_weight_ptr, nullptr, norm_hidden_state_ptr, up_proj_ptr, stream);
    SiLUMult::silu_mult_in_place(gate_proj, up_proj, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::intermediate_size(), down_proj_weight_ptr, nullptr, gate_proj_ptr, down_proj_ptr, stream);

    // Add back to hidden state via residual.
    residualAddKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(hidden_state_ptr, down_proj_ptr, Qwen2Config::hidden_size());
    checkCuda(cudaGetLastError());
}

template class Qwen2Layer<QWEN2_0_5B>;
