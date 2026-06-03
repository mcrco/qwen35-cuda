#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/Qwen2GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../gpu_ops/GpuFloat.cuh"

namespace qwen2_layer_detail {

constexpr int RES_ADD_THREADS = 128;
constexpr int RES_ADD_MAX_BLOCKS = 1024;

template<typename residual_t, typename value_t, typename compute_t>
__global__ void residualAddKernel(residual_t *residual, const value_t *values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        compute_t rval = gpu_ops::read_as<compute_t>(residual[i]);
        compute_t vval = gpu_ops::read_as<compute_t>(values[i]);
        residual[i] = gpu_ops::write_from<residual_t>(rval + vval);
    }
}

template<typename src_t, typename dst_t, typename compute_t>
__global__ void convertCopyKernel(const src_t *src, dst_t *dst, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        dst[i] = gpu_ops::write_from<dst_t>(gpu_ops::read_as<compute_t>(src[i]));
    }
}

template<typename src_t, typename dst_t, typename compute_t>
inline void convert_copy(const src_t *src, dst_t *dst, int n, cudaStream_t stream) {
    int threads = RES_ADD_THREADS;
    int blocks = min(RES_ADD_MAX_BLOCKS, (n + threads - 1) / threads);
    convertCopyKernel<src_t, dst_t, compute_t><<<blocks, threads, 0, stream>>>(src, dst, n);
    checkCuda(cudaGetLastError());
}

template<typename residual_t, typename value_t, typename compute_t>
inline void residual_add(residual_t *residual, const value_t *values, int n, cudaStream_t stream) {
    int threads = RES_ADD_THREADS;
    int blocks = min(RES_ADD_MAX_BLOCKS, (n + threads - 1) / threads);
    residualAddKernel<residual_t, value_t, compute_t><<<blocks, threads, 0, stream>>>(residual, values, n);
    checkCuda(cudaGetLastError());
}

} // namespace qwen2_layer_detail

template<
    Qwen2Size QWEN2_SIZE,
    typename weight_t = __nv_bfloat16,
    typename hidden_t = __nv_bfloat16,
    typename compute_t = float,
    typename cache_t = hidden_t>
class Qwen2Layer {
    std::shared_ptr<CudaBuffer> norm_hidden_state;
    std::shared_ptr<CudaBuffer> queries;
    std::shared_ptr<CudaBuffer> attention_output;
    std::shared_ptr<CudaBuffer> attention_proj;
    std::shared_ptr<CudaBuffer> gate_proj;
    std::shared_ptr<CudaBuffer> up_proj;
    std::shared_ptr<CudaBuffer> down_proj;
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    using WeightT = weight_t;
    using HiddenT = hidden_t;
    using ComputeT = compute_t;
    using CacheT = cache_t;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
        layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()) {
        norm_hidden_state = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(hidden_t));
        queries = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(hidden_t));
        attention_output = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(compute_t));
        attention_proj = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(hidden_t));
        gate_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(hidden_t));
        up_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(hidden_t));
        down_proj = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(hidden_t));
    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, intermediate_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)

    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        input_layernorm.normalize_hidden_state<hidden_t, weight_t, hidden_t, compute_t>(hidden_state, norm_hidden_state, Qwen2Config::hidden_size(), stream);

        const weight_t *q_proj_weight_ptr = static_cast<const weight_t *>(q_proj_weight->data);
        const weight_t *q_proj_bias_ptr = q_proj_bias ? static_cast<const weight_t *>(q_proj_bias->data) : nullptr;
        const weight_t *k_proj_weight_ptr = static_cast<const weight_t *>(k_proj_weight->data);
        const weight_t *k_proj_bias_ptr = k_proj_bias ? static_cast<const weight_t *>(k_proj_bias->data) : nullptr;
        const weight_t *v_proj_weight_ptr = static_cast<const weight_t *>(v_proj_weight->data);
        const weight_t *v_proj_bias_ptr = v_proj_bias ? static_cast<const weight_t *>(v_proj_bias->data) : nullptr;
        cache_t *k_cache_ptr = static_cast<cache_t *>(k_cache->data);
        cache_t *v_cache_ptr = static_cast<cache_t *>(v_cache->data);
        hidden_t *hidden_state_ptr = static_cast<hidden_t *>(hidden_state->data);
        hidden_t *norm_hidden_state_ptr = static_cast<hidden_t *>(norm_hidden_state->data);
        hidden_t *queries_ptr = static_cast<hidden_t *>(queries->data);

        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t, compute_t>(
            Qwen2Config::queries_size(), Qwen2Config::hidden_size(), q_proj_weight_ptr, q_proj_bias_ptr, norm_hidden_state_ptr, queries_ptr, stream);

        int token_pos = seq_len - 1;
        cache_t *keys_ptr = k_cache_ptr
            + token_pos * Qwen2Config::num_layers() * Qwen2Config::keys_size()
            + layer_num * Qwen2Config::keys_size();
        cache_t *vals_ptr = v_cache_ptr
            + token_pos * Qwen2Config::num_layers() * Qwen2Config::values_size()
            + layer_num * Qwen2Config::values_size();

        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, cache_t, compute_t>(
            Qwen2Config::keys_size(), Qwen2Config::hidden_size(), k_proj_weight_ptr, k_proj_bias_ptr, norm_hidden_state_ptr, keys_ptr, stream);
        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, cache_t, compute_t>(
            Qwen2Config::values_size(), Qwen2Config::hidden_size(), v_proj_weight_ptr, v_proj_bias_ptr, norm_hidden_state_ptr, vals_ptr, stream);

        RoPE::apply_rope_to_qk<hidden_t, compute_t>(queries_ptr, Qwen2Config::num_query_heads(), Qwen2Config::head_size(), Qwen2Config::rotary_dim(), token_pos, Qwen2Config::rope_theta_base(), stream);
        RoPE::apply_rope_to_qk<cache_t, compute_t>(keys_ptr, Qwen2Config::num_kv_heads(), Qwen2Config::head_size(), Qwen2Config::rotary_dim(), token_pos, Qwen2Config::rope_theta_base(), stream);

        compute_t *attention_output_ptr = static_cast<compute_t *>(attention_output->data);
        checkCuda(cudaMemsetAsync(attention_output->data, 0, attention_output->size, stream));
        Qwen2GroupQueryAttention<QWEN2_SIZE>::template sdpa<hidden_t, cache_t, cache_t, compute_t, compute_t>(
            queries_ptr, k_cache_ptr, v_cache_ptr, attention_output_ptr, layer_num, seq_len, stream);

        const weight_t *o_proj_weight_ptr = static_cast<const weight_t*>(o_proj_weight->data);
        hidden_t *attention_proj_ptr = static_cast<hidden_t *>(attention_proj->data);
        MatrixVectorMultiply::matmul<weight_t, weight_t, compute_t, hidden_t, compute_t>(
            Qwen2Config::hidden_size(), Qwen2Config::queries_size(), o_proj_weight_ptr, nullptr, attention_output_ptr, attention_proj_ptr, stream);

        qwen2_layer_detail::residual_add<hidden_t, hidden_t, compute_t>(hidden_state_ptr, attention_proj_ptr, Qwen2Config::hidden_size(), stream);

        post_attention_layernorm.normalize_hidden_state<hidden_t, weight_t, hidden_t, compute_t>(hidden_state, norm_hidden_state, Qwen2Config::hidden_size(), stream);

        const weight_t *gate_proj_weight_ptr = static_cast<const weight_t *>(gate_proj_weight->data);
        const weight_t *up_proj_weight_ptr = static_cast<const weight_t *>(up_proj_weight->data);
        const weight_t *down_proj_weight_ptr = static_cast<const weight_t *>(down_proj_weight->data);
        hidden_t *gate_proj_ptr = static_cast<hidden_t *>(gate_proj->data);
        hidden_t *up_proj_ptr = static_cast<hidden_t *>(up_proj->data);
        hidden_t *down_proj_ptr = static_cast<hidden_t *>(down_proj->data);

        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t, compute_t>(
            Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(), gate_proj_weight_ptr, nullptr, norm_hidden_state_ptr, gate_proj_ptr, stream);
        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t, compute_t>(
            Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(), up_proj_weight_ptr, nullptr, norm_hidden_state_ptr, up_proj_ptr, stream);
        SiLUMult::silu_mult_in_place<hidden_t, hidden_t, compute_t>(gate_proj, up_proj, Qwen2Config::intermediate_size(), stream);
        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t, compute_t>(
            Qwen2Config::hidden_size(), Qwen2Config::intermediate_size(), down_proj_weight_ptr, nullptr, gate_proj_ptr, down_proj_ptr, stream);

        qwen2_layer_detail::residual_add<hidden_t, hidden_t, compute_t>(hidden_state_ptr, down_proj_ptr, Qwen2Config::hidden_size(), stream);
    }
};
