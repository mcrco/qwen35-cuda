#pragma once

#include <memory>

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/GpuFloat.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "Qwen35Config.h"
#include "Qwen35Types.cuh"

namespace qwen35_layer_detail {

constexpr int VECTOR_THREADS = 128;
constexpr int VECTOR_MAX_BLOCKS = 1024;

template<typename residual_t, typename value_t, typename compute_t>
__global__ void residualAddKernel(residual_t *residual, const value_t *values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        compute_t sum = gpu_ops::read_as<compute_t>(residual[i]) + gpu_ops::read_as<compute_t>(values[i]);
        residual[i] = gpu_ops::write_from<residual_t>(sum);
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
    int threads = VECTOR_THREADS;
    int blocks = min(VECTOR_MAX_BLOCKS, (n + threads - 1) / threads);
    convertCopyKernel<src_t, dst_t, compute_t><<<blocks, threads, 0, stream>>>(src, dst, n);
    checkCuda(cudaGetLastError());
}

template<typename residual_t, typename value_t, typename compute_t>
inline void residual_add(residual_t *residual, const value_t *values, int n, cudaStream_t stream) {
    int threads = VECTOR_THREADS;
    int blocks = min(VECTOR_MAX_BLOCKS, (n + threads - 1) / threads);
    residualAddKernel<residual_t, value_t, compute_t><<<blocks, threads, 0, stream>>>(residual, values, n);
    checkCuda(cudaGetLastError());
}

} // namespace qwen35_layer_detail

struct Qwen35Cache {
    std::shared_ptr<CudaBuffer> keys;
    std::shared_ptr<CudaBuffer> values;
    std::shared_ptr<CudaBuffer> conv_states;
    std::shared_ptr<CudaBuffer> recurrent_states;
    size_t seq_len{};
};

template<
    Qwen35Size QWEN35_SIZE,
    typename weight_t = float,
    typename hidden_t = float,
    typename compute_t = float>
class Qwen35Layer {
public:
    explicit Qwen35Layer(size_t layer_num);
    virtual ~Qwen35Layer() = default;

    size_t layer_num{};
    LayerNorm input_layernorm;
    LayerNorm post_attention_layernorm;
    std::shared_ptr<CudaBuffer> up_proj_weight;
    std::shared_ptr<CudaBuffer> gate_proj_weight;
    std::shared_ptr<CudaBuffer> down_proj_weight;

    virtual void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state, cudaStream_t stream) = 0;

protected:
    std::shared_ptr<CudaBuffer> norm_hidden_state;
    std::shared_ptr<CudaBuffer> ffn_gate;
    std::shared_ptr<CudaBuffer> ffn_up;
    std::shared_ptr<CudaBuffer> ffn_down;

    void apply_mlp(const std::shared_ptr<CudaBuffer> &hidden_state, cudaStream_t stream);
};

template<
    Qwen35Size QWEN35_SIZE,
    typename weight_t = float,
    typename hidden_t = float,
    typename compute_t = float>
class Qwen35FullAttnLayer final : public Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t> {
public:
    explicit Qwen35FullAttnLayer(size_t layer_num);

    std::shared_ptr<CudaBuffer> q_proj_weight;
    std::shared_ptr<CudaBuffer> q_proj_bias;
    std::shared_ptr<CudaBuffer> k_proj_weight;
    std::shared_ptr<CudaBuffer> k_proj_bias;
    std::shared_ptr<CudaBuffer> v_proj_weight;
    std::shared_ptr<CudaBuffer> v_proj_bias;
    std::shared_ptr<CudaBuffer> o_proj_weight;
    std::shared_ptr<CudaBuffer> o_proj_bias;
    LayerNorm q_norm;
    LayerNorm k_norm;

    void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state, cudaStream_t stream) override;

private:
    using Base = Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t>;
    using Base::layer_num;
    using Base::norm_hidden_state;
    using Base::input_layernorm;
    using Base::apply_mlp;

    std::shared_ptr<CudaBuffer> q_proj;
    std::shared_ptr<CudaBuffer> queries;
    std::shared_ptr<CudaBuffer> gate;
    std::shared_ptr<CudaBuffer> attention_output;
    std::shared_ptr<CudaBuffer> attention_proj;
};

template<
    Qwen35Size QWEN35_SIZE,
    typename weight_t = float,
    typename hidden_t = float,
    typename compute_t = float>
class Qwen35LinearAttentionLayer final : public Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t> {
public:
    explicit Qwen35LinearAttentionLayer(size_t layer_num);

    std::shared_ptr<CudaBuffer> in_proj_qkv_weight;
    std::shared_ptr<CudaBuffer> in_proj_z_weight;
    std::shared_ptr<CudaBuffer> in_proj_b_weight;
    std::shared_ptr<CudaBuffer> in_proj_a_weight;
    std::shared_ptr<CudaBuffer> conv1d_weight;
    std::shared_ptr<CudaBuffer> conv1d_bias;
    std::shared_ptr<CudaBuffer> dt_bias;
    std::shared_ptr<CudaBuffer> A_log;
    LayerNorm norm;
    std::shared_ptr<CudaBuffer> out_proj_weight;
    std::shared_ptr<CudaBuffer> out_proj_bias;

    void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state, cudaStream_t stream) override;

private:
    using Base = Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t>;
    using Base::layer_num;
    using Base::norm_hidden_state;
    using Base::input_layernorm;
    using Base::apply_mlp;

    std::shared_ptr<CudaBuffer> qkv;
    std::shared_ptr<CudaBuffer> gates;
    std::shared_ptr<CudaBuffer> beta_raw;
    std::shared_ptr<CudaBuffer> decay_raw;
    std::shared_ptr<CudaBuffer> mixed_qkv;
    std::shared_ptr<CudaBuffer> queries_float;
    std::shared_ptr<CudaBuffer> keys_float;
    std::shared_ptr<CudaBuffer> values_float;
    std::shared_ptr<CudaBuffer> weighted_values_float;
    std::shared_ptr<CudaBuffer> gated_weighted_values;
    std::shared_ptr<CudaBuffer> attention_proj;
};
