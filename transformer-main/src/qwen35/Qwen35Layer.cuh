#pragma once

#include <memory>

#include "../CudaBuffer.cuh"
#include "Qwen35Config.h"
#include "Qwen35Types.cuh"

struct Qwen35Cache {
    std::shared_ptr<CudaBuffer> keys;
    std::shared_ptr<CudaBuffer> values;
    std::shared_ptr<CudaBuffer> conv_states;
    std::shared_ptr<CudaBuffer> recurrent_states;
    size_t seq_len{};
};

template<Qwen35Size QWEN35_SIZE>
class Qwen35Layer {
public:
    explicit Qwen35Layer(size_t layer_num);
    virtual ~Qwen35Layer() = default;

    size_t layer_num{};
    std::shared_ptr<CudaBuffer> input_layernorm;
    std::shared_ptr<CudaBuffer> post_attention_layernorm;
    std::shared_ptr<CudaBuffer> up_proj_weight;
    std::shared_ptr<CudaBuffer> gate_proj_weight;
    std::shared_ptr<CudaBuffer> down_proj_weight;

    virtual void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state) = 0;

protected:
    std::shared_ptr<CudaBuffer> norm_hidden_state;
    std::shared_ptr<CudaBuffer> ffn_gate;
    std::shared_ptr<CudaBuffer> ffn_up;
    std::shared_ptr<CudaBuffer> ffn_down;

    void apply_mlp(const std::shared_ptr<CudaBuffer> &hidden_state);
};

template<Qwen35Size QWEN35_SIZE>
class Qwen35FullAttnLayer final : public Qwen35Layer<QWEN35_SIZE> {
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
    std::shared_ptr<CudaBuffer> q_norm_weight;
    std::shared_ptr<CudaBuffer> k_norm_weight;

    void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state) override;

private:
    using Qwen35Layer<QWEN35_SIZE>::layer_num;
    using Qwen35Layer<QWEN35_SIZE>::norm_hidden_state;
    using Qwen35Layer<QWEN35_SIZE>::input_layernorm;
    using Qwen35Layer<QWEN35_SIZE>::apply_mlp;

    std::shared_ptr<CudaBuffer> q_proj;
    std::shared_ptr<CudaBuffer> queries;
    std::shared_ptr<CudaBuffer> gate;
    std::shared_ptr<CudaBuffer> attention_output;
    std::shared_ptr<CudaBuffer> attention_proj;
};

template<Qwen35Size QWEN35_SIZE>
class Qwen35LinearAttentionLayer final : public Qwen35Layer<QWEN35_SIZE> {
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
    std::shared_ptr<CudaBuffer> norm_weight;
    std::shared_ptr<CudaBuffer> out_proj_weight;
    std::shared_ptr<CudaBuffer> out_proj_bias;

    void forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state) override;

private:
    using Qwen35Layer<QWEN35_SIZE>::layer_num;
    using Qwen35Layer<QWEN35_SIZE>::norm_hidden_state;
    using Qwen35Layer<QWEN35_SIZE>::input_layernorm;
    using Qwen35Layer<QWEN35_SIZE>::apply_mlp;

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
