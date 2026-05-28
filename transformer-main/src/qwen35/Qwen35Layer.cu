#include "Qwen35Layer.cuh"

#include "../cpu_ops/BufferOps.cuh"
#include "../cpu_ops/Qwen35GroupQueryAttention.cuh"
#include "../cpu_ops/Qwen35LayerNorm.cuh"
#include "../cpu_ops/Qwen35LinearAttention.cuh"
#include "../cpu_ops/Qwen35MatrixVectorMultiply.cuh"
#include "../cpu_ops/Qwen35RoPE.cuh"
#include "../cpu_ops/Qwen35SiLUMult.cuh"

#include <cmath>

template<Qwen35Size QWEN35_SIZE>
Qwen35Layer<QWEN35_SIZE>::Qwen35Layer(size_t layer_num) : layer_num(layer_num) {
    norm_hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
    ffn_gate = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::intermediate_size() * sizeof(input_float_t));
    ffn_up = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::intermediate_size() * sizeof(input_float_t));
    ffn_down = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
}

template<Qwen35Size QWEN35_SIZE>
void Qwen35Layer<QWEN35_SIZE>::apply_mlp(const std::shared_ptr<CudaBuffer> &hidden_state) {
    auto hidden = static_cast<input_float_t *>(hidden_state->data);
    auto norm_hidden = static_cast<input_float_t *>(norm_hidden_state->data);
    auto post_norm = static_cast<input_float_t *>(post_attention_layernorm->data);
    Qwen35LayerNorm::zero_centered_rms_norm(post_norm, hidden, norm_hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());

    auto gate_weight = static_cast<input_float_t *>(gate_proj_weight->data);
    auto up_weight = static_cast<input_float_t *>(up_proj_weight->data);
    auto down_weight = static_cast<input_float_t *>(down_proj_weight->data);
    auto gate = static_cast<input_float_t *>(ffn_gate->data);
    auto up = static_cast<input_float_t *>(ffn_up->data);
    auto down = static_cast<input_float_t *>(ffn_down->data);

    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::intermediate_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), gate_weight, nullptr, norm_hidden, gate);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::intermediate_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), up_weight, nullptr, norm_hidden, up);
    Qwen35SiLUMult::silu_mult_in_place(gate, up, Qwen35Config<QWEN35_SIZE>::intermediate_size());
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::intermediate_size(), down_weight, nullptr, gate, down);
    BufferOps::add_in_place(hidden, down, Qwen35Config<QWEN35_SIZE>::hidden_size());
}

template<Qwen35Size QWEN35_SIZE>
Qwen35FullAttnLayer<QWEN35_SIZE>::Qwen35FullAttnLayer(size_t layer_num) : Qwen35Layer<QWEN35_SIZE>(layer_num) {
    q_proj = std::make_shared<CudaBuffer>(2 * Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
    queries = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
    gate = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
    attention_output = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
    attention_proj = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
}

template<Qwen35Size QWEN35_SIZE>
void Qwen35FullAttnLayer<QWEN35_SIZE>::forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state) {
    size_t seq_len = cache.seq_len;
    auto hidden = static_cast<input_float_t *>(hidden_state->data);
    auto norm_hidden = static_cast<input_float_t *>(norm_hidden_state->data);
    Qwen35LayerNorm::zero_centered_rms_norm(static_cast<input_float_t *>(input_layernorm->data), hidden, norm_hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());

    auto q_proj_out = static_cast<input_float_t *>(q_proj->data);
    Qwen35MatrixVectorMultiply::matmul(2 * Qwen35Config<QWEN35_SIZE>::queries_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(q_proj_weight->data), q_proj_bias ? static_cast<input_float_t *>(q_proj_bias->data) : nullptr, norm_hidden, q_proj_out);

    auto query_ptr = static_cast<input_float_t *>(queries->data);
    auto gate_ptr = static_cast<input_float_t *>(gate->data);
    for (size_t h = 0; h < Qwen35Config<QWEN35_SIZE>::num_query_heads(); h++) {
        Qwen35LayerNorm::zero_centered_rms_norm(
            static_cast<input_float_t *>(q_norm_weight->data),
            q_proj_out + h * 2 * Qwen35Config<QWEN35_SIZE>::head_size(),
            query_ptr + h * Qwen35Config<QWEN35_SIZE>::head_size(),
            Qwen35Config<QWEN35_SIZE>::head_size(),
            Qwen35Config<QWEN35_SIZE>::rms_norm_eps());
        BufferOps::copy(q_proj_out + h * 2 * Qwen35Config<QWEN35_SIZE>::head_size() + Qwen35Config<QWEN35_SIZE>::head_size(), gate_ptr + h * Qwen35Config<QWEN35_SIZE>::head_size(), Qwen35Config<QWEN35_SIZE>::head_size());
    }

    auto keys_cache = static_cast<input_float_t *>(cache.keys->data);
    auto values_cache = static_cast<input_float_t *>(cache.values->data);
    auto new_keys = keys_cache + seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::keys_size() + layer_num * Qwen35Config<QWEN35_SIZE>::keys_size();
    auto new_values = values_cache + seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::values_size() + layer_num * Qwen35Config<QWEN35_SIZE>::values_size();
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::keys_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(k_proj_weight->data), k_proj_bias ? static_cast<input_float_t *>(k_proj_bias->data) : nullptr, norm_hidden, new_keys);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::values_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(v_proj_weight->data), v_proj_bias ? static_cast<input_float_t *>(v_proj_bias->data) : nullptr, norm_hidden, new_values);

    for (size_t h = 0; h < Qwen35Config<QWEN35_SIZE>::num_kv_heads(); h++) {
        Qwen35LayerNorm::zero_centered_rms_norm(static_cast<input_float_t *>(k_norm_weight->data), new_keys + h * Qwen35Config<QWEN35_SIZE>::head_size(), new_keys + h * Qwen35Config<QWEN35_SIZE>::head_size(), Qwen35Config<QWEN35_SIZE>::head_size(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());
    }

    Qwen35RoPE::apply_partial_rope_to_qk(
        query_ptr,
        Qwen35Config<QWEN35_SIZE>::num_query_heads(),
        new_keys,
        Qwen35Config<QWEN35_SIZE>::num_kv_heads(),
        Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::rotary_dim(),
        seq_len,
        Qwen35Config<QWEN35_SIZE>::rope_theta_base());

    auto attn_out = static_cast<input_float_t *>(attention_output->data);
    Qwen35GroupQueryAttention::sdpa(
        query_ptr,
        keys_cache,
        values_cache,
        attn_out,
        gate_ptr,
        layer_num,
        seq_len,
        Qwen35Config<QWEN35_SIZE>::num_layers(),
        Qwen35Config<QWEN35_SIZE>::num_query_heads(),
        Qwen35Config<QWEN35_SIZE>::num_kv_heads(),
        Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::keys_size(),
        Qwen35Config<QWEN35_SIZE>::values_size());

    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::queries_size(), static_cast<input_float_t *>(o_proj_weight->data), o_proj_bias ? static_cast<input_float_t *>(o_proj_bias->data) : nullptr, attn_out, static_cast<input_float_t *>(attention_proj->data));
    BufferOps::add_in_place(hidden, static_cast<input_float_t *>(attention_proj->data), Qwen35Config<QWEN35_SIZE>::hidden_size());
    apply_mlp(hidden_state);
}

template<Qwen35Size QWEN35_SIZE>
Qwen35LinearAttentionLayer<QWEN35_SIZE>::Qwen35LinearAttentionLayer(size_t layer_num) : Qwen35Layer<QWEN35_SIZE>(layer_num) {
    qkv = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
    gates = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(input_float_t));
    beta_raw = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * sizeof(input_float_t));
    decay_raw = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * sizeof(input_float_t));
    mixed_qkv = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
    queries_float = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
    keys_float = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
    values_float = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
    weighted_values_float = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
    gated_weighted_values = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(input_float_t));
    attention_proj = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
}

template<Qwen35Size QWEN35_SIZE>
void Qwen35LinearAttentionLayer<QWEN35_SIZE>::forward(Qwen35Cache &cache, const std::shared_ptr<CudaBuffer> &hidden_state) {
    auto hidden = static_cast<input_float_t *>(hidden_state->data);
    auto norm_hidden = static_cast<input_float_t *>(norm_hidden_state->data);
    Qwen35LayerNorm::zero_centered_rms_norm(static_cast<input_float_t *>(input_layernorm->data), hidden, norm_hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());

    auto qkv_ptr = static_cast<input_float_t *>(qkv->data);
    auto gates_ptr = static_cast<input_float_t *>(gates->data);
    auto beta_raw_ptr = static_cast<input_float_t *>(beta_raw->data);
    auto decay_raw_ptr = static_cast<input_float_t *>(decay_raw->data);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::linear_conv_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(in_proj_qkv_weight->data), nullptr, norm_hidden, qkv_ptr);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::linear_values_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(in_proj_z_weight->data), nullptr, norm_hidden, gates_ptr);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(in_proj_b_weight->data), nullptr, norm_hidden, beta_raw_ptr);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size(), static_cast<input_float_t *>(in_proj_a_weight->data), nullptr, norm_hidden, decay_raw_ptr);

    auto conv_state = static_cast<input_float_t *>(cache.conv_states->data) + layer_num * Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim() * Qwen35Config<QWEN35_SIZE>::linear_conv_size();
    auto mixed = static_cast<input_float_t *>(mixed_qkv->data);
    auto conv_weight = static_cast<input_float_t *>(conv1d_weight->data);
    auto conv_bias = conv1d_bias ? static_cast<input_float_t *>(conv1d_bias->data) : nullptr;
    Qwen35LinearAttention::conv1d_silu(
        qkv_ptr,
        conv_state,
        conv_weight,
        conv_bias,
        mixed,
        Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim(),
        Qwen35Config<QWEN35_SIZE>::linear_conv_size());

    auto qf = static_cast<float *>(queries_float->data);
    auto kf = static_cast<float *>(keys_float->data);
    auto vf = static_cast<float *>(values_float->data);
    Qwen35LinearAttention::split_qkv(
        mixed,
        qf,
        kf,
        vf,
        Qwen35Config<QWEN35_SIZE>::linear_num_key_heads(),
        Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
        Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(),
        Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());
    Qwen35LayerNorm::l2norm_rows(qf, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(), 1.0f / std::sqrt(static_cast<float>(Qwen35Config<QWEN35_SIZE>::linear_key_head_dim())), 1e-6f);
    Qwen35LayerNorm::l2norm_rows(kf, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(), 1.0f, 1e-6f);

    auto state = static_cast<input_float_t *>(cache.recurrent_states->data) + layer_num * Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * Qwen35Config<QWEN35_SIZE>::linear_key_head_dim() * Qwen35Config<QWEN35_SIZE>::linear_value_head_dim();
    auto weighted = static_cast<float *>(weighted_values_float->data);
    auto dt_bias_ptr = static_cast<input_float_t *>(dt_bias->data);
    auto a_log_ptr = static_cast<input_float_t *>(A_log->data);
    Qwen35LinearAttention::gated_delta_update(
        state,
        qf,
        kf,
        vf,
        beta_raw_ptr,
        decay_raw_ptr,
        dt_bias_ptr,
        a_log_ptr,
        weighted,
        Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
        Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(),
        Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());

    Qwen35LayerNorm::gated_rms_norm(static_cast<input_float_t *>(norm_weight->data), weighted, gates_ptr, static_cast<input_float_t *>(gated_weighted_values->data), Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::linear_value_head_dim(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::linear_values_size(), static_cast<input_float_t *>(out_proj_weight->data), out_proj_bias ? static_cast<input_float_t *>(out_proj_bias->data) : nullptr, static_cast<input_float_t *>(gated_weighted_values->data), static_cast<input_float_t *>(attention_proj->data));
    BufferOps::add_in_place(hidden, static_cast<input_float_t *>(attention_proj->data), Qwen35Config<QWEN35_SIZE>::hidden_size());
    apply_mlp(hidden_state);
}

template class Qwen35Layer<QWEN35_0_8B>;
template class Qwen35Layer<QWEN35_4B>;
template class Qwen35Layer<QWEN35_9B>;
template class Qwen35FullAttnLayer<QWEN35_0_8B>;
template class Qwen35FullAttnLayer<QWEN35_4B>;
template class Qwen35FullAttnLayer<QWEN35_9B>;
template class Qwen35LinearAttentionLayer<QWEN35_0_8B>;
template class Qwen35LinearAttentionLayer<QWEN35_4B>;
template class Qwen35LinearAttentionLayer<QWEN35_9B>;
