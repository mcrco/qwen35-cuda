#include "Qwen35Layer.cuh"

#include "../cpu_ops/Qwen35LinearAttention.cuh"
#include "../cpu_ops/Qwen35RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/SiLUMult.cuh"

#include <cmath>

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t>
Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t>::Qwen35Layer(size_t layer_num)
    : layer_num(layer_num) {
  norm_hidden_state = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
  ffn_gate = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::intermediate_size() * sizeof(hidden_t));
  ffn_up = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::intermediate_size() * sizeof(hidden_t));
  ffn_down = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
}

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t>
void Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t>::
    apply_mlp(const std::shared_ptr<CudaBuffer> &hidden_state, cudaStream_t stream) {
  auto hidden = static_cast<hidden_t *>(hidden_state->data);
  auto norm_hidden = static_cast<hidden_t *>(norm_hidden_state->data);
  post_attention_layernorm
      .template zero_centered_rms_norm<hidden_t, weight_t, hidden_t,
                                       compute_t>(
          hidden_state, norm_hidden_state,
          Qwen35Config<QWEN35_SIZE>::hidden_size(),
          Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);

  auto gate_weight = static_cast<weight_t *>(gate_proj_weight->data);
  auto up_weight = static_cast<weight_t *>(up_proj_weight->data);
  auto down_weight = static_cast<weight_t *>(down_proj_weight->data);
  auto gate = static_cast<hidden_t *>(ffn_gate->data);
  auto up = static_cast<hidden_t *>(ffn_up->data);
  auto down = static_cast<hidden_t *>(ffn_down->data);

  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::intermediate_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(), gate_weight, nullptr,
      norm_hidden, gate, stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::intermediate_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(), up_weight, nullptr, norm_hidden,
      up, stream);
  SiLUMult::silu_mult_in_place<hidden_t, hidden_t, compute_t>(
      ffn_gate, ffn_up, Qwen35Config<QWEN35_SIZE>::intermediate_size(), stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, hidden_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      Qwen35Config<QWEN35_SIZE>::intermediate_size(), down_weight, nullptr,
      gate, down, stream);
  qwen35_layer_detail::residual_add<hidden_t, hidden_t, compute_t>(
      hidden, down, Qwen35Config<QWEN35_SIZE>::hidden_size(), stream);
}

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t,
          typename compute_t, typename cache_t>
Qwen35FullAttnLayer<QWEN35_SIZE, weight_t, hidden_t, compute_t,
                    cache_t>::Qwen35FullAttnLayer(size_t layer_num)
    : Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t>(
          layer_num) {
  q_proj = std::make_shared<CudaBuffer>(
      2 * Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
  queries = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
  gate = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
  attention_output = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::queries_size() * sizeof(input_float_t));
  attention_proj = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
}

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t,
          typename compute_t, typename cache_t>
void Qwen35FullAttnLayer<QWEN35_SIZE, weight_t, hidden_t, compute_t,
                         cache_t>::forward(Qwen35Cache &cache,
                                           const std::shared_ptr<CudaBuffer>
                                               &hidden_state,
                                           cudaStream_t stream) {
  checkCuda(cudaStreamSynchronize(stream));

  size_t seq_len = cache.seq_len;

  //
  auto hidden = static_cast<hidden_t *>(hidden_state->data);
  auto norm_hidden = static_cast<hidden_t *>(norm_hidden_state->data);
  input_layernorm
      .template zero_centered_rms_norm<hidden_t, weight_t, hidden_t,
                                       compute_t>(
          hidden_state, norm_hidden_state,
          Qwen35Config<QWEN35_SIZE>::hidden_size(),
          Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);

  auto q_proj_out = static_cast<input_float_t *>(q_proj->data);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      2 * Qwen35Config<QWEN35_SIZE>::queries_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(q_proj_weight->data),
      q_proj_bias ? static_cast<weight_t *>(q_proj_bias->data) : nullptr,
      norm_hidden, q_proj_out, stream);
  checkCuda(cudaStreamSynchronize(stream));

  auto query_ptr = static_cast<input_float_t *>(queries->data);
  auto gate_ptr = static_cast<input_float_t *>(gate->data);
  for (size_t h = 0; h < Qwen35Config<QWEN35_SIZE>::num_query_heads(); h++) {
    q_norm.template zero_centered_rms_norm<input_float_t, input_float_t,
                                           input_float_t, compute_t>(
        q_proj_out + h * 2 * Qwen35Config<QWEN35_SIZE>::head_size(),
        query_ptr + h * Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);
    qwen35_layer_detail::convert_copy<input_float_t, input_float_t, compute_t>(
        q_proj_out + h * 2 * Qwen35Config<QWEN35_SIZE>::head_size() +
            Qwen35Config<QWEN35_SIZE>::head_size(),
        gate_ptr + h * Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::head_size(), stream);
  }
  checkCuda(cudaStreamSynchronize(stream));

  auto keys_cache = static_cast<input_float_t *>(cache.keys->data);
  auto values_cache = static_cast<input_float_t *>(cache.values->data);
  auto new_keys = keys_cache +
                  seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() *
                      Qwen35Config<QWEN35_SIZE>::keys_size() +
                  layer_num * Qwen35Config<QWEN35_SIZE>::keys_size();
  auto new_values = values_cache +
                    seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() *
                        Qwen35Config<QWEN35_SIZE>::values_size() +
                    layer_num * Qwen35Config<QWEN35_SIZE>::values_size();
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::keys_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(k_proj_weight->data),
      k_proj_bias ? static_cast<weight_t *>(k_proj_bias->data) : nullptr,
      norm_hidden, new_keys, stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::values_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(v_proj_weight->data),
      v_proj_bias ? static_cast<weight_t *>(v_proj_bias->data) : nullptr,
      norm_hidden, new_values, stream);
  checkCuda(cudaStreamSynchronize(stream));

  for (size_t h = 0; h < Qwen35Config<QWEN35_SIZE>::num_kv_heads(); h++) {
    k_norm.template zero_centered_rms_norm<input_float_t, input_float_t,
                                           input_float_t, compute_t>(
        new_keys + h * Qwen35Config<QWEN35_SIZE>::head_size(),
        new_keys + h * Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::head_size(),
        Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);
  }
  checkCuda(cudaStreamSynchronize(stream));

  Qwen35RoPE::apply_partial_rope_to_qk(
      query_ptr, Qwen35Config<QWEN35_SIZE>::num_query_heads(), new_keys,
      Qwen35Config<QWEN35_SIZE>::num_kv_heads(),
      Qwen35Config<QWEN35_SIZE>::head_size(),
      Qwen35Config<QWEN35_SIZE>::rotary_dim(), seq_len,
      Qwen35Config<QWEN35_SIZE>::rope_theta_base());

  auto attn_out = static_cast<input_float_t *>(attention_output->data);
  GroupQueryAttention<QWEN35_SIZE>::
      template sdpa<input_float_t, input_float_t, input_float_t, input_float_t,
                    input_float_t, compute_t>(
          query_ptr, keys_cache, values_cache, attn_out, gate_ptr, layer_num,
          seq_len, stream);

  MatrixVectorMultiply::matmul<weight_t, weight_t, input_float_t, hidden_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      Qwen35Config<QWEN35_SIZE>::queries_size(),
      static_cast<weight_t *>(o_proj_weight->data),
      o_proj_bias ? static_cast<weight_t *>(o_proj_bias->data) : nullptr,
      attn_out, static_cast<hidden_t *>(attention_proj->data), stream);
  qwen35_layer_detail::residual_add<hidden_t, hidden_t, compute_t>(
      hidden, static_cast<hidden_t *>(attention_proj->data),
      Qwen35Config<QWEN35_SIZE>::hidden_size(), stream);
  checkCuda(cudaStreamSynchronize(stream));
  apply_mlp(hidden_state, stream);
}

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t,
          typename compute_t, typename cache_t>
Qwen35LinearAttentionLayer<QWEN35_SIZE, weight_t, hidden_t, compute_t,
                           cache_t>::Qwen35LinearAttentionLayer(size_t
                                                                    layer_num)
    : Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t>(
          layer_num) {
  qkv = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
  gates = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(input_float_t));
  beta_raw = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() *
      sizeof(input_float_t));
  decay_raw = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() *
      sizeof(input_float_t));
  mixed_qkv = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
  queries_float = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
  keys_float = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
  values_float = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
  weighted_values_float = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(float));
  gated_weighted_values = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size() * sizeof(input_float_t));
  attention_proj = std::make_shared<CudaBuffer>(
      Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
}

template <Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t,
          typename compute_t, typename cache_t>
void Qwen35LinearAttentionLayer<
    QWEN35_SIZE, weight_t, hidden_t, compute_t,
    cache_t>::forward(Qwen35Cache &cache,
                      const std::shared_ptr<CudaBuffer> &hidden_state,
                      cudaStream_t stream) {
  checkCuda(cudaStreamSynchronize(stream));
  auto hidden = static_cast<hidden_t *>(hidden_state->data);
  auto norm_hidden = static_cast<hidden_t *>(norm_hidden_state->data);
  input_layernorm
      .template zero_centered_rms_norm<hidden_t, weight_t, hidden_t,
                                       compute_t>(
          hidden_state, norm_hidden_state,
          Qwen35Config<QWEN35_SIZE>::hidden_size(),
          Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);

  auto qkv_ptr = static_cast<input_float_t *>(qkv->data);
  auto gates_ptr = static_cast<input_float_t *>(gates->data);
  auto beta_raw_ptr = static_cast<input_float_t *>(beta_raw->data);
  auto decay_raw_ptr = static_cast<input_float_t *>(decay_raw->data);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::linear_conv_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(in_proj_qkv_weight->data), nullptr, norm_hidden,
      qkv_ptr, stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::linear_values_size(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(in_proj_z_weight->data), nullptr, norm_hidden,
      gates_ptr, stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(in_proj_b_weight->data), nullptr, norm_hidden,
      beta_raw_ptr, stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, input_float_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      static_cast<weight_t *>(in_proj_a_weight->data), nullptr, norm_hidden,
      decay_raw_ptr, stream);
  checkCuda(cudaStreamSynchronize(stream));

  auto conv_state = static_cast<input_float_t *>(cache.conv_states->data) +
                    layer_num *
                        Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim() *
                        Qwen35Config<QWEN35_SIZE>::linear_conv_size();
  auto mixed = static_cast<input_float_t *>(mixed_qkv->data);
  auto conv_weight = static_cast<input_float_t *>(conv1d_weight->data);
  auto conv_bias =
      conv1d_bias ? static_cast<input_float_t *>(conv1d_bias->data) : nullptr;
  Qwen35LinearAttention::conv1d_silu(
      qkv_ptr, conv_state, conv_weight, conv_bias, mixed,
      Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim(),
      Qwen35Config<QWEN35_SIZE>::linear_conv_size());

  auto qf = static_cast<float *>(queries_float->data);
  auto kf = static_cast<float *>(keys_float->data);
  auto vf = static_cast<float *>(values_float->data);
  Qwen35LinearAttention::split_qkv(
      mixed, qf, kf, vf, Qwen35Config<QWEN35_SIZE>::linear_num_key_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(),
      Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());
  LayerNorm::l2_norm_rows<float, compute_t>(
      queries_float, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(),
      1.0f / std::sqrt(static_cast<float>(
                 Qwen35Config<QWEN35_SIZE>::linear_key_head_dim())),
      1e-6f, stream);
  LayerNorm::l2_norm_rows<float, compute_t>(
      keys_float, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(), 1.0f, 1e-6f, stream);
  checkCuda(cudaStreamSynchronize(stream));

  auto state = static_cast<input_float_t *>(cache.recurrent_states->data) +
               layer_num * Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() *
                   Qwen35Config<QWEN35_SIZE>::linear_key_head_dim() *
                   Qwen35Config<QWEN35_SIZE>::linear_value_head_dim();
  auto weighted = static_cast<float *>(weighted_values_float->data);
  auto dt_bias_ptr = static_cast<input_float_t *>(dt_bias->data);
  auto a_log_ptr = static_cast<input_float_t *>(A_log->data);
  Qwen35LinearAttention::gated_delta_update(
      state, qf, kf, vf, beta_raw_ptr, decay_raw_ptr, dt_bias_ptr, a_log_ptr,
      weighted, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_key_head_dim(),
      Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());

  norm.template normalize_gated_hidden_state<float, input_float_t,
                                             input_float_t, input_float_t,
                                             compute_t>(
      weighted_values_float, gates, gated_weighted_values,
      Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(),
      Qwen35Config<QWEN35_SIZE>::linear_value_head_dim(),
      Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);
  MatrixVectorMultiply::matmul<weight_t, weight_t, input_float_t, hidden_t,
                               compute_t>(
      Qwen35Config<QWEN35_SIZE>::hidden_size(),
      Qwen35Config<QWEN35_SIZE>::linear_values_size(),
      static_cast<weight_t *>(out_proj_weight->data),
      out_proj_bias ? static_cast<weight_t *>(out_proj_bias->data) : nullptr,
      static_cast<input_float_t *>(gated_weighted_values->data),
      static_cast<hidden_t *>(attention_proj->data), stream);
  qwen35_layer_detail::residual_add<hidden_t, hidden_t, compute_t>(
      hidden, static_cast<hidden_t *>(attention_proj->data),
      Qwen35Config<QWEN35_SIZE>::hidden_size(), stream);
  checkCuda(cudaStreamSynchronize(stream));
  apply_mlp(hidden_state, stream);
}

template class Qwen35Layer<QWEN35_0_8B, input_float_t, input_float_t, float,
                           input_float_t>;
template class Qwen35Layer<QWEN35_4B, input_float_t, input_float_t, float,
                           input_float_t>;
template class Qwen35Layer<QWEN35_9B, input_float_t, input_float_t, float,
                           input_float_t>;
template class Qwen35FullAttnLayer<QWEN35_0_8B, input_float_t, input_float_t,
                                   float, input_float_t>;
template class Qwen35FullAttnLayer<QWEN35_4B, input_float_t, input_float_t,
                                   float, input_float_t>;
template class Qwen35FullAttnLayer<QWEN35_9B, input_float_t, input_float_t,
                                   float, input_float_t>;
template class Qwen35LinearAttentionLayer<QWEN35_0_8B, input_float_t,
                                          input_float_t, float, input_float_t>;
template class Qwen35LinearAttentionLayer<QWEN35_4B, input_float_t,
                                          input_float_t, float, input_float_t>;
template class Qwen35LinearAttentionLayer<QWEN35_9B, input_float_t,
                                          input_float_t, float, input_float_t>;
