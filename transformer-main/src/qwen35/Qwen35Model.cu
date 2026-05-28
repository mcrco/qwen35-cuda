#include "Qwen35Model.cuh"

#include "../cpu_ops/BufferOps.cuh"
#include "../cpu_ops/Qwen35LayerNorm.cuh"
#include "../cpu_ops/Qwen35MatrixVectorMultiply.cuh"
#include "../cpu_ops/Qwen35Sampling.cuh"

#include <stdexcept>

template<Qwen35Size QWEN35_SIZE>
Qwen35Model<QWEN35_SIZE>::Qwen35Model() {
    hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
    norm_hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(input_float_t));
    output_scores = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::vocab_size() * sizeof(input_float_t));
    layers.reserve(Qwen35Config<QWEN35_SIZE>::num_layers());
}

template<Qwen35Size QWEN35_SIZE>
Qwen35Cache Qwen35Model<QWEN35_SIZE>::allocate_cache(size_t max_seq_len) const {
    Qwen35Cache cache;
    cache.keys = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::keys_size() * sizeof(input_float_t));
    cache.values = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::values_size() * sizeof(input_float_t));
    cache.conv_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim() * Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
    cache.recurrent_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * Qwen35Config<QWEN35_SIZE>::linear_key_head_dim() * Qwen35Config<QWEN35_SIZE>::linear_value_head_dim() * sizeof(input_float_t));
    BufferOps::zero(static_cast<input_float_t *>(cache.keys->data), cache.keys->size / sizeof(input_float_t));
    BufferOps::zero(static_cast<input_float_t *>(cache.values->data), cache.values->size / sizeof(input_float_t));
    BufferOps::zero(static_cast<input_float_t *>(cache.conv_states->data), cache.conv_states->size / sizeof(input_float_t));
    BufferOps::zero(static_cast<input_float_t *>(cache.recurrent_states->data), cache.recurrent_states->size / sizeof(input_float_t));
    return cache;
}

template<Qwen35Size QWEN35_SIZE>
int32_t Qwen35Model<QWEN35_SIZE>::forward(Qwen35Cache &cache, int32_t input_tok_id, float temperature) {
    if (input_tok_id < 0 || static_cast<size_t>(input_tok_id) >= Qwen35Config<QWEN35_SIZE>::vocab_size()) {
        throw std::runtime_error("invalid input token id");
    }

    auto embedding = static_cast<input_float_t *>(embedding_weight->data);
    auto hidden = static_cast<input_float_t *>(hidden_state->data);
    BufferOps::copy(embedding + static_cast<size_t>(input_tok_id) * Qwen35Config<QWEN35_SIZE>::hidden_size(), hidden, Qwen35Config<QWEN35_SIZE>::hidden_size());

    for (auto &layer : layers) {
        layer->forward(cache, hidden_state);
    }

    auto norm_hidden = static_cast<input_float_t *>(norm_hidden_state->data);
    Qwen35LayerNorm::zero_centered_rms_norm(static_cast<input_float_t *>(final_layernorm->data), hidden, norm_hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::rms_norm_eps());

    auto lm_head = static_cast<input_float_t *>(lm_head_weight->data);
    auto output_scores_ptr = static_cast<input_float_t *>(output_scores->data);
    Qwen35MatrixVectorMultiply::matmul(Qwen35Config<QWEN35_SIZE>::vocab_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), lm_head, nullptr, norm_hidden, output_scores_ptr);

    cache.seq_len++;
    return Qwen35Sampling::sample(output_scores_ptr, Qwen35Config<QWEN35_SIZE>::vocab_size(), temperature, rng);
}

template class Qwen35Model<QWEN35_0_8B>;
template class Qwen35Model<QWEN35_4B>;
template class Qwen35Model<QWEN35_9B>;
