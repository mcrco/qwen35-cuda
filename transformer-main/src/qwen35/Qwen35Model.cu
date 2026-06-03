#include "Qwen35Model.cuh"

#include "../cpu_ops/Qwen35LayerNorm.cuh"
#include "../cpu_ops/Qwen35Sampling.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"

#include <stdexcept>

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t, typename logits_t>
Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t, logits_t>::Qwen35Model() {
    hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
    norm_hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
    output_scores = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::vocab_size() * sizeof(logits_t));
    layers.reserve(Qwen35Config<QWEN35_SIZE>::num_layers());
    checkCuda(cudaStreamCreate(&stream));
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t, typename logits_t>
Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t, logits_t>::~Qwen35Model() {
    checkCuda(cudaStreamDestroy(stream));
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t, typename logits_t>
Qwen35Cache Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t, logits_t>::allocate_cache(size_t max_seq_len) const {
    Qwen35Cache cache;
    cache.keys = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::keys_size() * sizeof(cache_t));
    cache.values = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::values_size() * sizeof(cache_t));
    cache.conv_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim() * Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(input_float_t));
    cache.recurrent_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * Qwen35Config<QWEN35_SIZE>::linear_key_head_dim() * Qwen35Config<QWEN35_SIZE>::linear_value_head_dim() * sizeof(input_float_t));
    checkCuda(cudaMemsetAsync(cache.keys->data, 0, cache.keys->size, stream));
    checkCuda(cudaMemsetAsync(cache.values->data, 0, cache.values->size, stream));
    checkCuda(cudaMemsetAsync(cache.conv_states->data, 0, cache.conv_states->size, stream));
    checkCuda(cudaMemsetAsync(cache.recurrent_states->data, 0, cache.recurrent_states->size, stream));
    checkCuda(cudaStreamSynchronize(stream));
    return cache;
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t, typename cache_t, typename logits_t>
int32_t Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t, cache_t, logits_t>::forward(Qwen35Cache &cache, int32_t input_tok_id, float temperature) {
    if (input_tok_id < 0 || static_cast<size_t>(input_tok_id) >= Qwen35Config<QWEN35_SIZE>::vocab_size()) {
        throw std::runtime_error("invalid input token id");
    }

    auto embedding = static_cast<weight_t *>(embedding_weight->data);
    auto hidden = static_cast<hidden_t *>(hidden_state->data);
    qwen35_layer_detail::convert_copy<weight_t, hidden_t, compute_t>(embedding + static_cast<size_t>(input_tok_id) * Qwen35Config<QWEN35_SIZE>::hidden_size(), hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), stream);
    checkCuda(cudaStreamSynchronize(stream));

    for (auto &layer : layers) {
        layer->forward(cache, hidden_state, stream);
    }

    checkCuda(cudaStreamSynchronize(stream));
    auto norm_hidden = static_cast<hidden_t *>(norm_hidden_state->data);
    Qwen35LayerNorm::zero_centered_rms_norm(
        static_cast<weight_t *>(final_layernorm->data),
        hidden,
        norm_hidden,
        Qwen35Config<QWEN35_SIZE>::hidden_size(),
        Qwen35Config<QWEN35_SIZE>::rms_norm_eps());

    auto lm_head = static_cast<weight_t *>(lm_head_weight->data);
    auto output_scores_ptr = static_cast<logits_t *>(output_scores->data);
    MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, logits_t, compute_t>(Qwen35Config<QWEN35_SIZE>::vocab_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), lm_head, nullptr, norm_hidden, output_scores_ptr, stream);

    cache.seq_len++;
    if (temperature == 0.0f) {
        int32_t *next_token_idx_gpu = argmax.template argmax_as_float<logits_t>(output_scores, Qwen35Config<QWEN35_SIZE>::vocab_size(), stream);
        int32_t next_token_idx_cpu;
        checkCuda(cudaMemcpyAsync(&next_token_idx_cpu, next_token_idx_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        return next_token_idx_cpu;
    }

    checkCuda(cudaStreamSynchronize(stream));
    return Qwen35Sampling::sample(static_cast<input_float_t *>(output_scores->data), Qwen35Config<QWEN35_SIZE>::vocab_size(), temperature, rng);
}

template class Qwen35Model<QWEN35_0_8B, input_float_t, input_float_t, float, input_float_t, input_float_t>;
template class Qwen35Model<QWEN35_4B, input_float_t, input_float_t, float, input_float_t, input_float_t>;
template class Qwen35Model<QWEN35_9B, input_float_t, input_float_t, float, input_float_t, input_float_t>;
