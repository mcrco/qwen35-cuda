#include "Qwen35Model.cuh"

#include "../gpu_ops/MatrixVectorMultiply.cuh"

#include <stdexcept>
#include <random>

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::Qwen35Model() {
    hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
    norm_hidden_state = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::hidden_size() * sizeof(hidden_t));
    output_scores = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::vocab_size() * sizeof(float));
    layers.reserve(Qwen35Config<QWEN35_SIZE>::num_layers());
    checkCuda(cudaStreamCreate(&stream));
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::~Qwen35Model() {
    checkCuda(cudaStreamDestroy(stream));
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
Qwen35Cache Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::allocate_cache(size_t max_seq_len) const {
    Qwen35Cache cache;
    cache.keys = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::keys_size() * sizeof(hidden_t));
    cache.values = std::make_shared<CudaBuffer>(max_seq_len * Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::values_size() * sizeof(hidden_t));
    cache.conv_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim() * Qwen35Config<QWEN35_SIZE>::linear_conv_size() * sizeof(hidden_t));
    cache.recurrent_states = std::make_shared<CudaBuffer>(Qwen35Config<QWEN35_SIZE>::num_layers() * Qwen35Config<QWEN35_SIZE>::linear_num_value_heads() * Qwen35Config<QWEN35_SIZE>::linear_key_head_dim() * Qwen35Config<QWEN35_SIZE>::linear_value_head_dim() * sizeof(hidden_t));
    checkCuda(cudaMemsetAsync(cache.keys->data, 0, cache.keys->size, stream));
    checkCuda(cudaMemsetAsync(cache.values->data, 0, cache.values->size, stream));
    checkCuda(cudaMemsetAsync(cache.conv_states->data, 0, cache.conv_states->size, stream));
    checkCuda(cudaMemsetAsync(cache.recurrent_states->data, 0, cache.recurrent_states->size, stream));
    checkCuda(cudaStreamSynchronize(stream));
    return cache;
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
void Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::set_seed(uint32_t seed) {
    rng.seed(seed);
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
cudaStream_t Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::cuda_stream() const {
    return stream;
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
int32_t Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>::forward(Qwen35Cache &cache, int32_t input_tok_id, float temperature) {
    if (input_tok_id < 0 || static_cast<size_t>(input_tok_id) >= Qwen35Config<QWEN35_SIZE>::vocab_size()) {
        throw std::runtime_error("invalid input token id");
    }

    auto embedding = static_cast<weight_t *>(embedding_weight->data);
    auto hidden = static_cast<hidden_t *>(hidden_state->data);
    qwen35_layer_detail::convert_copy<weight_t, hidden_t, compute_t>(embedding + static_cast<size_t>(input_tok_id) * Qwen35Config<QWEN35_SIZE>::hidden_size(), hidden, Qwen35Config<QWEN35_SIZE>::hidden_size(), stream);

    for (auto &layer : layers) {
        layer->forward(cache, hidden_state, stream);
    }

    auto norm_hidden = static_cast<hidden_t *>(norm_hidden_state->data);
    final_layernorm
        .template zero_centered_rms_norm<hidden_t, weight_t, hidden_t,
                                         compute_t>(
            hidden_state, norm_hidden_state,
            Qwen35Config<QWEN35_SIZE>::hidden_size(),
            Qwen35Config<QWEN35_SIZE>::rms_norm_eps(), stream);

    auto lm_head = static_cast<weight_t *>(lm_head_weight->data);
    auto output_scores_ptr = static_cast<float *>(output_scores->data);
    MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, float, compute_t>(Qwen35Config<QWEN35_SIZE>::vocab_size(), Qwen35Config<QWEN35_SIZE>::hidden_size(), lm_head, nullptr, norm_hidden, output_scores_ptr, stream);

    cache.seq_len++;
    if (temperature == 0.0f) {
        int32_t *next_token_idx_gpu = argmax.template argmax_as_float<float>(output_scores, Qwen35Config<QWEN35_SIZE>::vocab_size(), stream);
        int32_t next_token_idx_cpu;
        checkCuda(cudaMemcpyAsync(&next_token_idx_cpu, next_token_idx_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        return next_token_idx_cpu;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int32_t *next_token_idx_gpu = sampling.sample(output_scores, Qwen35Config<QWEN35_SIZE>::vocab_size(), temperature, dist(rng), stream);
    int32_t next_token_idx_cpu;
    checkCuda(cudaMemcpyAsync(&next_token_idx_cpu, next_token_idx_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
    return next_token_idx_cpu;
}

template class Qwen35Model<QWEN35_0_8B, float, float, float>;
template class Qwen35Model<QWEN35_2B, float, float, float>;
template class Qwen35Model<QWEN35_4B, float, float, float>;
template class Qwen35Model<QWEN35_9B, float, float, float>;
