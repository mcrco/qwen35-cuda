#pragma once

#include <cstdint>
#include <memory>

#include "Qwen2Layer.cuh"
#include "Qwen2Config.h"
#include "../ErrorCheck.h"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/ArgMax.cuh"

template<
    Qwen2Size QWEN2_SIZE,
    typename weight_t = __nv_bfloat16,
    typename hidden_t = __nv_bfloat16,
    typename compute_t = float,
    typename cache_t = hidden_t,
    typename logits_t = hidden_t>
class Qwen2Model {
    cudaStream_t stream;
    std::shared_ptr<CudaBuffer> hidden_state;
    std::shared_ptr<CudaBuffer> output_scores;
    ArgMax argmax{Qwen2Config::vocab_size()};
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    using Qwen2Layer = Qwen2Layer<QWEN2_SIZE, weight_t, hidden_t, compute_t, cache_t>;
    using WeightT = weight_t;
    using HiddenT = hidden_t;
    using ComputeT = compute_t;
    using CacheT = cache_t;
    using LogitsT = logits_t;

    Qwen2Model() {
        hidden_state = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(hidden_t));
        output_scores = std::make_shared<CudaBuffer>(Qwen2Config::vocab_size() * sizeof(logits_t));
        checkCuda(cudaStreamCreate(&stream));
    }

    ~Qwen2Model() {
        checkCuda(cudaStreamDestroy(stream));
    }

    std::shared_ptr<CudaBuffer> embedding_weight; // (vocab_size, hidden_size)
    std::shared_ptr<Qwen2Layer> layers[Qwen2Config::num_layers()];
    LayerNorm final_layernorm{Qwen2Config::hidden_size()}; // (hidden_size,)

    /**
     *
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param seq_len current sequence length
     * @param input_tok_id last token in the sequence
     * @param temperature Sampling parameter. Always set to 0, for deterministic (greedy) decoding, see https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting.
     *                    You do not need to implement any other sampling methods.
     * @return
     */
    int32_t forward(const std::shared_ptr<CudaBuffer> &k_cache, const std::shared_ptr<CudaBuffer> &v_cache, int32_t seq_len, int32_t input_tok_id, float temperature) {
        // Get token embedding.
        // First get pointers from buffers.
        hidden_t *hidden_state_ptr = static_cast<hidden_t*>(hidden_state->data);
        const weight_t *embedding_ptr = static_cast<const weight_t*>(embedding_weight->data);
        const weight_t *token_embedding_offset = embedding_ptr + input_tok_id * Qwen2Config::hidden_size();
        // Copy in specific embedding for input token to hidden state.
        qwen2_layer_detail::convert_copy<weight_t, hidden_t, compute_t>(token_embedding_offset, hidden_state_ptr, Qwen2Config::hidden_size(), stream);
        // Ensure embedding is copied before we apply any layers.
        checkCuda(cudaStreamSynchronize(stream));

        // Apply each layer to hidden state.
        for (int layer_num = 0; layer_num < Qwen2Config::num_layers(); layer_num++) {
            layers[layer_num]->forward(k_cache, v_cache, hidden_state, seq_len, stream);
        }

        // Final layernorm.
        final_layernorm.normalize_hidden_state<hidden_t, weight_t, hidden_t, compute_t>(hidden_state, hidden_state, Qwen2Config::hidden_size(), stream);

        // Matmul the embeddings by the final hidden state to get logits.
        logits_t *output_scores_ptr = static_cast<logits_t*>(output_scores->data);
        MatrixVectorMultiply::matmul<weight_t, weight_t, hidden_t, logits_t, compute_t>(
            Qwen2Config::vocab_size(), Qwen2Config::hidden_size(), embedding_ptr, nullptr, hidden_state_ptr, output_scores_ptr, stream);
        // Argmax to find the most likely token.
        int32_t *next_token_idx_gpu = argmax.argmax_as_float<logits_t>(output_scores, Qwen2Config::vocab_size(), stream);

        // Copy to host memory and return.
        int32_t next_token_idx_cpu;
        checkCuda(cudaMemcpyAsync(&next_token_idx_cpu, next_token_idx_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        // Ensure result was copied before returning.
        checkCuda(cudaStreamSynchronize(stream));

        if (next_token_idx_cpu < 0 || next_token_idx_cpu >= Qwen2Config::vocab_size()) {
            throw std::runtime_error("invalid token id from argmax");
        }
        return next_token_idx_cpu;
    }
};
