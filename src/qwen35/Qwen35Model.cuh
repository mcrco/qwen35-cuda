#pragma once

#include <memory>
#include <random>
#include <vector>

#include "../CudaBuffer.cuh"
#include "../gpu_ops/ArgMax.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/Sampling.cuh"
#include "Qwen35Config.h"
#include "Qwen35Layer.cuh"
#include "Qwen35Types.cuh"

template<
    Qwen35Size QWEN35_SIZE,
    typename weight_t = float,
    typename hidden_t = float,
    typename compute_t = float>
class Qwen35Model {
public:
    Qwen35Model();
    ~Qwen35Model();

    std::shared_ptr<CudaBuffer> embedding_weight;
    std::shared_ptr<CudaBuffer> lm_head_weight;
    std::vector<std::shared_ptr<Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t>>> layers;
    LayerNorm final_layernorm;

    Qwen35Cache allocate_cache(size_t max_seq_len) const;
    int32_t forward(Qwen35Cache &cache, int32_t input_tok_id, float temperature);
    void set_seed(uint32_t seed);
    cudaStream_t cuda_stream() const;

private:
    cudaStream_t stream;
    std::shared_ptr<CudaBuffer> hidden_state;
    std::shared_ptr<CudaBuffer> norm_hidden_state;
    std::shared_ptr<CudaBuffer> output_scores;
    ArgMax argmax{Qwen35Config<QWEN35_SIZE>::vocab_size()};
    Sampling sampling{Qwen35Config<QWEN35_SIZE>::vocab_size()};
    std::mt19937 rng{0};
};
