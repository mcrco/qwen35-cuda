#pragma once

#include <memory>
#include <random>
#include <vector>

#include "../CudaBuffer.cuh"
#include "Qwen35Config.h"
#include "Qwen35Layer.cuh"
#include "Qwen35Types.cuh"

template<Qwen35Size QWEN35_SIZE>
class Qwen35Model {
public:
    Qwen35Model();

    std::shared_ptr<CudaBuffer> embedding_weight;
    std::shared_ptr<CudaBuffer> lm_head_weight;
    std::vector<std::shared_ptr<Qwen35Layer<QWEN35_SIZE>>> layers;
    std::shared_ptr<CudaBuffer> final_layernorm;

    Qwen35Cache allocate_cache(size_t max_seq_len) const;
    int32_t forward(Qwen35Cache &cache, int32_t input_tok_id, float temperature);

private:
    std::shared_ptr<CudaBuffer> hidden_state;
    std::shared_ptr<CudaBuffer> norm_hidden_state;
    std::shared_ptr<CudaBuffer> output_scores;
    std::mt19937 rng{0};
};
