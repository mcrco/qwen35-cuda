#pragma once

#include "../CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

class Sampling {
    static constexpr int32_t CHUNK_SIZE = 256;

    std::shared_ptr<CudaBuffer> block_maxes;
    std::shared_ptr<CudaBuffer> block_sums;
    std::shared_ptr<CudaBuffer> global_max;
    std::shared_ptr<CudaBuffer> total_sum;
    std::shared_ptr<CudaBuffer> block_prefix_sums;
    std::shared_ptr<CudaBuffer> selected_block;
    std::shared_ptr<CudaBuffer> result;
    int32_t max_blocks;

public:
    explicit Sampling(int32_t max_len);

    int32_t *sample(const std::shared_ptr<CudaBuffer> &scores_buffer, int32_t n, float temperature, float uniform01, cudaStream_t stream);
};
