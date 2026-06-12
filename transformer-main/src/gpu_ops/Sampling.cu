#include "Sampling.cuh"

#include "../ErrorCheck.h"

#include <cfloat>
#include <stdexcept>

namespace sampling_detail {

constexpr int32_t THREADS = 256;
constexpr int32_t PREFIX_THREADS = 1024;

// Need max kernels for 2-pass numerically stable softmax.
// This kernel gets the max (logit / temperature) in each block.
__global__ void blockMaxKernel(const float *scores, float *block_maxes, int32_t n, float temperature) {
    extern __shared__ float partial_maxes[];

    int32_t tidx = threadIdx.x;
    int32_t start = blockIdx.x * blockDim.x;
    int32_t idx = start + tidx;

    float local_max = -FLT_MAX;
    if (idx < n) {
        local_max = scores[idx] / temperature;
    }
    partial_maxes[tidx] = local_max;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            partial_maxes[tidx] = fmaxf(partial_maxes[tidx], partial_maxes[tidx + stride]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        block_maxes[blockIdx.x] = partial_maxes[0];
    }
}

// Gets the max (logit / temperature) across blocks.
__global__ void reduceMaxKernel(const float *block_maxes, float *global_max, int32_t num_blocks) {
    extern __shared__ float partial_maxes[];

    int32_t tidx = threadIdx.x;
    float local_max = -FLT_MAX;
    for (int32_t i = tidx; i < num_blocks; i += blockDim.x) {
        local_max = fmaxf(local_max, block_maxes[i]);
    }
    partial_maxes[tidx] = local_max;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            partial_maxes[tidx] = fmaxf(partial_maxes[tidx], partial_maxes[tidx + stride]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        *global_max = partial_maxes[0];
    }
}

// Sums unnormalized probabilities (softmax numerators) for CDF.
__global__ void blockSumKernel(const float *scores, float *block_sums, int32_t n, float temperature, const float *global_max) {
    extern __shared__ float partial_sums[];

    int32_t tidx = threadIdx.x;
    int32_t start = blockIdx.x * blockDim.x;
    int32_t idx = start + tidx;

    float local_sum = 0.0f;
    if (idx < n) {
        local_sum = expf(scores[idx] / temperature - *global_max);
    }
    partial_sums[tidx] = local_sum;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            partial_sums[tidx] += partial_sums[tidx + stride];
        }
        __syncthreads();
    }

    if (tidx == 0) {
        block_sums[blockIdx.x] = partial_sums[0];
    }
}

// Parallelized reduction for block sums (softmax denoms), prefix sums (CDF), and total sum (softmax denom).
// Also gets the block containing the sampled token based on CDF.
__global__ void prefixAndSelectBlockKernel(const float *block_sums, float *block_prefix_sums, float *total_sum,
                                           int32_t *selected_block, int32_t num_blocks, float uniform01) {
    __shared__ float prefix[PREFIX_THREADS];
    __shared__ int32_t candidates[PREFIX_THREADS];

    int32_t tidx = threadIdx.x;
    float value = tidx < num_blocks ? block_sums[tidx] : 0.0f;
    prefix[tidx] = value;
    __syncthreads();

    // Upsweep for prefix sum.
    for (int32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        float addend = tidx >= offset ? prefix[tidx - offset] : 0.0f;
        __syncthreads();
        prefix[tidx] += addend;
        __syncthreads();
    }

    // Get exclusive prefix sum.
    float exclusive_prefix = prefix[tidx] - value;
    if (tidx < num_blocks) {
        block_prefix_sums[tidx] = exclusive_prefix;
    }

    // Get total unnormalized probability sum.
    float total = prefix[blockDim.x - 1];
    if (tidx == 0) {
        *total_sum = total;
    }
    // Sample random point in CDF.
    float sample = uniform01 * total;
    // Get all blocks that are past sample point.
    candidates[tidx] = (tidx < num_blocks && sample <= prefix[tidx]) ? tidx : num_blocks - 1;
    __syncthreads();

    // Downsweep reduction to get first block past sample point (must contain desired token).
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            candidates[tidx] = min(candidates[tidx], candidates[tidx + stride]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        *selected_block = candidates[0];
    }
}

// Gets the exact token from with in a sampled block.
__global__ void sampleWithinBlockKernel(const float *scores, const float *block_prefix_sums, const float *total_sum,
                                        const int32_t *selected_block, int32_t *result, int32_t n,
                                        float temperature, const float *global_max, float uniform01) {
    __shared__ float prefix[THREADS];
    __shared__ int32_t candidates[THREADS];

    int32_t tidx = threadIdx.x;
    int32_t block = *selected_block;
    int32_t start = block * THREADS;
    int32_t idx = start + tidx;
    float sample = uniform01 * *total_sum;
    float base_cdf = block_prefix_sums[block];

    // Compute softmax numerator.
    prefix[tidx] = idx < n ? expf(scores[idx] / temperature - *global_max) : 0.0f;
    __syncthreads();

    // Upsweep for prefix sum.
    for (int32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        float addend = tidx >= offset ? prefix[tidx - offset] : 0.0f;
        __syncthreads();
        prefix[tidx] += addend;
        __syncthreads();
    }

    // Same thing as before, use prefix sum to find all tokens past sampled point on CDF.
    float cdf = base_cdf + prefix[tidx];
    candidates[tidx] = (idx < n && sample <= cdf) ? idx : n - 1;
    __syncthreads();

    // Downsweep to find sampled token.
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            candidates[tidx] = min(candidates[tidx], candidates[tidx + stride]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        *result = candidates[0];
    }
}

} // namespace sampling_detail

Sampling::Sampling(int32_t max_len) {
    max_blocks = (max_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    block_maxes = std::make_shared<CudaBuffer>(max_blocks * sizeof(float));
    block_sums = std::make_shared<CudaBuffer>(max_blocks * sizeof(float));
    global_max = std::make_shared<CudaBuffer>(sizeof(float));
    total_sum = std::make_shared<CudaBuffer>(sizeof(float));
    block_prefix_sums = std::make_shared<CudaBuffer>(max_blocks * sizeof(float));
    selected_block = std::make_shared<CudaBuffer>(sizeof(int32_t));
    result = std::make_shared<CudaBuffer>(sizeof(int32_t));
}

int32_t *Sampling::sample(const std::shared_ptr<CudaBuffer> &scores_buffer, int32_t n, float temperature, float uniform01, cudaStream_t stream) {
    if (n <= 0 || (n + CHUNK_SIZE - 1) / CHUNK_SIZE > max_blocks) {
        throw std::runtime_error("invalid sampling length");
    }

    const float *scores = static_cast<const float *>(scores_buffer->data);
    int32_t num_blocks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_blocks > sampling_detail::PREFIX_THREADS) {
        throw std::runtime_error("sampling length exceeds parallel prefix capacity");
    }
    int32_t shared_mem_size = CHUNK_SIZE * sizeof(float);

    // Get local block max logits for numerically stable softmax.
    sampling_detail::blockMaxKernel<<<num_blocks, CHUNK_SIZE, shared_mem_size, stream>>>(
        scores, static_cast<float *>(block_maxes->data), n, temperature);
    checkCuda(cudaGetLastError());

    // Get global max logit.
    sampling_detail::reduceMaxKernel<<<1, sampling_detail::THREADS, sampling_detail::THREADS * sizeof(float), stream>>>(
        static_cast<const float *>(block_maxes->data), static_cast<float *>(global_max->data), num_blocks);
    checkCuda(cudaGetLastError());

    // Get local block softmax denominators.
    sampling_detail::blockSumKernel<<<num_blocks, CHUNK_SIZE, shared_mem_size, stream>>>(
        scores, static_cast<float *>(block_sums->data), n, temperature, static_cast<const float *>(global_max->data));
    checkCuda(cudaGetLastError());

    // Compute block sums, inclusive/exclusive prefix sums, and total sum.
    // Also samples a chunk.
    sampling_detail::prefixAndSelectBlockKernel<<<1, sampling_detail::PREFIX_THREADS, 0, stream>>>(
        static_cast<const float *>(block_sums->data), static_cast<float *>(block_prefix_sums->data),
        static_cast<float *>(total_sum->data), static_cast<int32_t *>(selected_block->data), num_blocks, uniform01);
    checkCuda(cudaGetLastError());

    // Find sampled token within sampled chunk.
    sampling_detail::sampleWithinBlockKernel<<<1, sampling_detail::THREADS, 0, stream>>>(
        scores, static_cast<const float *>(block_prefix_sums->data), static_cast<const float *>(total_sum->data),
        static_cast<const int32_t *>(selected_block->data), static_cast<int32_t *>(result->data), n,
        temperature, static_cast<const float *>(global_max->data), uniform01);
    checkCuda(cudaGetLastError());

    return static_cast<int32_t *>(result->data);
}
