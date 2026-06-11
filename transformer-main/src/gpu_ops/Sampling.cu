#include "Sampling.cuh"

#include "../ErrorCheck.h"

#include <cfloat>
#include <stdexcept>

namespace sampling_detail {

constexpr int32_t THREADS = 256;

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

// Sums probabilities for CDF.
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

__global__ void prefixAndSelectBlockKernel(const float *block_sums, float *block_prefix_sums, float *total_sum,
                                           int32_t *selected_block, int32_t num_blocks, float uniform01) {
    if (threadIdx.x != 0) {
        return;
    }

    float total = 0.0f;
    for (int32_t i = 0; i < num_blocks; i++) {
        block_prefix_sums[i] = total;
        total += block_sums[i];
    }

    *total_sum = total;
    float sample = uniform01 * total;
    int32_t selected = num_blocks - 1;
    for (int32_t i = 0; i < num_blocks; i++) {
        if (sample <= block_prefix_sums[i] + block_sums[i]) {
            selected = i;
            break;
        }
    }
    *selected_block = selected;
}

__global__ void sampleWithinBlockKernel(const float *scores, const float *block_prefix_sums, const float *total_sum,
                                        const int32_t *selected_block, int32_t *result, int32_t n,
                                        float temperature, const float *global_max, float uniform01) {
    if (threadIdx.x != 0) {
        return;
    }

    int32_t block = *selected_block;
    int32_t start = block * THREADS;
    int32_t end = min(start + THREADS, n);
    float sample = uniform01 * *total_sum;
    float cdf = block_prefix_sums[block];

    for (int32_t i = start; i < end; i++) {
        cdf += expf(scores[i] / temperature - *global_max);
        if (sample <= cdf) {
            *result = i;
            return;
        }
    }

    *result = n - 1;
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
    int32_t shared_mem_size = CHUNK_SIZE * sizeof(float);

    sampling_detail::blockMaxKernel<<<num_blocks, CHUNK_SIZE, shared_mem_size, stream>>>(
        scores, static_cast<float *>(block_maxes->data), n, temperature);
    checkCuda(cudaGetLastError());

    sampling_detail::reduceMaxKernel<<<1, sampling_detail::THREADS, sampling_detail::THREADS * sizeof(float), stream>>>(
        static_cast<const float *>(block_maxes->data), static_cast<float *>(global_max->data), num_blocks);
    checkCuda(cudaGetLastError());

    sampling_detail::blockSumKernel<<<num_blocks, CHUNK_SIZE, shared_mem_size, stream>>>(
        scores, static_cast<float *>(block_sums->data), n, temperature, static_cast<const float *>(global_max->data));
    checkCuda(cudaGetLastError());

    sampling_detail::prefixAndSelectBlockKernel<<<1, 1, 0, stream>>>(
        static_cast<const float *>(block_sums->data), static_cast<float *>(block_prefix_sums->data),
        static_cast<float *>(total_sum->data), static_cast<int32_t *>(selected_block->data), num_blocks, uniform01);
    checkCuda(cudaGetLastError());

    sampling_detail::sampleWithinBlockKernel<<<1, 1, 0, stream>>>(
        scores, static_cast<const float *>(block_prefix_sums->data), static_cast<const float *>(total_sum->data),
        static_cast<const int32_t *>(selected_block->data), static_cast<int32_t *>(result->data), n,
        temperature, static_cast<const float *>(global_max->data), uniform01);
    checkCuda(cudaGetLastError());

    return static_cast<int32_t *>(result->data);
}
