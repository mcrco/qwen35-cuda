#include "GroupQueryAttention.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

#include <cfloat>

namespace group_query_attention_detail {

constexpr int GQA_THREADS = 256;
constexpr int GQA_WARP_SIZE = 32;
constexpr int GQA_FLASH_TILE_TOKENS = GQA_THREADS / GQA_WARP_SIZE;

template<typename compute_t>
__device__ inline compute_t sigmoid(compute_t x) {
    return static_cast<compute_t>(1) / (static_cast<compute_t>(1) + exp(-x));
}

// Warp-level reduction that sums up all GQA_WARP_SIZE elements.
template<typename compute_t>
__device__ inline compute_t warp_sum(compute_t value) {
    for (int offset = GQA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        // basically value += shared_mem[tid + offset]
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template<Qwen35Size QWEN35_SIZE, typename query_t, typename key_t, typename value_t, typename output_t, typename gate_t, typename compute_t>
__global__ void flashDecodeKernel(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, const gate_t *gate, int32_t layer_num, int32_t token_pos) {
    using Qwen35Config = Qwen35Config<QWEN35_SIZE>;
    static_assert(Qwen35Config::head_size() <= GQA_THREADS);

    // Tiled gemm partials.
    __shared__ compute_t tile_scores[GQA_FLASH_TILE_TOKENS];
    __shared__ compute_t tile_coeffs[GQA_FLASH_TILE_TOKENS];
    // Online softmax partials.
    __shared__ compute_t shared_max;
    __shared__ compute_t shared_denom;
    __shared__ compute_t shared_adjustment_ratio;

    int token_count = token_pos + 1;

    // Assign each block to one query head.
    int qi = blockIdx.x;
    int group_size = Qwen35Config::num_query_heads() / Qwen35Config::num_kv_heads();
    int ki = qi / group_size;

    // Assign each warp to one query token.
    int tid = threadIdx.x;
    int warp_id = tid / GQA_WARP_SIZE;
    // Assign each thread to perform dot product for elements with i == lane + warp_size * k.
    int lane = tid % GQA_WARP_SIZE;

    int d_k = Qwen35Config::head_size();
    compute_t scale = rsqrt(static_cast<compute_t>(d_k));
    compute_t acc = static_cast<compute_t>(0);
    if (tid == 0) {
        shared_max = -FLT_MAX;
        shared_denom = static_cast<compute_t>(0);
    }
    __syncthreads();

    // Compute partial dot products for GQA_FLASH_TILE_TOKENS at a time.
    for (int tile_start = 0; tile_start < token_count; tile_start += GQA_FLASH_TILE_TOKENS) {
        // Get current thread's key token.
        int t = tile_start + warp_id;
        compute_t dot_product = static_cast<compute_t>(0);
        if (t < token_count) {
            // Flattened index for key token.
            int k_base_index =
                t * Qwen35Config::num_layers() * Qwen35Config::keys_size() +
                layer_num * Qwen35Config::keys_size() +
                ki * d_k;
            // Partial dot product for elements in qk vectors with i == lane + warp_size * k.
            for (int d = lane; d < d_k; d += GQA_WARP_SIZE) {
                compute_t qval = gpu_ops::read_as<compute_t>(queries[qi * d_k + d]);
                compute_t kval = gpu_ops::read_as<compute_t>(k_cache[k_base_index + d]);
                dot_product += qval * kval;
            }
            // Reduce sum partial dot product across threads in warp for current query token.
            dot_product = warp_sum(dot_product);
        }

        // Find max among current tile.
        if (lane == 0) {
            tile_scores[warp_id] = t < token_count ? dot_product * scale : -FLT_MAX;
        }
        __syncthreads();

        // Update partials for online softmax.
        if (tid == 0) {
            compute_t tile_max = shared_max;
            for (int i = 0; i < GQA_FLASH_TILE_TOKENS; i++) {
                int tile_t = tile_start + i;
                if (tile_t < token_count) {
                    tile_max = fmax(tile_max, tile_scores[i]);
                }
            }

            shared_adjustment_ratio = exp(shared_max - tile_max);
            compute_t tile_denom = static_cast<compute_t>(0);
            for (int i = 0; i < GQA_FLASH_TILE_TOKENS; i++) {
                int tile_t = tile_start + i;
                compute_t coeff = tile_t < token_count
                    ? exp(tile_scores[i] - tile_max)
                    : static_cast<compute_t>(0);
                tile_coeffs[i] = coeff;
                tile_denom += coeff;
            }

            shared_denom = shared_denom * shared_adjustment_ratio + tile_denom;
            shared_max = tile_max;
        }
        __syncthreads();

        // Update unnormalized weighted values (softmax numerator * value).
        // Assign each thread to one of the output vector elements.
        for (int d = tid; d < d_k; d += blockDim.x) {
            acc *= shared_adjustment_ratio;
            for (int i = 0; i < GQA_FLASH_TILE_TOKENS; i++) {
                int tile_t = tile_start + i;
                if (tile_t < token_count) {
                    int v_base_index =
                        tile_t * Qwen35Config::num_layers() * Qwen35Config::values_size() +
                        layer_num * Qwen35Config::values_size() +
                        ki * d_k;
                    compute_t vval = gpu_ops::read_as<compute_t>(v_cache[v_base_index + tid]);
                    acc += tile_coeffs[i] * vval;
                }
            }
        }
        __syncthreads();
    }

    // Assign each thread to one of the output vector elements.
    // Normalize by online softmax denominator (fully accumualted across all tiles by now).
    for (int d = tid; d < d_k; d += blockDim.x) {
        int idx = qi * d_k + tid;
        compute_t gateval = gpu_ops::read_as<compute_t>(gate[idx]);
        weighted_values[idx] = gpu_ops::write_from<output_t>((acc / shared_denom) * sigmoid(gateval));
    }
}

} // namespace group_query_attention_detail

template<Qwen35Size QWEN35_SIZE>
template<typename query_t, typename key_t, typename value_t, typename output_t, typename gate_t, typename compute_t>
void GroupQueryAttention<QWEN35_SIZE>::sdpa(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, const gate_t *gate, int32_t layer_num, int32_t token_pos, cudaStream_t stream) {
    int blocks = Qwen35Config::num_query_heads();
    int threads = group_query_attention_detail::GQA_THREADS;
    group_query_attention_detail::flashDecodeKernel<QWEN35_SIZE, query_t, key_t, value_t, output_t, gate_t, compute_t><<<blocks, threads, 0, stream>>>(queries, k_cache, v_cache, weighted_values, gate, layer_num, token_pos);
    checkCuda(cudaGetLastError());
}

template class GroupQueryAttention<QWEN35_0_8B>;
template class GroupQueryAttention<QWEN35_2B>;
template class GroupQueryAttention<QWEN35_4B>;
template class GroupQueryAttention<QWEN35_9B>;

template void GroupQueryAttention<QWEN35_0_8B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN35_2B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN35_4B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN35_9B>::sdpa<float, float, float, float, float, float>(
    const float*, const float*, const float*, float*, const float*, int32_t, int32_t, cudaStream_t);
