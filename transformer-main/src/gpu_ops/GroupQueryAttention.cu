#include "../qwen2/Qwen2Config.h"
#include <cfloat>
#include "GroupQueryAttention.cuh"
#include <cuda_bf16.h>
#include <memory>
#include "../ErrorCheck.h"

const int GQA_BLOCKS = 1;

template<Qwen2Size QWEN2_SIZE>
__global__ void sdpaKernel(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len) {
    /*
     * Flash attention tiled approach seems to complicated (and not the scope of
     * the project), so the next best parallelization approach that I'm gonna
     * use is a assigning each thread to one softmax(qK^T / sqrt(d_k)) * V, e.g.
     * each query and kv head pair.
     *
     * This works well because we need each thread to manage one softmax, as
     * required by the online softmax algorithm.
     */

    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    // Every thread corresponds to a query head. q_i = query head index.
    int qi = threadIdx.x + blockDim.x * blockIdx.x;

    // Get which kv head to use.
    int group_size = Qwen2Config::num_query_heads() / Qwen2Config::num_kv_heads();
    int ki = qi / group_size;

    int d_k = Qwen2Config::head_size();
    float max = -FLT_MAX;
    float denom = 0.0;
    for (int t = 0; t < seq_len; t++) {
        // kv shape is shape (seq_len, num_layers, num_kv_heads, d_k/dv)
        // so the flattened indices for the kv start at
        int k_base_index =
            t * Qwen2Config::num_layers() * Qwen2Config::keys_size() +
            layer_num * Qwen2Config::keys_size() +
            ki * d_k;
        int v_base_index =
            t * Qwen2Config::num_layers() * Qwen2Config::values_size() +
            layer_num * Qwen2Config::values_size() +
            ki * d_k;
        // k_base_index and v_base_index should be the same.

        // Calculate dot product q * K[t]
        float dot_product = 0.0;
        for (int i = 0; i < d_k; i++) {
            float qval = __bfloat162float(queries[qi * d_k + i]);
            float kval = __bfloat162float(k_cache[k_base_index + i]);
            dot_product += qval * kval;
        }
        dot_product /= sqrtf(d_k);

        // online softmax update
        float new_max = fmaxf(max, dot_product);
        float adjustment_ratio = expf(max - new_max);
        float score = expf(dot_product - new_max);
        denom = denom * adjustment_ratio + score;
        for (int i = 0; i < d_k; i++) {
            float vval = __bfloat162float(v_cache[v_base_index + i]);
            float wval = weighted_values[qi * d_k + i];
            weighted_values[qi * d_k + i] = wval * adjustment_ratio + score * vval;
        }
        max = new_max;
    }
    for (int i = 0; i < d_k; i++) {
        weighted_values[qi * d_k + i] /= denom;
    }
}

template<Qwen2Size QWEN2_SIZE>
void GroupQueryAttention<QWEN2_SIZE>::sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
    int threads = Qwen2Config::num_query_heads();
    sdpaKernel<QWEN2_SIZE><<<GQA_BLOCKS, threads, 0, stream>>>(queries, k_cache, v_cache, weighted_values, layer_num, seq_len);
    checkCuda(cudaGetLastError());
}

template class GroupQueryAttention<QWEN2_0_5B>;
