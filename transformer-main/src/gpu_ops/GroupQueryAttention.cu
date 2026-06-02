#include "../qwen2/Qwen2Config.h"
#include "GroupQueryAttention.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

#include <cfloat>

namespace group_query_attention_detail {

constexpr int GQA_BLOCKS = 1;

template<Qwen2Size QWEN2_SIZE, typename query_t, typename key_t, typename value_t, typename output_t, typename compute_t>
__global__ void sdpaKernel(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, int32_t layer_num, int32_t seq_len) {
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    int qi = threadIdx.x + blockDim.x * blockIdx.x;

    int group_size = Qwen2Config::num_query_heads() / Qwen2Config::num_kv_heads();
    int ki = qi / group_size;

    int d_k = Qwen2Config::head_size();
    compute_t max = -FLT_MAX;
    compute_t denom = static_cast<compute_t>(0);
    for (int t = 0; t < seq_len; t++) {
        int k_base_index =
            t * Qwen2Config::num_layers() * Qwen2Config::keys_size() +
            layer_num * Qwen2Config::keys_size() +
            ki * d_k;
        int v_base_index =
            t * Qwen2Config::num_layers() * Qwen2Config::values_size() +
            layer_num * Qwen2Config::values_size() +
            ki * d_k;

        compute_t dot_product = static_cast<compute_t>(0);
        for (int i = 0; i < d_k; i++) {
            compute_t qval = gpu_ops::read_as<compute_t>(queries[qi * d_k + i]);
            compute_t kval = gpu_ops::read_as<compute_t>(k_cache[k_base_index + i]);
            dot_product += qval * kval;
        }
        dot_product /= sqrt(static_cast<compute_t>(d_k));

        compute_t new_max = fmax(max, dot_product);
        compute_t adjustment_ratio = exp(max - new_max);
        compute_t score = exp(dot_product - new_max);
        denom = denom * adjustment_ratio + score;
        for (int i = 0; i < d_k; i++) {
            compute_t vval = gpu_ops::read_as<compute_t>(v_cache[v_base_index + i]);
            compute_t wval = gpu_ops::read_as<compute_t>(weighted_values[qi * d_k + i]);
            weighted_values[qi * d_k + i] = gpu_ops::write_from<output_t>(wval * adjustment_ratio + score * vval);
        }
        max = new_max;
    }
    for (int i = 0; i < d_k; i++) {
        compute_t wval = gpu_ops::read_as<compute_t>(weighted_values[qi * d_k + i]);
        weighted_values[qi * d_k + i] = gpu_ops::write_from<output_t>(wval / denom);
    }
}

} // namespace group_query_attention_detail

template<Qwen2Size QWEN2_SIZE>
template<typename query_t, typename key_t, typename value_t, typename output_t, typename compute_t>
void GroupQueryAttention<QWEN2_SIZE>::sdpa(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
    int threads = Qwen2Config::num_query_heads();
    group_query_attention_detail::sdpaKernel<QWEN2_SIZE, query_t, key_t, value_t, output_t, compute_t><<<group_query_attention_detail::GQA_BLOCKS, threads, 0, stream>>>(queries, k_cache, v_cache, weighted_values, layer_num, seq_len);
    checkCuda(cudaGetLastError());
}

template<Qwen2Size QWEN2_SIZE>
void GroupQueryAttention<QWEN2_SIZE>::sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
    sdpa<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, float>(queries, k_cache, v_cache, weighted_values, layer_num, seq_len, stream);
}

template class GroupQueryAttention<QWEN2_0_5B>;

template void GroupQueryAttention<QWEN2_0_5B>::sdpa<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, float>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN2_0_5B>::sdpa<float, float, float, float, float>(
    const float*, const float*, const float*, float*, int32_t, int32_t, cudaStream_t);
