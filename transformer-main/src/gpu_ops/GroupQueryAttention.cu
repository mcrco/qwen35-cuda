#include "GroupQueryAttention.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

#include <cfloat>

namespace group_query_attention_detail {

constexpr int GQA_BLOCKS = 1;

template<typename compute_t>
__device__ inline compute_t sigmoid(compute_t x) {
    return static_cast<compute_t>(1) / (static_cast<compute_t>(1) + exp(-x));
}

template<Qwen35Size QWEN35_SIZE, typename query_t, typename key_t, typename value_t, typename output_t, typename gate_t, typename compute_t>
__global__ void sdpaKernel(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, const gate_t *gate, int32_t layer_num, int32_t token_pos) {
    using Qwen35Config = Qwen35Config<QWEN35_SIZE>;

    int qi = threadIdx.x + blockDim.x * blockIdx.x;

    int group_size = Qwen35Config::num_query_heads() / Qwen35Config::num_kv_heads();
    int ki = qi / group_size;

    int d_k = Qwen35Config::head_size();
    compute_t max = -FLT_MAX;
    compute_t denom = static_cast<compute_t>(0);
    for (int i = 0; i < d_k; i++) {
        weighted_values[qi * d_k + i] = gpu_ops::write_from<output_t>(static_cast<compute_t>(0));
    }
    for (int t = 0; t <= token_pos; t++) {
        int k_base_index =
            t * Qwen35Config::num_layers() * Qwen35Config::keys_size() +
            layer_num * Qwen35Config::keys_size() +
            ki * d_k;
        int v_base_index =
            t * Qwen35Config::num_layers() * Qwen35Config::values_size() +
            layer_num * Qwen35Config::values_size() +
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
        int idx = qi * d_k + i;
        compute_t wval = gpu_ops::read_as<compute_t>(weighted_values[idx]);
        compute_t gateval = gpu_ops::read_as<compute_t>(gate[idx]);
        weighted_values[idx] = gpu_ops::write_from<output_t>((wval / denom) * sigmoid(gateval));
    }
}

} // namespace group_query_attention_detail

template<Qwen35Size QWEN35_SIZE>
template<typename query_t, typename key_t, typename value_t, typename output_t, typename gate_t, typename compute_t>
void GroupQueryAttention<QWEN35_SIZE>::sdpa(const query_t *queries, const key_t *k_cache, const value_t *v_cache, output_t *weighted_values, const gate_t *gate, int32_t layer_num, int32_t token_pos, cudaStream_t stream) {
    int threads = Qwen35Config::num_query_heads();
    group_query_attention_detail::sdpaKernel<QWEN35_SIZE, query_t, key_t, value_t, output_t, gate_t, compute_t><<<group_query_attention_detail::GQA_BLOCKS, threads, 0, stream>>>(queries, k_cache, v_cache, weighted_values, gate, layer_num, token_pos);
    checkCuda(cudaGetLastError());
}

template class GroupQueryAttention<QWEN35_0_8B>;
template class GroupQueryAttention<QWEN35_4B>;
template class GroupQueryAttention<QWEN35_9B>;

template void GroupQueryAttention<QWEN35_0_8B>::sdpa<input_float_t, input_float_t, input_float_t, input_float_t, input_float_t, float>(
    const input_float_t*, const input_float_t*, const input_float_t*, input_float_t*, const input_float_t*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN35_4B>::sdpa<input_float_t, input_float_t, input_float_t, input_float_t, input_float_t, float>(
    const input_float_t*, const input_float_t*, const input_float_t*, input_float_t*, const input_float_t*, int32_t, int32_t, cudaStream_t);
template void GroupQueryAttention<QWEN35_9B>::sdpa<input_float_t, input_float_t, input_float_t, input_float_t, input_float_t, float>(
    const input_float_t*, const input_float_t*, const input_float_t*, input_float_t*, const input_float_t*, int32_t, int32_t, cudaStream_t);
