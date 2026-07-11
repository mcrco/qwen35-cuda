#include "../CudaBuffer.cuh"
#include "../cpu_ops/LinearAttention.cuh"
#include "../gpu_ops/LinearAttention.cuh"
#include "TestUtils.cuh"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

void normalize_rows(std::vector<float> &values, int32_t rows, int32_t cols, float scale) {
    for (int32_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int32_t col = 0; col < cols; col++) {
            float value = values[row * cols + col];
            sum += value * value;
        }
        float coeff = scale / std::sqrt(sum + 1.0e-6f);
        for (int32_t col = 0; col < cols; col++) {
            values[row * cols + col] *= coeff;
        }
    }
}

void copy_to_cuda(const std::vector<float> &src, CudaBuffer &dst) {
    std::copy(src.begin(), src.end(), static_cast<float *>(dst.data));
}

void test_linear_attention(
        int32_t num_key_heads,
        int32_t num_value_heads,
        int32_t key_head_dim,
        int32_t value_head_dim,
        int32_t conv_kernel_dim) {
    int32_t keys_size = num_key_heads * key_head_dim;
    int32_t values_size = num_value_heads * value_head_dim;
    int32_t conv_size = 2 * keys_size + values_size;
    int32_t recurrent_state_size = num_value_heads * key_head_dim * value_head_dim;

    std::mt19937 generator{123};
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    std::vector<float> qkv(conv_size);
    std::vector<float> conv_state(conv_kernel_dim * conv_size);
    std::vector<float> conv_weight(conv_size * conv_kernel_dim);
    std::vector<float> conv_bias(conv_size);
    std::vector<float> beta_raw(num_value_heads);
    std::vector<float> decay_raw(num_value_heads);
    std::vector<float> dt_bias(num_value_heads);
    std::vector<float> a_log(num_value_heads);
    std::vector<float> state(recurrent_state_size);

    for (float &value : qkv) value = distribution(generator);
    for (float &value : conv_state) value = distribution(generator);
    for (float &value : conv_weight) value = distribution(generator);
    for (float &value : conv_bias) value = distribution(generator);
    for (float &value : beta_raw) value = distribution(generator);
    for (float &value : decay_raw) value = distribution(generator);
    for (float &value : dt_bias) value = distribution(generator);
    for (float &value : a_log) value = distribution(generator);
    for (float &value : state) value = distribution(generator);

    std::vector<float> cpu_conv_state = conv_state;
    std::vector<float> cpu_mixed_qkv(conv_size);
    std::vector<float> cpu_queries(values_size);
    std::vector<float> cpu_keys(values_size);
    std::vector<float> cpu_values(values_size);
    std::vector<float> cpu_state = state;
    std::vector<float> cpu_weighted_values(values_size);

    CpuLinearAttention::conv1d_silu(
        qkv.data(), cpu_conv_state.data(), conv_weight.data(), conv_bias.data(),
        cpu_mixed_qkv.data(), conv_kernel_dim, conv_size);
    CpuLinearAttention::split_qkv(
        cpu_mixed_qkv.data(), cpu_queries.data(), cpu_keys.data(), cpu_values.data(),
        num_key_heads, num_value_heads, key_head_dim, value_head_dim);
    normalize_rows(
        cpu_queries, num_value_heads, key_head_dim,
        1.0f / std::sqrt(static_cast<float>(key_head_dim)));
    normalize_rows(cpu_keys, num_value_heads, key_head_dim, 1.0f);
    CpuLinearAttention::gated_delta_update(
        cpu_state.data(), cpu_queries.data(), cpu_keys.data(), cpu_values.data(),
        beta_raw.data(), decay_raw.data(), dt_bias.data(), a_log.data(),
        cpu_weighted_values.data(), num_value_heads, key_head_dim, value_head_dim);

    CudaBuffer d_qkv(qkv.size() * sizeof(float));
    CudaBuffer d_conv_state(conv_state.size() * sizeof(float));
    CudaBuffer d_conv_weight(conv_weight.size() * sizeof(float));
    CudaBuffer d_conv_bias(conv_bias.size() * sizeof(float));
    CudaBuffer d_mixed_qkv(conv_size * sizeof(float));
    CudaBuffer d_beta_raw(beta_raw.size() * sizeof(float));
    CudaBuffer d_decay_raw(decay_raw.size() * sizeof(float));
    CudaBuffer d_dt_bias(dt_bias.size() * sizeof(float));
    CudaBuffer d_a_log(a_log.size() * sizeof(float));
    CudaBuffer d_state(state.size() * sizeof(float));
    CudaBuffer d_weighted_values(values_size * sizeof(float));

    copy_to_cuda(qkv, d_qkv);
    copy_to_cuda(conv_state, d_conv_state);
    copy_to_cuda(conv_weight, d_conv_weight);
    copy_to_cuda(conv_bias, d_conv_bias);
    copy_to_cuda(beta_raw, d_beta_raw);
    copy_to_cuda(decay_raw, d_decay_raw);
    copy_to_cuda(dt_bias, d_dt_bias);
    copy_to_cuda(a_log, d_a_log);
    copy_to_cuda(state, d_state);

    LinearAttention::conv1d_silu<float, float, float>(
        static_cast<float *>(d_qkv.data), static_cast<float *>(d_conv_state.data),
        static_cast<float *>(d_conv_weight.data), static_cast<float *>(d_conv_bias.data),
        static_cast<float *>(d_mixed_qkv.data), conv_kernel_dim, conv_size, cudaStreamPerThread);
    LinearAttention::normalize_mixed_qk<float, float>(
        static_cast<float *>(d_mixed_qkv.data), num_key_heads, key_head_dim, cudaStreamPerThread);
    LinearAttention::gated_delta_update<float, float, float>(
        static_cast<float *>(d_state.data), static_cast<float *>(d_mixed_qkv.data),
        static_cast<float *>(d_beta_raw.data), static_cast<float *>(d_decay_raw.data),
        static_cast<float *>(d_dt_bias.data), static_cast<float *>(d_a_log.data),
        static_cast<float *>(d_weighted_values.data), num_key_heads, num_value_heads,
        key_head_dim, value_head_dim, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    check_fp32_allclose(static_cast<float *>(d_weighted_values.data), cpu_weighted_values.data(), values_size, 1e-4f, 1e-5f);
    check_fp32_allclose(static_cast<float *>(d_state.data), cpu_state.data(), recurrent_state_size, 1e-4f, 1e-5f);
}

} // namespace

int main() {
    test_linear_attention(2, 2, 8, 8, 4);
    test_linear_attention(2, 4, 8, 8, 4);
    test_linear_attention(16, 32, 128, 128, 4);
}
