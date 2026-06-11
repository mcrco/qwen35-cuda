#include <memory>
#include "../CudaBuffer.cuh"
#include <random>
#include <cuda_bf16.h>
#include "../gpu_ops/RoPE.cuh"
#include "TestUtils.cuh"

template<typename x_t>
static float normalize_test_value(x_t x) {
    return static_cast<float>(x);
}

template<>
float normalize_test_value<__nv_bfloat16>(__nv_bfloat16 x) {
    return static_cast<float>(x);
}

static __nv_bfloat16 test_value_from_float_bf16(float x) {
    return __nv_bfloat16{x};
}

template<typename x_t>
static x_t test_value_from_float(float x) {
    return static_cast<x_t>(x);
}

template<>
__nv_bfloat16 test_value_from_float<__nv_bfloat16>(float x) {
    return test_value_from_float_bf16(x);
}

template<typename x_t>
void check_rope_allclose(x_t *gpu_vals, x_t *cpu_vals, int32_t len) {
    check_fp32_allclose(gpu_vals, cpu_vals, len);
}

template<>
void check_rope_allclose<__nv_bfloat16>(__nv_bfloat16 *gpu_vals, __nv_bfloat16 *cpu_vals, int32_t len) {
    check_bf16_allclose(gpu_vals, cpu_vals, len);
}

template<typename x_t>
void test_rope(int32_t num_heads, int32_t head_dim, int32_t rotary_dim, float theta_base) {
    int32_t position_idx = 13;

    auto queries = std::make_shared<CudaBuffer>(num_heads * head_dim * sizeof(x_t));
    x_t *queries_ptr = static_cast<x_t*>(queries->data);

    // seeded random
    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 1.0f);
    for (int32_t i = 0; i < num_heads * head_dim; i++) {
        queries_ptr[i] = test_value_from_float<x_t>(distribution(generator));
    }

    float cos_vals[rotary_dim / 2];
    float sin_vals[rotary_dim / 2];
    for (int32_t theta_idx = 0; theta_idx < rotary_dim / 2; theta_idx++) {
        float theta_idx_frac = static_cast<float>(theta_idx) / static_cast<float>(rotary_dim / 2);
        float theta = powf(theta_base, -theta_idx_frac);
        float angle = theta * static_cast<float>(position_idx);
        cos_vals[theta_idx] = cosf(angle);
        sin_vals[theta_idx] = sinf(angle);
    }

    x_t cpu_out[head_dim * num_heads];
    for (int32_t head = 0; head < num_heads; head++) {
        x_t *cpu_out_row = cpu_out + head * head_dim;
        x_t *in_row = queries_ptr + head * head_dim;

        float in_row_rotated_half[rotary_dim];
        for (int32_t i = 0; i < rotary_dim / 2; i++) {
            in_row_rotated_half[i] = -normalize_test_value(in_row[i + rotary_dim / 2]);
        }
        for (int32_t i = rotary_dim / 2; i < rotary_dim; i++) {
            in_row_rotated_half[i] = normalize_test_value(in_row[i - rotary_dim / 2]);
        }

        for (int32_t i = 0; i < rotary_dim; i++) {
            float cos_val = cos_vals[i % (rotary_dim / 2)];
            float sin_val = sin_vals[i % (rotary_dim / 2)];
            cpu_out_row[i] = test_value_from_float<x_t>(normalize_test_value(in_row[i]) * cos_val + in_row_rotated_half[i] * sin_val);
        }
        for (int32_t i = rotary_dim; i < head_dim; i++) {
            cpu_out_row[i] = in_row[i];
        }
    }

    RoPE::apply_rope_to_qk(queries_ptr, num_heads, head_dim, rotary_dim, position_idx, theta_base, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    check_rope_allclose(queries_ptr, cpu_out, num_heads * head_dim);
}

int main() {
    // Full-head BF16 RoPE coverage
    test_rope<__nv_bfloat16>(14, 64, 64, 1e6f);
    // Qwen3.5 partial RoPE shape
    test_rope<float>(16, 256, 64, 10000000.0f);
}
