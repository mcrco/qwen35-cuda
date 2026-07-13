#include "../CudaBuffer.cuh"
#include "../HostBuffer.h"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../qwen35/Qwen35Config.h"
#include <random>
#include "TestUtils.cuh"

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<Qwen35Size QWEN35_SIZE>
void test_gqa(int32_t max_seq_len, int32_t token_pos, int32_t layer_num) {
    using Qwen35Config = Qwen35Config<QWEN35_SIZE>;

    CudaBuffer k_cache{max_seq_len * Qwen35Config::num_layers() * Qwen35Config::keys_size() * sizeof(float)};
    CudaBuffer v_cache{max_seq_len * Qwen35Config::num_layers() * Qwen35Config::values_size() * sizeof(float)};
    CudaBuffer queries{Qwen35Config::queries_size() * sizeof(float)};
    CudaBuffer gate{Qwen35Config::queries_size() * sizeof(float)};
    CudaBuffer out{Qwen35Config::queries_size() * sizeof(float)};
    HostBuffer out_cpu{Qwen35Config::queries_size() * sizeof(float)};

    std::mt19937 generator{123};
    std::uniform_int_distribution distribution(-8, 7);
    fill_random_fp32_storage(k_cache, distribution, generator);
    fill_random_fp32_storage(v_cache, distribution, generator);
    fill_random_fp32_storage(queries, distribution, generator);
    fill_random_fp32_storage(gate, distribution, generator);

    // group query attention
    for (int32_t head_idx = 0; head_idx < Qwen35Config::num_query_heads(); head_idx++) {
        float *head_query = static_cast<float*>(queries.data) + head_idx * Qwen35Config::head_size();
        // each key is used multiple times for different queries
        int32_t key_idx = head_idx * Qwen35Config::num_kv_heads() / Qwen35Config::num_query_heads();

        float *out_row = static_cast<float*>(out_cpu.data) + head_idx * Qwen35Config::head_size();
        for (int32_t el_idx = 0; el_idx < Qwen35Config::head_size(); el_idx++) {
            out_row[el_idx] = static_cast<float>(0.0f);
        }

        // QK^T/sqrt(d_k), with the same online stable softmax update as the GPU kernel
        float max_score = -INFINITY;
        float denom = 0.0f;
        for (int32_t sequence_pos = 0; sequence_pos <= token_pos; sequence_pos++) {
            float sum = 0.0f;
            for (int32_t el_idx = 0; el_idx < Qwen35Config::head_size(); el_idx++) {
                float key_el = *(static_cast<float*>(k_cache.data) +
                    sequence_pos * (Qwen35Config::num_layers() * Qwen35Config::keys_size()) +
                    layer_num * Qwen35Config::keys_size() +
                    key_idx * Qwen35Config::head_size() +
                    el_idx
                );
                sum += static_cast<float>(key_el) * static_cast<float>(head_query[el_idx]);
            }
            float scaled_dot_product = sum / sqrtf(Qwen35Config::head_size());
            float new_max = fmaxf(max_score, scaled_dot_product);
            float adjustment_ratio = expf(max_score - new_max);
            float score = expf(scaled_dot_product - new_max);
            denom = denom * adjustment_ratio + score;
            for (int32_t el_idx = 0; el_idx < Qwen35Config::head_size(); el_idx++) {
                float val_el = *(static_cast<float*>(v_cache.data) +
                    sequence_pos * (Qwen35Config::num_layers() * Qwen35Config::values_size()) +
                    layer_num * Qwen35Config::values_size() +
                    key_idx * Qwen35Config::head_size() +
                    el_idx
                );
                float current = static_cast<float>(out_row[el_idx]);
                out_row[el_idx] = static_cast<float>(current * adjustment_ratio + score * static_cast<float>(val_el));
            }
            max_score = new_max;
        }

        // Qwen35 output gating
        float *gate_row = static_cast<float*>(gate.data) + head_idx * Qwen35Config::head_size();
        for (int32_t el_idx = 0; el_idx < Qwen35Config::head_size(); el_idx++) {
            float v = static_cast<float>(out_row[el_idx]) / denom;
            out_row[el_idx] = static_cast<float>(v * sigmoid(static_cast<float>(gate_row[el_idx])));
        }
    }

    for (int run = 0; run < 2; run++) {
        // run twice to ensure that temp space is reused correctly
        GroupQueryAttention<QWEN35_SIZE> gqa{max_seq_len};
        gqa.sdpa(static_cast<float*>(queries.data),
            static_cast<float*>(k_cache.data),
            static_cast<float*>(v_cache.data),
            static_cast<float*>(out.data),
            static_cast<float*>(gate.data),
            layer_num, token_pos, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        check_fp32_storage_allclose(static_cast<float*>(out.data), static_cast<float*>(out_cpu.data),
            Qwen35Config::queries_size());
    }
}

int main() {
    test_gqa<QWEN35_0_8B>(1000, 732, 5);
    test_gqa<QWEN35_0_8B>(1000, 0, 0);
    test_gqa<QWEN35_2B>(100, 2, 23);
    test_gqa<QWEN35_4B>(100, 2, 23);
    test_gqa<QWEN35_9B>(100, 2, 31);
}
