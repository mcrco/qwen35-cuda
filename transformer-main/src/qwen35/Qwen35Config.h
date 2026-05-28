#pragma once

#include <cstddef>
#include <stdexcept>

enum Qwen35Size {
    QWEN35_0_8B,
    QWEN35_4B,
    QWEN35_9B,
};

template<Qwen35Size size>
struct Qwen35Config {
    static constexpr size_t hidden_size() {
        if constexpr (size == QWEN35_0_8B) return 1024;
        if constexpr (size == QWEN35_4B) return 2560;
        if constexpr (size == QWEN35_9B) return 4096;
        throw std::logic_error("Unknown Qwen3.5 size");
    }

    static constexpr size_t num_layers() {
        if constexpr (size == QWEN35_0_8B) return 24;
        if constexpr (size == QWEN35_4B || size == QWEN35_9B) return 32;
        throw std::logic_error("Unknown Qwen3.5 size");
    }

    static constexpr size_t intermediate_size() {
        if constexpr (size == QWEN35_0_8B) return 3584;
        if constexpr (size == QWEN35_4B) return 9216;
        if constexpr (size == QWEN35_9B) return 12288;
        throw std::logic_error("Unknown Qwen3.5 size");
    }

    static constexpr bool embedding_tying() {
        if constexpr (size == QWEN35_0_8B || size == QWEN35_4B) return true;
        if constexpr (size == QWEN35_9B) return false;
        throw std::logic_error("Unknown Qwen3.5 size");
    }

    static constexpr size_t vocab_size() {
        return 248320;
    }

    static constexpr size_t num_query_heads() {
        if constexpr (size == QWEN35_0_8B) return 8;
        return 16;
    }

    static constexpr size_t num_kv_heads() {
        if constexpr (size == QWEN35_0_8B) return 2;
        return 4;
    }

    static constexpr size_t head_size() {
        return 256;
    }

    static constexpr float rope_theta_base() {
        return 10000000.0f;
    }

    static constexpr size_t rotary_dim() {
        return 64;
    }

    static constexpr size_t linear_num_key_heads() {
        return 16;
    }

    static constexpr size_t linear_num_value_heads() {
        if constexpr (size == QWEN35_0_8B) return 16;
        return 32;
    }

    static constexpr size_t linear_key_head_dim() {
        return 128;
    }

    static constexpr size_t linear_value_head_dim() {
        return 128;
    }

    static constexpr size_t linear_conv_kernel_dim() {
        return 4;
    }

    static constexpr float rms_norm_eps() {
        return 1e-6f;
    }

    static constexpr bool full_attention_layer(size_t layer_idx) {
        return (layer_idx + 1) % 4 == 0;
    }

    static constexpr size_t queries_size() {
        return num_query_heads() * head_size();
    }

    static constexpr size_t keys_size() {
        return num_kv_heads() * head_size();
    }

    static constexpr size_t values_size() {
        return num_kv_heads() * head_size();
    }

    static constexpr size_t linear_keys_size() {
        return linear_num_key_heads() * linear_key_head_dim();
    }

    static constexpr size_t linear_values_size() {
        return linear_num_value_heads() * linear_value_head_dim();
    }

    static constexpr size_t linear_conv_size() {
        return 2 * linear_keys_size() + linear_values_size();
    }
};
