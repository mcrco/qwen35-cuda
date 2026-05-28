#pragma once

#include <cmath>

inline float qwen35_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float qwen35_silu(float x) {
    return x * qwen35_sigmoid(x);
}

inline float qwen35_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}
