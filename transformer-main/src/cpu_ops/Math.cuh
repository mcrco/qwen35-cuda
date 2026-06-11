#pragma once

#include <cmath>

inline float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float cpu_silu(float x) {
    return x * cpu_sigmoid(x);
}

inline float cpu_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}
