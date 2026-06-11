#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>
#include <cstdint>
#include <random>

class Qwen35Sampling {
public:
    static int32_t sample(const float *scores, size_t vocab_size, float temperature, std::mt19937 &rng);
};
