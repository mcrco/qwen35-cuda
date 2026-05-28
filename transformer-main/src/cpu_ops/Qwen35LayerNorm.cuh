#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35LayerNorm {
public:
    static void zero_centered_rms_norm(const input_float_t *weight, const input_float_t *input, input_float_t *output, size_t n, float eps);
    static void gated_rms_norm(const input_float_t *weight, const float *input, const input_float_t *gate, input_float_t *output, size_t heads, size_t dim, float eps);
    static void l2norm_rows(float *values, size_t rows, size_t cols, float scale, float eps);
};
