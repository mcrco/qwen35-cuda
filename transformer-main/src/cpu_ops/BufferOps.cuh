#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class BufferOps {
public:
    static void copy(const input_float_t *src, input_float_t *dst, size_t n);
    static void zero(input_float_t *dst, size_t n);
    static void zero_float(float *dst, size_t n);
    static void add_in_place(input_float_t *residual, const input_float_t *values, size_t n);
};
