#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35MatrixVectorMultiply {
public:
    static void matmul(size_t m, size_t k, const input_float_t *mat, const input_float_t *bias, const input_float_t *vec, input_float_t *out);
};
