#include "Qwen35MatrixVectorMultiply.cuh"

void Qwen35MatrixVectorMultiply::matmul(size_t m, size_t k, const input_float_t *mat, const input_float_t *bias, const input_float_t *vec, input_float_t *out) {
    matmul<input_float_t, input_float_t, input_float_t, input_float_t, float>(m, k, mat, bias, vec, out);
}
