#include "Qwen35MatrixVectorMultiply.cuh"

void Qwen35MatrixVectorMultiply::matmul(size_t m, size_t k, const float *mat, const float *bias, const float *vec, float *out) {
    matmul<float, float, float, float, float>(m, k, mat, bias, vec, out);
}
