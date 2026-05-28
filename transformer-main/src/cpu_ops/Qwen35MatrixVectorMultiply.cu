#include "Qwen35MatrixVectorMultiply.cuh"

void Qwen35MatrixVectorMultiply::matmul(size_t m, size_t k, const input_float_t *mat, const input_float_t *bias, const input_float_t *vec, input_float_t *out) {
    for (size_t row = 0; row < m; row++) {
        float sum = bias ? normalize_input_float(bias[row]) : 0.0f;
        for (size_t col = 0; col < k; col++) {
            sum += normalize_input_float(mat[row * k + col]) * normalize_input_float(vec[col]);
        }
        out[row] = input_float_from_float(sum);
    }
}
