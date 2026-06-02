#include "Qwen35LayerNorm.cuh"

void Qwen35LayerNorm::zero_centered_rms_norm(const input_float_t *weight, const input_float_t *input, input_float_t *output, size_t n, float eps) {
    zero_centered_rms_norm<input_float_t, input_float_t, input_float_t, float>(weight, input, output, n, eps);
}

void Qwen35LayerNorm::gated_rms_norm(const input_float_t *weight, const float *input, const input_float_t *gate, input_float_t *output, size_t heads, size_t dim, float eps) {
    gated_rms_norm<input_float_t, float, input_float_t, input_float_t, float>(weight, input, gate, output, heads, dim, eps);
}

void Qwen35LayerNorm::l2norm_rows(float *values, size_t rows, size_t cols, float scale, float eps) {
    for (size_t r = 0; r < rows; r++) {
        float sum_squares = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float v = values[r * cols + c];
            sum_squares += v * v;
        }
        float coeff = scale / std::sqrt(sum_squares + eps);
        for (size_t c = 0; c < cols; c++) {
            values[r * cols + c] *= coeff;
        }
    }
}
