#include "Qwen35LayerNorm.cuh"
#include "Qwen35Math.cuh"

#include <cmath>

void Qwen35LayerNorm::zero_centered_rms_norm(const input_float_t *weight, const input_float_t *input, input_float_t *output, size_t n, float eps) {
    float sum_squares = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = normalize_input_float(input[i]);
        sum_squares += v * v;
    }
    float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; i++) {
        output[i] = input_float_from_float(normalize_input_float(input[i]) * inv_rms * (1.0f + normalize_input_float(weight[i])));
    }
}

void Qwen35LayerNorm::gated_rms_norm(const input_float_t *weight, const float *input, const input_float_t *gate, input_float_t *output, size_t heads, size_t dim, float eps) {
    for (size_t h = 0; h < heads; h++) {
        const float *in_row = input + h * dim;
        input_float_t *out_row = output + h * dim;
        const input_float_t *gate_row = gate + h * dim;
        float sum_squares = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            sum_squares += in_row[d] * in_row[d];
        }
        float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(dim) + eps);
        for (size_t d = 0; d < dim; d++) {
            float v = in_row[d] * inv_rms * normalize_input_float(weight[d]) * qwen35_silu(normalize_input_float(gate_row[d]));
            out_row[d] = input_float_from_float(v);
        }
    }
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
