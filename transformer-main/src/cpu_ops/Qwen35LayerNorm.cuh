#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include "../gpu_ops/GpuFloat.cuh"
#include "Qwen35Math.cuh"
#include <cstddef>
#include <cmath>

class Qwen35LayerNorm {
public:
    template<typename weight_t, typename input_t, typename output_t, typename compute_t = float>
    static void zero_centered_rms_norm(const weight_t *weight, const input_t *input, output_t *output, size_t n, float eps) {
        compute_t sum_squares = static_cast<compute_t>(0);
        for (size_t i = 0; i < n; i++) {
            compute_t v = gpu_ops::read_as<compute_t>(input[i]);
            sum_squares += v * v;
        }
        compute_t inv_rms = static_cast<compute_t>(1) / static_cast<compute_t>(std::sqrt(static_cast<double>(sum_squares) / static_cast<double>(n) + eps));
        for (size_t i = 0; i < n; i++) {
            compute_t v = gpu_ops::read_as<compute_t>(input[i]) * inv_rms * (static_cast<compute_t>(1) + gpu_ops::read_as<compute_t>(weight[i]));
            output[i] = gpu_ops::write_from<output_t>(v);
        }
    }

    template<typename weight_t, typename input_t, typename gate_t, typename output_t, typename compute_t = float>
    static void gated_rms_norm(const weight_t *weight, const input_t *input, const gate_t *gate, output_t *output, size_t heads, size_t dim, float eps) {
        for (size_t h = 0; h < heads; h++) {
            const input_t *in_row = input + h * dim;
            output_t *out_row = output + h * dim;
            const gate_t *gate_row = gate + h * dim;
            compute_t sum_squares = static_cast<compute_t>(0);
            for (size_t d = 0; d < dim; d++) {
                compute_t v = gpu_ops::read_as<compute_t>(in_row[d]);
                sum_squares += v * v;
            }
            compute_t inv_rms = static_cast<compute_t>(1) / static_cast<compute_t>(std::sqrt(static_cast<double>(sum_squares) / static_cast<double>(dim) + eps));
            for (size_t d = 0; d < dim; d++) {
                compute_t v = gpu_ops::read_as<compute_t>(in_row[d]) * inv_rms * gpu_ops::read_as<compute_t>(weight[d]) *
                    static_cast<compute_t>(qwen35_silu(static_cast<float>(gpu_ops::read_as<compute_t>(gate_row[d]))));
                out_row[d] = gpu_ops::write_from<output_t>(v);
            }
        }
    }

    static void zero_centered_rms_norm(const input_float_t *weight, const input_float_t *input, input_float_t *output, size_t n, float eps);
    static void gated_rms_norm(const input_float_t *weight, const float *input, const input_float_t *gate, input_float_t *output, size_t heads, size_t dim, float eps);
    static void l2norm_rows(float *values, size_t rows, size_t cols, float scale, float eps);
};
