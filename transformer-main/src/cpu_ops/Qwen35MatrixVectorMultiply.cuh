#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include "../gpu_ops/GpuFloat.cuh"
#include <cstddef>

class Qwen35MatrixVectorMultiply {
public:
    template<typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t = float>
    static void matmul(size_t m, size_t k, const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out) {
        for (size_t row = 0; row < m; row++) {
            compute_t sum = bias ? gpu_ops::read_as<compute_t>(bias[row]) : static_cast<compute_t>(0);
            for (size_t col = 0; col < k; col++) {
                sum += gpu_ops::read_as<compute_t>(mat[row * k + col]) * gpu_ops::read_as<compute_t>(vec[col]);
            }
            out[row] = gpu_ops::write_from<out_t>(sum);
        }
    }

    static void matmul(size_t m, size_t k, const float *mat, const float *bias, const float *vec, float *out);
};
