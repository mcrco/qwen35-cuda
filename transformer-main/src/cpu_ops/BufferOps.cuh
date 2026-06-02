#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include "../gpu_ops/GpuFloat.cuh"
#include <cstddef>

class BufferOps {
public:
    template<typename src_t, typename dst_t, typename compute_t = float>
    static void copy(const src_t *src, dst_t *dst, size_t n) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = gpu_ops::write_from<dst_t>(gpu_ops::read_as<compute_t>(src[i]));
        }
    }

    template<typename dst_t, typename compute_t = float>
    static void zero(dst_t *dst, size_t n) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = gpu_ops::write_from<dst_t>(static_cast<compute_t>(0));
        }
    }

    template<typename residual_t, typename value_t, typename compute_t = float>
    static void add_in_place(residual_t *residual, const value_t *values, size_t n) {
        for (size_t i = 0; i < n; i++) {
            compute_t sum = gpu_ops::read_as<compute_t>(residual[i]) + gpu_ops::read_as<compute_t>(values[i]);
            residual[i] = gpu_ops::write_from<residual_t>(sum);
        }
    }

    static void copy(const input_float_t *src, input_float_t *dst, size_t n);
    static void zero(input_float_t *dst, size_t n);
    static void zero_float(float *dst, size_t n);
    static void add_in_place(input_float_t *residual, const input_float_t *values, size_t n);
};
