#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include "../gpu_ops/GpuFloat.cuh"
#include "Math.cuh"
#include <cstddef>

/**
 * Parallelization strategy:
 * Already did this in previous homework.
 * This is an elementwise fused activation and multiply.
 * 1. Assign one thread to each element i.
 * 2. Each thread reads x[i] and y[i], computes silu(x[i]) * y[i], and writes
 *    the result back to x[i].
 */
class CpuSiLUMult {
public:
    template<typename x_t, typename y_t, typename compute_t = float>
    static void silu_mult_in_place(x_t *x, const y_t *y, size_t n) {
        for (size_t i = 0; i < n; i++) {
            compute_t xv = gpu_ops::read_as<compute_t>(x[i]);
            compute_t yv = gpu_ops::read_as<compute_t>(y[i]);
            x[i] = gpu_ops::write_from<x_t>(static_cast<compute_t>(cpu_silu(static_cast<float>(xv))) * yv);
        }
    }

    static void silu_mult_in_place(float *x, const float *y, size_t n);
};
