#pragma once

#include "../gpu_ops/GpuFloat.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

/**
 * Parallelization Strategy:
 * Already did this previously.
 * 1. Treat value index pair as long.
 * 2. Find block-level argmax using reduction in shared memory.
 * 3. Atomic argmax over first element in each block to get global argmax.
 */

/**
 * Returns the output index of the maximum value of a CPU array.
 * If there are multiple maximum values, return the one with lower index.
 */
class CpuArgMax {
public:
    template<typename data_t, typename compute_t = float>
    static int32_t argmax_as_float(const data_t *data, size_t n) {
        int32_t best_idx = 0;
        compute_t best_value = -std::numeric_limits<compute_t>::infinity();
        for (size_t i = 0; i < n; i++) {
            compute_t value = gpu_ops::read_as<compute_t>(data[i]);
            if (value > best_value) {
                best_value = value;
                best_idx = static_cast<int32_t>(i);
            }
        }
        return best_idx;
    }

    static int32_t argmax_as_float(const float *data, size_t n);
};
