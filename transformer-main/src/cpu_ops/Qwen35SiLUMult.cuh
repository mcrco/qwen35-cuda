#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

class Qwen35SiLUMult {
public:
    static void silu_mult_in_place(input_float_t *x, const input_float_t *y, size_t n);
};
