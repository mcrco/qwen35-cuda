#include "Qwen35SiLUMult.cuh"

void Qwen35SiLUMult::silu_mult_in_place(input_float_t *x, const input_float_t *y, size_t n) {
    silu_mult_in_place<input_float_t, input_float_t, float>(x, y, n);
}
