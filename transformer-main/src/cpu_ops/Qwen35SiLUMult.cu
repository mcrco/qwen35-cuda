#include "Qwen35SiLUMult.cuh"
#include "Qwen35Math.cuh"

void Qwen35SiLUMult::silu_mult_in_place(input_float_t *x, const input_float_t *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = input_float_from_float(qwen35_silu(normalize_input_float(x[i])) * normalize_input_float(y[i]));
    }
}
