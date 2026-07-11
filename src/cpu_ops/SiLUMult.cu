#include "SiLUMult.cuh"

void CpuSiLUMult::silu_mult_in_place(float *x, const float *y, size_t n) {
    silu_mult_in_place<float, float, float>(x, y, n);
}
