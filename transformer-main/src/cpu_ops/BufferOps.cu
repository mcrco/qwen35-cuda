#include "BufferOps.cuh"

void BufferOps::copy(const input_float_t *src, input_float_t *dst, size_t n) {
    copy<input_float_t, input_float_t, float>(src, dst, n);
}

void BufferOps::zero(input_float_t *dst, size_t n) {
    zero<input_float_t, float>(dst, n);
}

void BufferOps::zero_float(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.0f;
    }
}

void BufferOps::add_in_place(input_float_t *residual, const input_float_t *values, size_t n) {
    add_in_place<input_float_t, input_float_t, float>(residual, values, n);
}
