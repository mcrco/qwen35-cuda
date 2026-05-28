#include "BufferOps.cuh"

void BufferOps::copy(const input_float_t *src, input_float_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void BufferOps::zero(input_float_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = input_float_from_float(0.0f);
    }
}

void BufferOps::zero_float(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.0f;
    }
}

void BufferOps::add_in_place(input_float_t *residual, const input_float_t *values, size_t n) {
    for (size_t i = 0; i < n; i++) {
        residual[i] = input_float_from_float(normalize_input_float(residual[i]) + normalize_input_float(values[i]));
    }
}
