#include "BufferOps.cuh"

void BufferOps::copy(const float *src, float *dst, size_t n) {
    copy<float, float, float>(src, dst, n);
}

void BufferOps::zero(float *dst, size_t n) {
    zero<float, float>(dst, n);
}

void BufferOps::zero_float(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.0f;
    }
}

void BufferOps::add_in_place(float *residual, const float *values, size_t n) {
    add_in_place<float, float, float>(residual, values, n);
}
