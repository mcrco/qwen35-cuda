#include "ArgMax.cuh"

int32_t CpuArgMax::argmax_as_float(const float *data, size_t n) {
    return argmax_as_float<float, float>(data, n);
}
