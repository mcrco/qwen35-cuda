#pragma once

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

struct int4_t {
    int8_t value{};
};

using input_float_t = float;

inline float normalize_input_float(float x) {
    return x;
}

inline float normalize_input_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

inline input_float_t input_float_from_float(float x) {
    return x;
}
