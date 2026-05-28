#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

struct int4_t {
    int8_t value{};
};

using input_float_t = int4_t;

inline float normalize_input_float(float x) {
    return x;
}

inline float normalize_input_float(input_float_t x) {
    return static_cast<float>(x.value) / 16.0f;
}

inline input_float_t input_float_from_float(float x) {
    int rounded = static_cast<int>(std::lrintf(x * 16.0f));
    return input_float_t{static_cast<int8_t>(std::clamp(rounded, -8, 7))};
}
