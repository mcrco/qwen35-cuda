#pragma once

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

struct int4_t {
    int8_t value{};
};
