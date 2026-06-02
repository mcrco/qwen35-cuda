#pragma once

#include <cuda_bf16.h>
#include <cmath>
#include "../qwen35/Qwen35Types.cuh"

namespace gpu_ops {

template<typename compute_t>
__host__ __device__ inline compute_t clamp_compute(compute_t x, compute_t lo, compute_t hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

template<typename compute_t, typename storage_t>
__host__ __device__ inline compute_t read_as(storage_t x) {
    return static_cast<compute_t>(x);
}

template<typename compute_t>
__host__ __device__ inline compute_t read_as(__nv_bfloat16 x) {
    return static_cast<compute_t>(__bfloat162float(x));
}

template<typename compute_t>
__host__ __device__ inline compute_t read_as(int4_t x) {
    return static_cast<compute_t>(x.value) / static_cast<compute_t>(16.0f);
}

template<typename storage_t>
struct FloatWriter {
    template<typename compute_t>
    __host__ __device__ static storage_t write(compute_t x) {
        return static_cast<storage_t>(x);
    }
};

template<>
struct FloatWriter<__nv_bfloat16> {
    template<typename compute_t>
    __host__ __device__ static __nv_bfloat16 write(compute_t x) {
        return __float2bfloat16(static_cast<float>(x));
    }
};

template<>
struct FloatWriter<int4_t> {
    template<typename compute_t>
    __host__ __device__ static int4_t write(compute_t x) {
        compute_t scaled = clamp_compute(x * static_cast<compute_t>(16.0f), static_cast<compute_t>(-8.0f), static_cast<compute_t>(7.0f));
        return int4_t{static_cast<int8_t>(llrint(static_cast<double>(scaled)))};
    }
};

template<typename storage_t, typename compute_t>
__host__ __device__ inline storage_t write_from(compute_t x) {
    return FloatWriter<storage_t>::write(x);
}

} // namespace gpu_ops
