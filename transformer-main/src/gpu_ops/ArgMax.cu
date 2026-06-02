#include "ArgMax.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

#include <cfloat>

namespace argmax_detail {

constexpr int ARGMAX_THREADS = 128;
constexpr int ARGMAX_MAX_BLOCKS = 1024;

struct ValueIndexPair {
    float val;
    int idx;
};

__device__ inline ValueIndexPair pair_argmax(ValueIndexPair a, ValueIndexPair b) {
    if (a.val == b.val) {
        if (a.idx < b.idx) {
            return a;
        }
        return b;
    }
    if (a.val > b.val) {
        return a;
    }
    return b;
}

__device__ inline void atomicArgMax(ValueIndexPair *pair, ValueIndexPair other) {
    unsigned long long* address = reinterpret_cast<unsigned long long*>(pair);
    unsigned long long old = *address;
    unsigned long long assumed;
    do {
        assumed = old;
        ValueIndexPair best = pair_argmax(other, *reinterpret_cast<ValueIndexPair*>(&assumed));
        old = ::atomicCAS(address, assumed, *reinterpret_cast<unsigned long long*>(&best));
    } while (assumed != old);
}

template<typename data_t>
__global__ void argmaxKernel(const data_t *data, ValueIndexPair *result, int n) {
    extern __shared__ ValueIndexPair partial_argmaxes[];

    int tidx = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    ValueIndexPair local_argmax = {-FLT_MAX, -1};
    for (int i = idx; i < n; i += stride) {
        ValueIndexPair input = {gpu_ops::read_as<float>(data[i]), i};
        local_argmax = pair_argmax(local_argmax, input);
    }
    partial_argmaxes[tidx] = local_argmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            partial_argmaxes[tidx] = pair_argmax(partial_argmaxes[tidx], partial_argmaxes[tidx + s]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        atomicArgMax(result, partial_argmaxes[tidx]);
    }
}

} // namespace argmax_detail

ArgMax::ArgMax(int32_t len) {
    temp_space = std::make_shared<CudaBuffer>(sizeof(argmax_detail::ValueIndexPair));
}

template<typename data_t>
int32_t *ArgMax::argmax_as_float(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream) {
    const data_t *data = static_cast<const data_t*>(data_buffer->data);
    auto* result = static_cast<argmax_detail::ValueIndexPair*>(temp_space->data);
    result->val = -FLT_MAX;
    result->idx = -1;

    int threads = argmax_detail::ARGMAX_THREADS;
    int blocks = min((n + threads - 1) / threads, argmax_detail::ARGMAX_MAX_BLOCKS);
    int shared_mem_size = threads * sizeof(argmax_detail::ValueIndexPair);
    argmax_detail::argmaxKernel<data_t><<<blocks, threads, shared_mem_size, stream>>>(data, result, n);
    checkCuda(cudaGetLastError());

    return &result->idx;
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    int n = bf16_data->size / sizeof(__nv_bfloat16);
    return argmax_as_float<__nv_bfloat16>(bf16_data, n, stream);
}

template int32_t *ArgMax::argmax_as_float<__nv_bfloat16>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
template int32_t *ArgMax::argmax_as_float<float>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
template int32_t *ArgMax::argmax_as_float<int4_t>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
