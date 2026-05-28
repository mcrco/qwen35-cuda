#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"


const int SILU_MULT_THREADS = 128;
const int SILU_MAX_BLOCKS = 1024;

__device__ inline float sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

__global__ void siluKernel(__nv_bfloat16 *x, __nv_bfloat16 *y, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        float xval = __bfloat162float(x[i]);
        float yval = __bfloat162float(y[i]);
        float result = xval * yval * sigmoid(xval);
        x[i] = __float2bfloat16(result);
    }
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    int n = x->size / sizeof(__nv_bfloat16);

    int threads = SILU_MULT_THREADS;
    int blocks = min((n + threads - 1) / threads, SILU_MAX_BLOCKS);

    __nv_bfloat16 *x_ptr = static_cast<__nv_bfloat16*>(x->data);
    __nv_bfloat16 *y_ptr = static_cast<__nv_bfloat16*>(y->data);

    siluKernel<<<blocks, threads, 0, stream>>>(x_ptr, y_ptr, n);
    checkCuda(cudaGetLastError());
}
