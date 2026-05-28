#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

const int MATMUL_THREADS = 128;
const int MATMUL_MAX_BLOCKS = 1024;

__device__ inline float normalize_float(float x) {
    return x;
}

__device__ inline float normalize_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename input_float_t>
__global__ void matrixVectorMultiplyKernel(__nv_bfloat16 *mat, __nv_bfloat16 *bias, input_float_t *vec, __nv_bfloat16 *out, int m, int k) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < m; i += stride) {
        float sum = 0;
        for (int j = 0; j < k; j++) {
            float mval = normalize_float(mat[i * k + j]);
            float vval = normalize_float(vec[j]);
            sum += mval * vval;
        }
        float bval = bias == nullptr ? 0.0f : normalize_float(bias[i]);
        out[i] = __float2bfloat16(sum + bval);
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    int threads = MATMUL_THREADS;
    int blocks = min((m + threads - 1) / threads, MATMUL_MAX_BLOCKS);

    matrixVectorMultiplyKernel<input_float_t><<<blocks, threads, 0, stream>>>(mat, bias, vec, out, m, k);
    checkCuda(cudaGetLastError());
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
