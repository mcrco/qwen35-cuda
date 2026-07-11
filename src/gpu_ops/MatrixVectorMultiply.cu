#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace matrix_vector_multiply_detail {

constexpr int MATMUL_THREADS = 256;

// Block reduction parallelized across each row.
template <typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t>
__global__ void matrixVectorMultiplyKernel(const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out, int m, int k) {
    extern __shared__ unsigned char shared_mem[];
    auto partials = reinterpret_cast<compute_t *>(shared_mem);
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= m) {
        return;
    }

    compute_t sum = static_cast<compute_t>(0);
    for (int col = tid; col < k; col += blockDim.x) {
        compute_t mval = gpu_ops::read_as<compute_t>(mat[row * k + col]);
        compute_t vval = gpu_ops::read_as<compute_t>(vec[col]);
        sum += mval * vval;
    }

    partials[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        compute_t bval = bias == nullptr ? static_cast<compute_t>(0) : gpu_ops::read_as<compute_t>(bias[row]);
        out[row] = gpu_ops::write_from<out_t>(partials[0] + bval);
    }
}

} // namespace matrix_vector_multiply_detail

template<typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t>
void MatrixVectorMultiply::matmul(int32_t m, int32_t k, const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }

    int threads = matrix_vector_multiply_detail::MATMUL_THREADS;
    int blocks = m;
    size_t shared_bytes = static_cast<size_t>(threads) * sizeof(compute_t);

    matrix_vector_multiply_detail::matrixVectorMultiplyKernel<mat_t, bias_t, vec_t, out_t, compute_t><<<blocks, threads, shared_bytes, stream>>>(mat, bias, vec, out, m, k);
    checkCuda(cudaGetLastError());
}

template<typename vec_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, const __nv_bfloat16 *mat, const __nv_bfloat16 *bias, const vec_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    matmul<__nv_bfloat16, __nv_bfloat16, vec_t, __nv_bfloat16, float>(m, k, mat, bias, vec, out, stream);
}

template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, float, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, float*, cudaStream_t);
template void MatrixVectorMultiply::matmul<float, float, float, float, float>(int32_t, int32_t, const float*, const float*, const float*, float*, cudaStream_t);
template void MatrixVectorMultiply::matmul<int4_t, float, float, float, float>(int32_t, int32_t, const int4_t*, const float*, const float*, float*, cudaStream_t);
template void MatrixVectorMultiply::matmul<int4_t, int4_t, int4_t, int4_t, float>(int32_t, int32_t, const int4_t*, const int4_t*, const int4_t*, int4_t*, cudaStream_t);

template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
