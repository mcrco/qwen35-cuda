#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"
#include "GpuFloat.cuh"

namespace matrix_vector_multiply_detail {

constexpr int MATMUL_THREADS = 128;
constexpr int MATMUL_MAX_BLOCKS = 1024;

template <typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t>
__global__ void matrixVectorMultiplyKernel(const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out, int m, int k) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < m; i += stride) {
        compute_t sum = static_cast<compute_t>(0);
        for (int j = 0; j < k; j++) {
            compute_t mval = gpu_ops::read_as<compute_t>(mat[i * k + j]);
            compute_t vval = gpu_ops::read_as<compute_t>(vec[j]);
            sum += mval * vval;
        }
        compute_t bval = bias == nullptr ? static_cast<compute_t>(0) : gpu_ops::read_as<compute_t>(bias[i]);
        out[i] = gpu_ops::write_from<out_t>(sum + bval);
    }
}

} // namespace matrix_vector_multiply_detail

template<typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t>
void MatrixVectorMultiply::matmul(int32_t m, int32_t k, const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out, cudaStream_t stream) {
    int threads = matrix_vector_multiply_detail::MATMUL_THREADS;
    int blocks = min((m + threads - 1) / threads, matrix_vector_multiply_detail::MATMUL_MAX_BLOCKS);

    matrix_vector_multiply_detail::matrixVectorMultiplyKernel<mat_t, bias_t, vec_t, out_t, compute_t><<<blocks, threads, 0, stream>>>(mat, bias, vec, out, m, k);
    checkCuda(cudaGetLastError());
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, const __nv_bfloat16 *mat, const __nv_bfloat16 *bias, const input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    matmul<__nv_bfloat16, __nv_bfloat16, input_float_t, __nv_bfloat16, float>(m, k, mat, bias, vec, out, stream);
}

template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, float, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, float*, cudaStream_t);
template void MatrixVectorMultiply::matmul<int4_t, float, float, float, float>(int32_t, int32_t, const int4_t*, const float*, const float*, float*, cudaStream_t);

template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
