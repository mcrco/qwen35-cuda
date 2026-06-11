#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include "../qwen35/Qwen35Types.cuh"

class MatrixVectorMultiply {
public:
    /**
     * Matrix vector multiplication, with optional bias vector to add to the result.
     * Storage types may differ from compute_t; all inputs are converted to compute_t
     * before arithmetic and output is rounded/cast to out_t at the end.
     * @param m Rows in matrix/number of elements in output vector
     * @param k Columns in matrix/number of elements in input vector
     * @param mat Row-major matrix
     * @param bias Bias vector, null for 0 bias
     * @param vec Input vector
     * @param out Output vector
     * @param stream CUDA stream
     */
    template<typename mat_t, typename bias_t, typename vec_t, typename out_t, typename compute_t = float>
    static void matmul(int32_t m, int32_t k, const mat_t *mat, const bias_t *bias, const vec_t *vec, out_t *out, cudaStream_t stream);

    template<typename mat_t, typename vec_t, typename out_t, typename compute_t = float>
    static void matmul_no_bias(int32_t m, int32_t k, const mat_t *mat, const vec_t *vec, out_t *out, cudaStream_t stream) {
        matmul<mat_t, mat_t, vec_t, out_t, compute_t>(m, k, mat, nullptr, vec, out, stream);
    }

    template<typename vec_t>
    static void bf16_matmul(int32_t m, int32_t k, const __nv_bfloat16 *mat, const __nv_bfloat16 *bias, const vec_t *vec, __nv_bfloat16 *out, cudaStream_t stream);
};

extern template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
extern template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
extern template void MatrixVectorMultiply::matmul<__nv_bfloat16, __nv_bfloat16, float, float, float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, float*, cudaStream_t);
extern template void MatrixVectorMultiply::matmul<float, float, float, float, float>(int32_t, int32_t, const float*, const float*, const float*, float*, cudaStream_t);
extern template void MatrixVectorMultiply::matmul<int4_t, float, float, float, float>(int32_t, int32_t, const int4_t*, const float*, const float*, float*, cudaStream_t);
extern template void MatrixVectorMultiply::matmul<int4_t, int4_t, int4_t, int4_t, float>(int32_t, int32_t, const int4_t*, const int4_t*, const int4_t*, int4_t*, cudaStream_t);

extern template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
extern template void MatrixVectorMultiply::bf16_matmul<float>(int32_t, int32_t, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, cudaStream_t);
