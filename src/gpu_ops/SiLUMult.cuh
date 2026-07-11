#pragma once

#include <memory>
#include <cstdint>
#include "../CudaBuffer.cuh"
#include "../qwen35/Qwen35Types.cuh"
#include <cuda_bf16.h>

class SiLUMult {
public:
    /**
     * Fused sigmoid linear unit and element-wise multiplication. Part of Swish Gate Linear Unit (SwiGLU), see https://arxiv.org/pdf/2002.05202.
     * Writes result in-place over x, such that:
     * x = x / (1 + exp(-x)) * y
     * @param x vector, calculated with gate_proj(ffn_input)
     * @param y vector, calculated with up_proj(ffn_input)
     * @param n Number of elements
     * @param stream CUDA stream for asynchronous operation
     */
    template<typename x_t, typename y_t, typename compute_t = float>
    static void silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, int32_t n, cudaStream_t stream);

    static void silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream);
};

extern template void SiLUMult::silu_mult_in_place<__nv_bfloat16, __nv_bfloat16, float>(const std::shared_ptr<CudaBuffer>&, const std::shared_ptr<CudaBuffer>&, int32_t, cudaStream_t);
extern template void SiLUMult::silu_mult_in_place<float, float, float>(const std::shared_ptr<CudaBuffer>&, const std::shared_ptr<CudaBuffer>&, int32_t, cudaStream_t);
extern template void SiLUMult::silu_mult_in_place<int4_t, int4_t, float>(const std::shared_ptr<CudaBuffer>&, const std::shared_ptr<CudaBuffer>&, int32_t, cudaStream_t);
