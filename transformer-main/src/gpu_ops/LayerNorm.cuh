#pragma once

#include "../CudaBuffer.cuh"
#include <memory>
#include <cuda_bf16.h>

/**
 * Layer normalization without bias, as used in the T5 paper https://arxiv.org/pdf/1910.10683
 * Internally, uses the requested compute type for all calculations, and only rounds/casts at the end.
 */
class LayerNorm {
public:
    /// epsilon to add in denominator square root, for numerical stability
    static constexpr float EPS = 1.0e-6f;

    /// GPU vector of shape (hidden_size,)
    std::shared_ptr<CudaBuffer> weights;

    /**
     * Initialize temporary space
     */
    explicit LayerNorm(int32_t len);

    /**
     * Apply variance correction and scaling factors to hidden state.
     * Storage types may differ from compute_t.
     * @param hidden_state GPU input
     * @param output Location to write output
     * @param n Number of elements
     * @param stream CUDA stream for asycnhronous operation
     */
    template<typename hidden_t, typename weight_t, typename output_t, typename compute_t = float>
    void normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, int32_t n, cudaStream_t stream);

    void normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream);
};

extern template void LayerNorm::normalize_hidden_state<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    cudaStream_t stream);

extern template void LayerNorm::normalize_hidden_state<float, __nv_bfloat16, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    cudaStream_t stream);
