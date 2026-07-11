#pragma once

#include "../CudaBuffer.cuh"
#include "../qwen35/Qwen35Types.cuh"
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
    explicit LayerNorm(int32_t len = 0);

    /**
     * Apply variance correction and scaling factors to hidden state.
     * Storage types may differ from compute_t.
     * Not actually used anymore in Qwen3.5 (switched to zero-centered).
     * @param hidden_state GPU input
     * @param output Location to write output
     * @param n Number of elements
     * @param stream CUDA stream for asycnhronous operation
     */
    template<typename hidden_t, typename weight_t, typename output_t, typename compute_t = float>
    void normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, int32_t n, cudaStream_t stream);

    /**
     * Qwen3.5 zero-centered RMS norm: input * (1 + weight) / rms.
     */
    template<typename hidden_t, typename weight_t, typename output_t, typename compute_t = float>
    void zero_centered_rms_norm(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, int32_t n, float eps, cudaStream_t stream);

    template<typename hidden_t, typename weight_t, typename output_t, typename compute_t = float>
    void zero_centered_rms_norm(const hidden_t *hidden_state, output_t *output, int32_t n, float eps, cudaStream_t stream);

    /**
     * Row-wise gated RMS norm: input * weight * silu(gate) / rms(input row).
     * @param hidden_state GPU input
     * @param gate Gate vector with shape (rows, cols)
     * @param output Location to write output
     * @param rows Number of rows
     * @param cols Number of columns per row
     * @param stream CUDA stream for asycnhronous operation
     */
    template<typename hidden_t, typename gate_t, typename weight_t, typename output_t, typename compute_t = float>
    void normalize_gated_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &gate, const std::shared_ptr<CudaBuffer> &output, int32_t rows, int32_t cols, float eps, cudaStream_t stream);

    template<typename value_t, typename compute_t = float>
    static void l2_norm_rows(const std::shared_ptr<CudaBuffer> &values, int32_t rows, int32_t cols, float scale, float eps, cudaStream_t stream);

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

extern template void LayerNorm::zero_centered_rms_norm<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    float eps,
    cudaStream_t stream);

extern template void LayerNorm::zero_centered_rms_norm<float, float, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t n,
    float eps,
    cudaStream_t stream);

extern template void LayerNorm::zero_centered_rms_norm<float, float, float, float>(
    const float *hidden_state,
    float *output,
    int32_t n,
    float eps,
    cudaStream_t stream);

extern template void LayerNorm::normalize_gated_hidden_state<float, float, float, float, float>(
    const std::shared_ptr<CudaBuffer> &hidden_state,
    const std::shared_ptr<CudaBuffer> &gate,
    const std::shared_ptr<CudaBuffer> &output,
    int32_t rows,
    int32_t cols,
    float eps,
    cudaStream_t stream);

extern template void LayerNorm::l2_norm_rows<float, float>(
    const std::shared_ptr<CudaBuffer> &values,
    int32_t rows,
    int32_t cols,
    float scale,
    float eps,
    cudaStream_t stream);
