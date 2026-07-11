#pragma once

#include "../CudaBuffer.cuh"
#include <memory>
#include <cstdint>
#include <cuda_bf16.h>
#include "../qwen35/Qwen35Types.cuh"

/**
 * Returns the output index of the maximum value of a GPU array.
 * If there are multiple maximum values, return the one with lower index.
 */
class ArgMax {
    std::shared_ptr<CudaBuffer> temp_space;
    std::shared_ptr<CudaBuffer> result_space;
public:
    /**
     * Initialize temporary space
     */
    explicit ArgMax(int32_t len);

    /**
     * Queues the argmax kernel on the stream. Values are converted from data_t
     * to float before comparison so the cross-block aggregation can keep using
     * a 64-bit {float value, int index} atomicCAS pair.
     * @param data GPU values
     * @param n Number of elements
     * @param stream CUDA stream to execute kernels on asynchronously
     * @return pointer in GPU memory to the index. Not valid until cudaStreamSynchronize() is called later.
     */
    template<typename data_t>
    int32_t *argmax_as_float(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);

    int32_t *bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream);
};

extern template int32_t *ArgMax::argmax_as_float<__nv_bfloat16>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
extern template int32_t *ArgMax::argmax_as_float<float>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
extern template int32_t *ArgMax::argmax_as_float<int4_t>(const std::shared_ptr<CudaBuffer> &data_buffer, int32_t n, cudaStream_t stream);
