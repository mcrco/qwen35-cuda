#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>
#include <cstdint>
#include <random>

/**
 * Parallelization strategy 1:
 * For temperature == 0, sampling is argmax. Split the vocabulary into chunks,
 * compute a local best (score, token_id) per chunk, then reduce the local best
 * pairs. Ties should keep the lower token id.
 *
 * Parallelization strategy 2:
 * For temperature > 0, parallelize softmax sampling over vocabulary chunks.
 * 1. Split the scores into chunks and compute a local max score per chunk.
 * 2. Reduce local max values to get the global max score.
 * 3. In parallel, compute exp(score[t] / temperature - global_max) for each
 *    token and a local sum per chunk.
 * 4. Reduce local sums to get the total softmax denominator.
 * 5. Draw a uniform sample in [0, total_sum), then use a parallel prefix sum
 *    over chunk sums to find the selected chunk.
 * 6. Scan or prefix-sum within the selected chunk to find the first token where
 *    the cumulative probability reaches the sample.
 *
 * Parallelization strategy 3:
 * Fuse exponentiation and local prefix generation when memory bandwidth matters.
 * Each chunk keeps its local probabilities/prefix data, while global reductions
 * only exchange compact per-chunk max, sum, and prefix metadata.
 */
class CpuSampling {
public:
    static int32_t sample(const float *scores, size_t vocab_size, float temperature, std::mt19937 &rng);
};
