#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

/**
 * Parallelization strategy 1:
 * Already implemented this in homework 6.
 * Assign one thread to each of the query and key-value head pairs.
 * Couldn't think of anything better at the time, but there are better
 * alternatives.
 *
 * Parallelization strategy 2:
 * Use online softmax block reduction over chunks (range over t) of qkv vectors.
 * 1. Assign blocks to each query head and sequence chunk.
 * 2. Each block computes complete dot-product scores for its token chunk:
 *    score[t] = dot(query[qh], key[t, kvh]) / sqrt(head_size).
 * 3. Within each block, keep the unnormalized online-softmax state:
 *    local_max, local_sum, and local_weighted_values[head_size].
 * 4. Merge block states for the same query head pairwise using the online
 *    softmax merge rule. Repeat as a tree reduction over 2, 4, 8, ... blocks.
 * 5. Normalize the final weighted-values numerator by the final denominator,
 *    then apply the output gate.
 *
 * Parallelization strategy 3:
 * Flash attention.
 */

class CpuGroupQueryAttention {
public:
    static void sdpa(
        float *queries,
        float *keys_cache,
        float *values_cache,
        float *weighted_values,
        float *gate,
        size_t layer_num,
        size_t seq_len,
        size_t num_layers,
        size_t num_query_heads,
        size_t num_kv_heads,
        size_t head_size,
        size_t keys_size,
        size_t values_size);
};
