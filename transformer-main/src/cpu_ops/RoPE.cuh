#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

/**
 * Parallelization strategy:
 * Already did this in previous homework.
 * RoPE rotates independent pairs inside each query/key head.
 * 1. Assign work over (head, rotary_pair) for both query and key tensors.
 * 2. Each worker computes the pair's angle, sin, and cos, then writes the two
 *    rotated elements back in place.
 * 3. Elements outside rotary_dim are unchanged and need no work.
 * 4. If trig cost dominates, precompute or cache sin/cos values for each
 *    (position, rotary_pair) and reuse them across heads.
 */
class CpuRoPE {
public:
    static void apply_partial_rope_to_qk(
        float *queries,
        size_t num_query_heads,
        float *keys,
        size_t num_kv_heads,
        size_t head_size,
        size_t rotary_dim,
        size_t position_idx,
        float theta_base);
};
