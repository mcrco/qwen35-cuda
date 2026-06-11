#pragma once

#include "../qwen35/Qwen35Types.cuh"
#include <cstddef>

/*
 * Parallelization Strategy:
 *
 * Conv1d:
 *
 * Strategy 1:
 * We've done this before. Allocate an output vector, then assign
 * one thread per output value, and have it sum over the rolling window.
 *
 * Strategy 2:
 * If the rolling window is large enough, assign one block per output element(s).
 * Then, assign each thread to element(s) in the rolling window and do a reduction
 * to sum over the rolling window.
 *
 * Delta net update:
 *
 * Strategy 1:
 * Assign one block to each value head. Within each value head, the state matrix
 * has shape (key_head_dim, value_head_dim). Apply the decay coefficient to all
 * state elements in parallel because those updates are independent (vectorized
 * division).
 *
 * Strategy 2:
 * Parallelize the predicted-value and output reductions over value columns.
 * 1. Assign work to each (value_head, value_dim) pair.
 * 2. Compute predicted[v] = sum_k state[h, k, v] * key[h, k] with a reduction
 *    over key_head_dim.
 * 3. Compute delta[v] = (value[h, v] - predicted[v]) * beta[h].
 * 4. Update state[h, k, v] += key[h, k] * delta[v] in parallel over k.
 * 5. Compute weighted_values[h, v] = sum_k state[h, k, v] * query[h, k] with
 *    another reduction over key_head_dim.
 *
 * Strategy 3:
 * Fuse the per-head work to reduce memory traffic. For each value head, keep
 * beta, decay, key, and query values close to the state update. The dependency
 * is per (head, value_dim): predicted must be computed before delta, delta must
 * be known before the state update, and the output reduction must use the
 * updated state. More precisely, notice that the formula for any weighted value is
 *
 * decayed_state[h, k, v] = decay_coeff[h] * old_state[h, k, v]
 * predicted[h, v] = sum_k decayed_state[h, k, v] * keys[h, k]
 * delta[h, v] = (values[h, v] - predicted[h, v]) * beta[h]
 * new_state[h, k, v] = decayed_state[h, k, v] + keys[h, k] * delta[h, v]
 *
 * weighted_values[h, v] = sum_k new_state[h, k, v] * queries[h, k]
 *                       = sum_k (
 *                             decay_coeff[h] * old_state[h, k, v]
 *                             + keys[h, k] * beta[h]
 *                               * (values[h, v]
 *                                  - sum_j decay_coeff[h] * old_state[h, j, v]
 *                                    * keys[h, j])
 *                         ) * queries[h, k]
 *
 * We can assign one block per (h, v) and reduce over k.
 */

class CpuLinearAttention {
public:
    static void conv1d_silu(
        const float *qkv,
        float *conv_state,
        const float *conv_weight,
        const float *conv_bias,
        float *mixed_qkv,
        size_t conv_kernel_dim,
        size_t conv_size);

    static void split_qkv(
        const float *mixed_qkv,
        float *queries,
        float *keys,
        float *values,
        size_t num_key_heads,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim);

    static void gated_delta_update(
        float *state,
        const float *queries,
        const float *keys,
        const float *values,
        const float *beta_raw,
        const float *decay_raw,
        const float *dt_bias,
        const float *A_log,
        float *weighted_values,
        size_t num_value_heads,
        size_t key_head_dim,
        size_t value_head_dim);
};
