#include "Qwen35Sampling.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

int32_t Qwen35Sampling::sample(const input_float_t *scores, size_t vocab_size, float temperature, std::mt19937 &rng) {
    int32_t best_idx = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    if (temperature == 0.0f) {
        for (size_t tok = 0; tok < vocab_size; tok++) {
            float score = normalize_input_float(scores[tok]);
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int32_t>(tok);
            }
        }
        return best_idx;
    }

    std::vector<float> probs(vocab_size);
    for (size_t tok = 0; tok < vocab_size; tok++) {
        probs[tok] = normalize_input_float(scores[tok]) / temperature;
        best_score = std::max(best_score, probs[tok]);
    }
    float total = 0.0f;
    for (float &p : probs) {
        p = std::exp(p - best_score);
        total += p;
    }
    std::uniform_real_distribution<float> dist(0.0f, total);
    float sample = dist(rng);
    float cdf = 0.0f;
    for (size_t tok = 0; tok < vocab_size; tok++) {
        cdf += probs[tok];
        if (sample <= cdf) {
            return static_cast<int32_t>(tok);
        }
    }
    return static_cast<int32_t>(vocab_size - 1);
}
