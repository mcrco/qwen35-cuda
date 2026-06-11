#include "../ErrorCheck.h"
#include "../gpu_ops/Sampling.cuh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

int32_t reference_sample(const float *scores, int32_t n, float temperature, float uniform01) {
    float best_score = -std::numeric_limits<float>::infinity();
    std::vector<float> probs(n);

    for (int32_t i = 0; i < n; i++) {
        probs[i] = scores[i] / temperature;
        best_score = std::max(best_score, probs[i]);
    }

    float total = 0.0f;
    for (float &p : probs) {
        p = std::exp(p - best_score);
        total += p;
    }

    float sample = uniform01 * total;
    float cdf = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        cdf += probs[i];
        if (sample <= cdf) {
            return i;
        }
    }
    return n - 1;
}

void test_sampling(int32_t num_els, float temperature, float uniform01) {
    auto buf = std::make_shared<CudaBuffer>(num_els * sizeof(float));
    auto *scores = static_cast<float *>(buf->data);

    std::mt19937 generator{42};
    std::normal_distribution<float> distribution(0.0f, 8.0f);
    for (int32_t i = 0; i < num_els; i++) {
        scores[i] = distribution(generator);
    }

    int32_t expected = reference_sample(scores, num_els, temperature, uniform01);

    Sampling sampling(num_els);
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream));
    int32_t *calculated_ptr = sampling.sample(buf, num_els, temperature, uniform01, stream);
    checkCuda(cudaStreamSynchronize(stream));
    int32_t calculated = *calculated_ptr;
    checkCuda(cudaStreamDestroy(stream));

    if (calculated != expected) {
        std::cerr << "got index " << calculated << ", expected index " << expected
                  << " for n=" << num_els << ", temperature=" << temperature
                  << ", uniform01=" << uniform01 << std::endl;
        std::exit(1);
    }
}

int main() {
    test_sampling(1, 0.7f, 0.0f);
    test_sampling(1234, 0.7f, 0.2f);
    test_sampling(1234, 1.3f, 0.8f);
    test_sampling(151936, 0.8f, 0.5f);
}
