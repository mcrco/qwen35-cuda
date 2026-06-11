#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../cpu_ops/ArgMax.cuh"
#include "../cpu_ops/GroupQueryAttention.cuh"
#include "../cpu_ops/LayerNorm.cuh"
#include "../cpu_ops/MatrixVectorMultiply.cuh"
#include "../cpu_ops/RoPE.cuh"
#include "../cpu_ops/Sampling.cuh"
#include "../cpu_ops/SiLUMult.cuh"
#include "../gpu_ops/ArgMax.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/Sampling.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../qwen35/Qwen35Config.h"
#include "../vendor/argparse.hpp"
#include "../vendor/json.hpp"

#ifndef TRANSFORMER_GIT_COMMIT
#define TRANSFORMER_GIT_COMMIT "unknown"
#endif

namespace {

struct BenchArgs {
    std::string module;
    std::string preset;
    std::string dtype;
    std::string json_path;
    std::string label;
    int32_t seed{};
    int32_t warmup_iters{};
    int32_t gpu_iters{};
    int32_t cpu_iters{};
    int32_t m{};
    int32_t k{};
    int32_t n{};
    int32_t rows{};
    int32_t cols{};
    int32_t seq_len{};
    int32_t layer_num{};
    int32_t vocab_size{};
    float temperature{};
};

struct Shape {
    int32_t hidden_size{};
    int32_t intermediate_size{};
    int32_t num_layers{};
    int32_t num_query_heads{};
    int32_t num_kv_heads{};
    int32_t head_size{};
    int32_t rotary_dim{};
    int32_t queries_size{};
    int32_t keys_size{};
    int32_t values_size{};
    float rope_theta_base{};
    float rms_norm_eps{};
};

struct BenchOutput {
    std::string module;
    nlohmann::json shapes;
    double cpu_total_us{};
    double gpu_total_us{};
};

std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);

    std::ostringstream out;
    out << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S%z");
    return out.str();
}

template<Qwen35Size QWEN35_SIZE>
Shape shape_for() {
    using Config = Qwen35Config<QWEN35_SIZE>;
    return Shape{
        .hidden_size = static_cast<int32_t>(Config::hidden_size()),
        .intermediate_size = static_cast<int32_t>(Config::intermediate_size()),
        .num_layers = static_cast<int32_t>(Config::num_layers()),
        .num_query_heads = static_cast<int32_t>(Config::num_query_heads()),
        .num_kv_heads = static_cast<int32_t>(Config::num_kv_heads()),
        .head_size = static_cast<int32_t>(Config::head_size()),
        .rotary_dim = static_cast<int32_t>(Config::rotary_dim()),
        .queries_size = static_cast<int32_t>(Config::queries_size()),
        .keys_size = static_cast<int32_t>(Config::keys_size()),
        .values_size = static_cast<int32_t>(Config::values_size()),
        .rope_theta_base = Config::rope_theta_base(),
        .rms_norm_eps = Config::rms_norm_eps(),
    };
}

Shape shape_for_preset(const std::string &preset) {
    if (preset == "qwen35-0.8b") return shape_for<QWEN35_0_8B>();
    if (preset == "qwen35-4b") return shape_for<QWEN35_4B>();
    if (preset == "qwen35-9b") return shape_for<QWEN35_9B>();
    throw std::runtime_error("unsupported --preset value: " + preset);
}

template<typename fn_t>
double time_cpu(int32_t iters, fn_t fn) {
    if (iters <= 0) return 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < iters; i++) {
        fn();
    }
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

template<typename fn_t>
double time_gpu(int32_t warmup_iters, int32_t iters, const char *range_name, fn_t fn) {
    if (iters <= 0) return 0.0;
    nvtxRangePush("module_bench_warmup");
    for (int32_t i = 0; i < warmup_iters; i++) {
        fn();
    }
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
    nvtxRangePop();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start, cudaStreamPerThread));
    nvtxRangePush("module_bench_measure_gpu");
    for (int32_t i = 0; i < iters; i++) {
        nvtxRangePush(range_name);
        fn();
        nvtxRangePop();
    }
    nvtxRangePop();
    checkCuda(cudaEventRecord(stop, cudaStreamPerThread));
    checkCuda(cudaEventSynchronize(stop));

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return static_cast<double>(ms) * 1000.0;
}

void fill_random(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float &value : values) {
        value = dist(rng);
    }
}

void copy_to_cuda(const std::vector<float> &src, CudaBuffer &dst) {
    std::copy(src.begin(), src.end(), static_cast<float *>(dst.data));
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    // CudaBuffer uses managed memory; prefetch after CPU writes so timed kernels do not include page migration.
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    checkCuda(cudaMemPrefetchAsync(dst.data, dst.size, location, 0, cudaStreamPerThread));
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
}

int32_t cpu_sample_with_uniform01(const float *scores, int32_t n, float temperature, float uniform01) {
    if (temperature == 0.0f) {
        return CpuArgMax::argmax_as_float(scores, n);
    }
    float best_score = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n; i++) {
        best_score = std::max(best_score, scores[i] / temperature);
    }
    float total = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        total += std::exp(scores[i] / temperature - best_score);
    }
    float sample = uniform01 * total;
    float cdf = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        cdf += std::exp(scores[i] / temperature - best_score);
        if (sample <= cdf) {
            return i;
        }
    }
    return n - 1;
}

template<Qwen35Size QWEN35_SIZE>
BenchOutput bench_gqa(const BenchArgs &args, const Shape &shape) {
    int32_t seq_len = args.seq_len > 0 ? args.seq_len : 128;
    int32_t layer_num = args.layer_num;
    if (layer_num < 0 || layer_num >= shape.num_layers) {
        throw std::runtime_error("--layer-num out of range for preset");
    }
    size_t cache_tokens = static_cast<size_t>(seq_len);
    size_t k_cache_len = cache_tokens * shape.num_layers * shape.keys_size;
    size_t v_cache_len = cache_tokens * shape.num_layers * shape.values_size;

    std::mt19937 rng(args.seed);
    std::vector<float> queries(shape.queries_size);
    std::vector<float> gate(shape.queries_size);
    std::vector<float> keys(k_cache_len);
    std::vector<float> values(v_cache_len);
    std::vector<float> cpu_out(shape.queries_size);
    fill_random(queries, rng);
    fill_random(gate, rng);
    fill_random(keys, rng);
    fill_random(values, rng);

    CudaBuffer d_queries(queries.size() * sizeof(float));
    CudaBuffer d_gate(gate.size() * sizeof(float));
    CudaBuffer d_keys(keys.size() * sizeof(float));
    CudaBuffer d_values(values.size() * sizeof(float));
    CudaBuffer d_out(cpu_out.size() * sizeof(float));
    copy_to_cuda(queries, d_queries);
    copy_to_cuda(gate, d_gate);
    copy_to_cuda(keys, d_keys);
    copy_to_cuda(values, d_values);

    BenchOutput out;
    out.module = "gqa_sdpa";
    out.shapes = {{"seq_len", seq_len}, {"layer_num", layer_num}, {"num_layers", shape.num_layers},
                  {"num_query_heads", shape.num_query_heads}, {"num_kv_heads", shape.num_kv_heads},
                  {"head_size", shape.head_size}};
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        CpuGroupQueryAttention::sdpa(queries.data(), keys.data(), values.data(), cpu_out.data(), gate.data(), layer_num,
                                     seq_len - 1, shape.num_layers, shape.num_query_heads, shape.num_kv_heads,
                                     shape.head_size, shape.keys_size, shape.values_size);
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "gqa_sdpa", [&] {
        GroupQueryAttention<QWEN35_SIZE>::template sdpa<float, float, float, float, float, float>(
            static_cast<float *>(d_queries.data), static_cast<float *>(d_keys.data), static_cast<float *>(d_values.data),
            static_cast<float *>(d_out.data), static_cast<float *>(d_gate.data), layer_num, seq_len - 1, cudaStreamPerThread);
    });
    return out;
}

BenchOutput bench_matvec(const BenchArgs &args, const Shape &shape) {
    int32_t m = args.m > 0 ? args.m : shape.intermediate_size;
    int32_t k = args.k > 0 ? args.k : shape.hidden_size;
    std::mt19937 rng(args.seed);
    std::vector<float> mat(static_cast<size_t>(m) * k);
    std::vector<float> bias(m);
    std::vector<float> vec(k);
    std::vector<float> cpu_out(m);
    fill_random(mat, rng);
    fill_random(bias, rng);
    fill_random(vec, rng);
    CudaBuffer d_mat(mat.size() * sizeof(float));
    CudaBuffer d_bias(bias.size() * sizeof(float));
    CudaBuffer d_vec(vec.size() * sizeof(float));
    CudaBuffer d_out(cpu_out.size() * sizeof(float));
    copy_to_cuda(mat, d_mat);
    copy_to_cuda(bias, d_bias);
    copy_to_cuda(vec, d_vec);

    BenchOutput out;
    out.module = "matvec";
    out.shapes = {{"m", m}, {"k", k}};
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        CpuMatrixVectorMultiply::matmul(m, k, mat.data(), bias.data(), vec.data(), cpu_out.data());
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "matvec", [&] {
        MatrixVectorMultiply::matmul<float, float, float, float, float>(
            m, k, static_cast<float *>(d_mat.data), static_cast<float *>(d_bias.data), static_cast<float *>(d_vec.data),
            static_cast<float *>(d_out.data), cudaStreamPerThread);
    });
    return out;
}

BenchOutput bench_layernorm(const BenchArgs &args, const Shape &shape, const std::string &module) {
    int32_t n = args.n > 0 ? args.n : shape.hidden_size;
    int32_t rows = args.rows > 0 ? args.rows : shape.num_query_heads;
    int32_t cols = args.cols > 0 ? args.cols : shape.head_size;
    std::mt19937 rng(args.seed);
    BenchOutput out;
    out.module = module;

    if (module == "zero_centered_rms_norm") {
        std::vector<float> weight(n), input(n), cpu_out(n);
        fill_random(weight, rng);
        fill_random(input, rng);
        auto d_weight = std::make_shared<CudaBuffer>(weight.size() * sizeof(float));
        auto d_input = std::make_shared<CudaBuffer>(input.size() * sizeof(float));
        auto d_out = std::make_shared<CudaBuffer>(cpu_out.size() * sizeof(float));
        copy_to_cuda(weight, *d_weight);
        copy_to_cuda(input, *d_input);
        LayerNorm layernorm(n);
        layernorm.weights = d_weight;
        out.shapes = {{"n", n}};
        out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
            CpuLayerNorm::zero_centered_rms_norm(weight.data(), input.data(), cpu_out.data(), n, shape.rms_norm_eps);
        });
        out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "zero_centered_rms_norm", [&] {
            layernorm.zero_centered_rms_norm<float, float, float, float>(d_input, d_out, n, shape.rms_norm_eps, cudaStreamPerThread);
        });
        return out;
    }

    if (module == "gated_rms_norm") {
        size_t len = static_cast<size_t>(rows) * cols;
        std::vector<float> weight(cols), input(len), gate(len), cpu_out(len);
        fill_random(weight, rng);
        fill_random(input, rng);
        fill_random(gate, rng);
        auto d_weight = std::make_shared<CudaBuffer>(weight.size() * sizeof(float));
        auto d_input = std::make_shared<CudaBuffer>(input.size() * sizeof(float));
        auto d_gate = std::make_shared<CudaBuffer>(gate.size() * sizeof(float));
        auto d_out = std::make_shared<CudaBuffer>(cpu_out.size() * sizeof(float));
        copy_to_cuda(weight, *d_weight);
        copy_to_cuda(input, *d_input);
        copy_to_cuda(gate, *d_gate);
        LayerNorm layernorm(cols);
        layernorm.weights = d_weight;
        out.shapes = {{"rows", rows}, {"cols", cols}};
        out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
            CpuLayerNorm::gated_rms_norm(weight.data(), input.data(), gate.data(), cpu_out.data(), rows, cols, shape.rms_norm_eps);
        });
        out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "gated_rms_norm", [&] {
            layernorm.normalize_gated_hidden_state<float, float, float, float, float>(d_input, d_gate, d_out, rows, cols, shape.rms_norm_eps, cudaStreamPerThread);
        });
        return out;
    }

    if (module == "l2_norm_rows") {
        size_t len = static_cast<size_t>(rows) * cols;
        std::vector<float> input(len), cpu_values;
        fill_random(input, rng);
        cpu_values = input;
        auto d_values = std::make_shared<CudaBuffer>(input.size() * sizeof(float));
        copy_to_cuda(input, *d_values);
        out.shapes = {{"rows", rows}, {"cols", cols}};
        std::vector<float> cpu_time_values = input;
        out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
            CpuLayerNorm::l2norm_rows(cpu_time_values.data(), rows, cols, 1.0f, shape.rms_norm_eps);
        });
        out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "l2_norm_rows", [&] {
            LayerNorm::l2_norm_rows<float, float>(d_values, rows, cols, 1.0f, shape.rms_norm_eps, cudaStreamPerThread);
        });
        return out;
    }

    throw std::runtime_error("unknown layernorm module: " + module);
}

BenchOutput bench_silu(const BenchArgs &args, const Shape &shape) {
    int32_t n = args.n > 0 ? args.n : shape.intermediate_size;
    std::mt19937 rng(args.seed);
    std::vector<float> x(n), y(n), cpu_x;
    fill_random(x, rng);
    fill_random(y, rng);
    cpu_x = x;
    auto d_x = std::make_shared<CudaBuffer>(x.size() * sizeof(float));
    auto d_y = std::make_shared<CudaBuffer>(y.size() * sizeof(float));
    copy_to_cuda(x, *d_x);
    copy_to_cuda(y, *d_y);
    BenchOutput out;
    out.module = "silu_mult";
    out.shapes = {{"n", n}};
    std::vector<float> cpu_time_x = x;
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        CpuSiLUMult::silu_mult_in_place(cpu_time_x.data(), y.data(), n);
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "silu_mult", [&] {
        SiLUMult::silu_mult_in_place<float, float, float>(d_x, d_y, n, cudaStreamPerThread);
    });
    return out;
}

BenchOutput bench_rope(const BenchArgs &args, const Shape &shape) {
    int32_t position = args.seq_len > 0 ? args.seq_len - 1 : 127;
    std::mt19937 rng(args.seed);
    std::vector<float> queries(shape.queries_size), keys(shape.keys_size);
    fill_random(queries, rng);
    fill_random(keys, rng);
    std::vector<float> cpu_queries = queries;
    std::vector<float> cpu_keys = keys;
    CudaBuffer d_queries(queries.size() * sizeof(float));
    CudaBuffer d_keys(keys.size() * sizeof(float));
    copy_to_cuda(queries, d_queries);
    copy_to_cuda(keys, d_keys);
    BenchOutput out;
    out.module = "rope";
    out.shapes = {{"num_query_heads", shape.num_query_heads}, {"num_kv_heads", shape.num_kv_heads},
                  {"head_size", shape.head_size}, {"rotary_dim", shape.rotary_dim}, {"position", position}};
    std::vector<float> cpu_time_queries = queries;
    std::vector<float> cpu_time_keys = keys;
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        CpuRoPE::apply_partial_rope_to_qk(cpu_time_queries.data(), shape.num_query_heads, cpu_time_keys.data(), shape.num_kv_heads,
                                          shape.head_size, shape.rotary_dim, position, shape.rope_theta_base);
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "rope", [&] {
        RoPE::apply_rope_to_qk<float, float>(static_cast<float *>(d_queries.data), shape.num_query_heads, shape.head_size,
                                             shape.rotary_dim, position, shape.rope_theta_base, cudaStreamPerThread);
        RoPE::apply_rope_to_qk<float, float>(static_cast<float *>(d_keys.data), shape.num_kv_heads, shape.head_size,
                                             shape.rotary_dim, position, shape.rope_theta_base, cudaStreamPerThread);
    });
    return out;
}

BenchOutput bench_argmax(const BenchArgs &args, const Shape &) {
    int32_t n = args.n > 0 ? args.n : (args.vocab_size > 0 ? args.vocab_size : 248320);
    std::mt19937 rng(args.seed);
    std::vector<float> values(n);
    fill_random(values, rng);
    auto d_values_ptr = std::make_shared<CudaBuffer>(values.size() * sizeof(float));
    copy_to_cuda(values, *d_values_ptr);
    ArgMax argmax(n);
    BenchOutput out;
    out.module = "argmax";
    out.shapes = {{"n", n}};
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        volatile int32_t idx = CpuArgMax::argmax_as_float(values.data(), n);
        (void)idx;
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "argmax", [&] {
        argmax.argmax_as_float<float>(d_values_ptr, n, cudaStreamPerThread);
    });
    return out;
}

BenchOutput bench_sampling(const BenchArgs &args, const Shape &) {
    int32_t n = args.n > 0 ? args.n : (args.vocab_size > 0 ? args.vocab_size : 248320);
    float temperature = args.temperature;
    std::mt19937 rng(args.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float uniform01 = uniform(rng);
    std::vector<float> scores(n);
    fill_random(scores, rng);
    auto d_scores_ptr = std::make_shared<CudaBuffer>(scores.size() * sizeof(float));
    copy_to_cuda(scores, *d_scores_ptr);
    Sampling sampling(n);
    BenchOutput out;
    out.module = "sampling";
    out.shapes = {{"n", n}, {"temperature", temperature}};
    out.cpu_total_us = time_cpu(args.cpu_iters, [&] {
        volatile int32_t idx = cpu_sample_with_uniform01(scores.data(), n, temperature, uniform01);
        (void)idx;
    });
    out.gpu_total_us = time_gpu(args.warmup_iters, args.gpu_iters, "sampling", [&] {
        sampling.sample(d_scores_ptr, n, temperature, uniform01, cudaStreamPerThread);
    });
    return out;
}

BenchOutput dispatch_one(const BenchArgs &args, const std::string &module) {
    Shape shape = shape_for_preset(args.preset);
    if (module == "matvec") return bench_matvec(args, shape);
    if (module == "zero_centered_rms_norm" || module == "gated_rms_norm" || module == "l2_norm_rows") {
        return bench_layernorm(args, shape, module);
    }
    if (module == "silu_mult") return bench_silu(args, shape);
    if (module == "rope") return bench_rope(args, shape);
    if (module == "argmax") return bench_argmax(args, shape);
    if (module == "sampling") return bench_sampling(args, shape);
    if (module == "gqa_sdpa") {
        if (args.preset == "qwen35-0.8b") return bench_gqa<QWEN35_0_8B>(args, shape);
        if (args.preset == "qwen35-4b") return bench_gqa<QWEN35_4B>(args, shape);
        if (args.preset == "qwen35-9b") return bench_gqa<QWEN35_9B>(args, shape);
    }
    throw std::runtime_error("unsupported --module value: " + module);
}

nlohmann::json output_json(const BenchArgs &args, const BenchOutput &bench) {
    double gpu_per_iter = args.gpu_iters > 0 ? bench.gpu_total_us / args.gpu_iters : 0.0;
    double cpu_per_iter = args.cpu_iters > 0 ? bench.cpu_total_us / args.cpu_iters : 0.0;
    return {
        {"timestamp", current_timestamp()},
        {"label", args.label},
        {"git", {{"commit", TRANSFORMER_GIT_COMMIT}}},
        {"benchmark", {
            {"name", "qwen35_module_bench"},
            {"module", bench.module},
            {"preset", args.preset},
            {"dtype", args.dtype},
        }},
        {"config", {
            {"seed", args.seed},
            {"warmup_iters", args.warmup_iters},
            {"gpu_iters", args.gpu_iters},
            {"cpu_iters", args.cpu_iters},
            {"shapes", bench.shapes},
        }},
        {"result", {
            {"cpu_total_us", bench.cpu_total_us},
            {"cpu_us_per_iter", cpu_per_iter},
            {"gpu_total_us", bench.gpu_total_us},
            {"gpu_us_per_iter", gpu_per_iter},
            {"speedup_cpu_over_gpu", gpu_per_iter > 0.0 ? cpu_per_iter / gpu_per_iter : 0.0},
        }},
    };
}

void validate_args(const BenchArgs &args) {
    if (args.dtype != "fp32") throw std::runtime_error("unsupported --dtype value: " + args.dtype);
    if (args.warmup_iters < 0) throw std::runtime_error("--warmup-iters must be non-negative");
    if (args.gpu_iters <= 0) throw std::runtime_error("--gpu-iters must be positive");
    if (args.cpu_iters < 0) throw std::runtime_error("--cpu-iters must be non-negative");
    if (args.temperature < 0.0f) throw std::runtime_error("--temperature must be non-negative");
    if (args.module.empty()) throw std::runtime_error("--module is required");
}

} // namespace

int main(int argc, const char *argv[]) {
    argparse::ArgumentParser program("qwen35_module_bench");
    program.add_argument("--module").help("Module to benchmark, or all").default_value(std::string("matvec"));
    program.add_argument("--preset").help("qwen35-0.8b, qwen35-4b, or qwen35-9b").default_value(std::string("qwen35-4b"));
    program.add_argument("--dtype").help("Currently only fp32").default_value(std::string("fp32"));
    program.add_argument("--json").help("Output JSON path; stdout when omitted").default_value(std::string(""));
    program.add_argument("--label").help("Optional run label").default_value(std::string(""));
    program.add_argument("--seed").default_value(0).scan<'d', int32_t>();
    program.add_argument("--warmup-iters").default_value(10).scan<'d', int32_t>();
    program.add_argument("--gpu-iters").default_value(100).scan<'d', int32_t>();
    program.add_argument("--cpu-iters").default_value(3).scan<'d', int32_t>();
    program.add_argument("--m").default_value(0).scan<'d', int32_t>();
    program.add_argument("--k").default_value(0).scan<'d', int32_t>();
    program.add_argument("--n").default_value(0).scan<'d', int32_t>();
    program.add_argument("--rows").default_value(0).scan<'d', int32_t>();
    program.add_argument("--cols").default_value(0).scan<'d', int32_t>();
    program.add_argument("--seq-len").default_value(0).scan<'d', int32_t>();
    program.add_argument("--layer-num").default_value(0).scan<'d', int32_t>();
    program.add_argument("--vocab-size").default_value(0).scan<'d', int32_t>();
    program.add_argument("--temperature").default_value(1.0f).scan<'g', float>();

    try {
        program.parse_args(argc, argv);
        BenchArgs args{
            .module = program.get<std::string>("module"),
            .preset = program.get<std::string>("preset"),
            .dtype = program.get<std::string>("dtype"),
            .json_path = program.get<std::string>("json"),
            .label = program.get<std::string>("label"),
            .seed = program.get<int32_t>("seed"),
            .warmup_iters = program.get<int32_t>("warmup-iters"),
            .gpu_iters = program.get<int32_t>("gpu-iters"),
            .cpu_iters = program.get<int32_t>("cpu-iters"),
            .m = program.get<int32_t>("m"),
            .k = program.get<int32_t>("k"),
            .n = program.get<int32_t>("n"),
            .rows = program.get<int32_t>("rows"),
            .cols = program.get<int32_t>("cols"),
            .seq_len = program.get<int32_t>("seq-len"),
            .layer_num = program.get<int32_t>("layer-num"),
            .vocab_size = program.get<int32_t>("vocab-size"),
            .temperature = program.get<float>("temperature"),
        };
        validate_args(args);

        std::vector<std::string> modules;
        if (args.module == "all") {
            modules = {"matvec", "zero_centered_rms_norm", "gated_rms_norm", "l2_norm_rows",
                       "silu_mult", "rope", "argmax", "sampling", "gqa_sdpa"};
        } else {
            modules = {args.module};
        }

        nlohmann::json output;
        if (modules.size() == 1) {
            output = output_json(args, dispatch_one(args, modules.front()));
        } else {
            output = nlohmann::json::array();
            for (const std::string &module : modules) {
                output.push_back(output_json(args, dispatch_one(args, module)));
            }
        }

        if (args.json_path.empty()) {
            std::cout << output.dump(2) << std::endl;
        } else {
            std::ofstream out(args.json_path);
            if (!out.good()) throw std::runtime_error("failed to open JSON output path: " + args.json_path);
            out << output.dump(2) << std::endl;
        }
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    return 0;
}
