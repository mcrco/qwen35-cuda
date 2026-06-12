#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "../BPE.h"
#include "../ErrorCheck.h"
#include "../qwen35/Qwen35Loader.h"
#include "../vendor/argparse.hpp"
#include "../vendor/json.hpp"

#ifndef TRANSFORMER_GIT_COMMIT
#define TRANSFORMER_GIT_COMMIT "unknown"
#endif

#ifndef TRANSFORMER_REPO_ROOT
#define TRANSFORMER_REPO_ROOT "."
#endif

namespace {

constexpr const char *DEFAULT_PROMPT =
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain CUDA kernels briefly.<|im_end|>\n"
    "<|im_start|>assistant\n";

struct BenchArgs {
    int32_t max_seq_len{};
    int32_t warmup_tokens{};
    int32_t measure_tokens{};
    int32_t input_token{};
    float temperature{};
    int32_t seed{};
    bool prefill{};
    std::string prompt;
    std::string json_path;
    std::string label;
    std::string git_commit;
    std::string dtype;
};

struct BenchResult {
    std::string model_size;
    int32_t prefill_tokens{};
    size_t start_seq_len{};
    size_t end_seq_len{};
    float gpu_total_us{};
    double wall_total_us{};
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

std::filesystem::path json_output_dir(const std::string &json_path) {
    std::filesystem::path path(json_path);
    if (path.has_extension()) {
        return path.parent_path();
    }
    return path;
}

std::string short_git_commit(const BenchArgs &args) {
    std::string commit = args.git_commit;
    return commit.substr(0, std::min<size_t>(commit.size(), 12));
}

std::filesystem::path json_output_path(const BenchArgs &args) {
    return json_output_dir(args.json_path) / ("forward-" + short_git_commit(args) + ".json");
}

void validate_args(const BenchArgs &args) {
    if (args.dtype != "fp32") {
        throw std::runtime_error("unsupported --dtype value: " + args.dtype + " (only fp32 is supported)");
    }
    if (args.max_seq_len <= 0) {
        throw std::runtime_error("--max-seq-len must be positive");
    }
    if (args.warmup_tokens < 0) {
        throw std::runtime_error("--warmup-tokens must be non-negative");
    }
    if (args.measure_tokens <= 0) {
        throw std::runtime_error("--measure-tokens must be positive");
    }
    if (args.input_token < 0) {
        throw std::runtime_error("--input-token must be non-negative");
    }
    if (args.temperature < 0.0f) {
        throw std::runtime_error("--temperature must be non-negative");
    }
    if (args.seed < 0) {
        throw std::runtime_error("--seed must be non-negative");
    }
}

template<Qwen35Size QWEN35_SIZE>
BenchResult run_bench(const BenchArgs &args, const std::string &model_size) {
    using Model = Qwen35Model<QWEN35_SIZE, float, float, float>;

    std::shared_ptr<Model> model;
    {
        nvtxRangePush("qwen35_bench_load_model");
        model = Qwen35Loader::load_qwen35<QWEN35_SIZE, float, float, float>(Qwen35Loader::get_model_dir());
        model->set_seed(static_cast<uint32_t>(args.seed));
        nvtxRangePop();
    }

    Qwen35Cache cache;
    {
        nvtxRangePush("qwen35_bench_allocate_cache");
        cache = model->allocate_cache(static_cast<size_t>(args.max_seq_len));
        nvtxRangePop();
    }

    int32_t latest_token = args.input_token;
    int32_t prefill_tokens = 0;

    if (args.prefill) {
        nvtxRangePush("qwen35_bench_prefill");
        BPE tokenizer(Qwen35Loader::get_model_dir());
        const std::string &prompt = args.prompt.empty() ? DEFAULT_PROMPT : args.prompt;
        std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt);
        prefill_tokens = static_cast<int32_t>(prompt_tokens.size());

        if (prefill_tokens + args.warmup_tokens + args.measure_tokens > args.max_seq_len) {
            nvtxRangePop();
            throw std::runtime_error("prefill tokens + warmup tokens + measure tokens exceed --max-seq-len");
        }

        for (uint32_t token : prompt_tokens) {
            latest_token = model->forward(cache, static_cast<int32_t>(token), args.temperature);
        }
        nvtxRangePop();
    } else if (args.warmup_tokens + args.measure_tokens > args.max_seq_len) {
        throw std::runtime_error("warmup tokens + measure tokens exceed --max-seq-len");
    }

    {
        nvtxRangePush("qwen35_bench_warmup");
        for (int32_t i = 0; i < args.warmup_tokens; i++) {
            latest_token = model->forward(cache, latest_token, args.temperature);
        }
        nvtxRangePop();
    }

    BenchResult result;
    result.model_size = model_size;
    result.prefill_tokens = prefill_tokens;
    result.start_seq_len = cache.seq_len;

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    checkCuda(cudaEventCreate(&start_event));
    checkCuda(cudaEventCreate(&stop_event));
    checkCuda(cudaDeviceSynchronize());

    nvtxRangePush("qwen35_bench_measure");
    checkCuda(cudaEventRecord(start_event, model->cuda_stream()));
    auto wall_start = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < args.measure_tokens; i++) {
        if (i + 1 == args.measure_tokens) {
            nvtxRangePush("last_token");
        }
        nvtxRangePush("qwen35_bench_token");
        latest_token = model->forward(cache, latest_token, args.temperature);
        nvtxRangePop();
        if (i + 1 == args.measure_tokens) {
            nvtxRangePop();
        }
    }
    auto wall_stop = std::chrono::steady_clock::now();
    checkCuda(cudaEventRecord(stop_event, model->cuda_stream()));
    checkCuda(cudaEventSynchronize(stop_event));
    nvtxRangePop();

    float gpu_total_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_total_ms, start_event, stop_event));
    checkCuda(cudaEventDestroy(start_event));
    checkCuda(cudaEventDestroy(stop_event));

    result.end_seq_len = cache.seq_len;
    result.gpu_total_us = gpu_total_ms * 1000.0f;
    result.wall_total_us = std::chrono::duration<double, std::micro>(wall_stop - wall_start).count();
    return result;
}

nlohmann::json make_json(const BenchArgs &args, const BenchResult &result) {
    double measure_tokens = static_cast<double>(args.measure_tokens);
    double gpu_total_us = static_cast<double>(result.gpu_total_us);
    double wall_total_us = result.wall_total_us;

    return {
        {"timestamp", current_timestamp()},
        {"label", args.label},
        {"git", {
            {"commit", args.git_commit},
        }},
        {"model", {
            {"name", "qwen35"},
            {"size", result.model_size},
            {"dtype", {
                {"weight", "fp32"},
                {"hidden", "fp32"},
                {"compute", "fp32"},
            }},
        }},
        {"config", {
            {"max_seq_len", args.max_seq_len},
            {"prefill_enabled", args.prefill},
            {"prefill_tokens", result.prefill_tokens},
            {"warmup_tokens", args.warmup_tokens},
            {"measure_tokens", args.measure_tokens},
            {"input_token", args.input_token},
            {"temperature", args.temperature},
            {"seed", args.seed},
        }},
        {"result", {
            {"start_seq_len", result.start_seq_len},
            {"end_seq_len", result.end_seq_len},
            {"gpu_total_us", gpu_total_us},
            {"gpu_us_per_token", gpu_total_us / measure_tokens},
            {"wall_total_us", wall_total_us},
            {"wall_us_per_token", wall_total_us / measure_tokens},
            {"tokens_per_sec", measure_tokens / (wall_total_us / 1000000.0)},
        }},
    };
}

BenchResult dispatch_bench(const BenchArgs &args) {
    auto config_json_path = Qwen35Loader::get_model_dir() + "/config.json";
    std::ifstream config_file(config_json_path);
    if (!config_file.good()) {
        throw std::runtime_error("failed to open " + config_json_path);
    }
    nlohmann::json root = nlohmann::json::parse(config_file);
    nlohmann::json config = root.contains("text_config") ? root["text_config"] : root;

    if (config["hidden_size"] == Qwen35Config<QWEN35_0_8B>::hidden_size() &&
        config["intermediate_size"] == Qwen35Config<QWEN35_0_8B>::intermediate_size()) {
        return run_bench<QWEN35_0_8B>(args, "0.8B");
    }
    if (config["hidden_size"] == Qwen35Config<QWEN35_4B>::hidden_size() &&
        config["intermediate_size"] == Qwen35Config<QWEN35_4B>::intermediate_size()) {
        return run_bench<QWEN35_4B>(args, "4B");
    }
    if (config["hidden_size"] == Qwen35Config<QWEN35_9B>::hidden_size() &&
        config["intermediate_size"] == Qwen35Config<QWEN35_9B>::intermediate_size()) {
        return run_bench<QWEN35_9B>(args, "9B");
    }

    throw std::runtime_error(
        "unknown Qwen3.5 model size with hidden_size: " + std::to_string(config["hidden_size"].get<size_t>()) +
        ", intermediate_size: " + std::to_string(config["intermediate_size"].get<size_t>()));
}

} // namespace

int main(int argc, const char *argv[]) {
    argparse::ArgumentParser program("qwen35_forward_bench");

    program.add_argument("--max-seq-len")
        .help("Maximum cache sequence length")
        .default_value(1024)
        .scan<'d', int32_t>();
    program.add_argument("--warmup-tokens")
        .help("Number of warmup forward passes before measurement")
        .default_value(32)
        .scan<'d', int32_t>();
    program.add_argument("--measure-tokens")
        .help("Number of measured forward passes")
        .default_value(128)
        .scan<'d', int32_t>();
    program.add_argument("--input-token")
        .help("Initial token id used before autoregressive chaining")
        .default_value(64)
        .scan<'d', int32_t>();
    program.add_argument("--temperature")
        .help("Sampling temperature; 0 uses argmax")
        .default_value(1.0f)
        .scan<'g', float>();
    program.add_argument("--seed")
        .help("Deterministic sampling seed")
        .default_value(0)
        .scan<'d', int32_t>();
    program.add_argument("--prefill")
        .help("Tokenize and run prompt before warmup and measurement")
        .flag();
    program.add_argument("--prompt")
        .help("Inline prompt text for --prefill")
        .default_value(std::string(""));
    program.add_argument("--json")
        .help("Output JSON directory")
        .default_value(std::string(TRANSFORMER_REPO_ROOT) + "/bench-results");
    program.add_argument("--label")
        .help("Optional run label included in JSON output")
        .default_value(std::string(""));
    program.add_argument("--git-commit")
        .help("Git commit to record in output metadata and filename")
        .default_value(std::string(TRANSFORMER_GIT_COMMIT));
    program.add_argument("--dtype")
        .help("Benchmark dtype config; currently only fp32")
        .default_value(std::string("fp32"));

    try {
        program.parse_args(argc, argv);

        BenchArgs args{
            .max_seq_len = program.get<int32_t>("max-seq-len"),
            .warmup_tokens = program.get<int32_t>("warmup-tokens"),
            .measure_tokens = program.get<int32_t>("measure-tokens"),
            .input_token = program.get<int32_t>("input-token"),
            .temperature = program.get<float>("temperature"),
            .seed = program.get<int32_t>("seed"),
            .prefill = program.get<bool>("prefill"),
            .prompt = program.get<std::string>("prompt"),
            .json_path = program.get<std::string>("json"),
            .label = program.get<std::string>("label"),
            .git_commit = program.get<std::string>("git-commit"),
            .dtype = program.get<std::string>("dtype"),
        };
        validate_args(args);

        BenchResult result = dispatch_bench(args);
        nlohmann::json output = make_json(args, result);
        if (args.json_path.empty()) {
            std::cout << output.dump(2) << std::endl;
        } else {
            std::filesystem::path output_path = json_output_path(args);
            if (output_path.has_parent_path()) {
                std::filesystem::create_directories(output_path.parent_path());
            }
            std::ofstream out(output_path);
            if (!out.good()) {
                throw std::runtime_error("failed to open JSON output path: " + output_path.string());
            }
            out << output.dump(2) << std::endl;
        }
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    return 0;
}
