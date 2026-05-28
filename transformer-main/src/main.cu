#include <fstream>
#include <nvtx3/nvToolsExt.h>

#include "vendor/argparse.hpp"
#include "vendor/json.hpp"
#include "qwen2/Qwen2Config.h"
#include "qwen2/Qwen2Model.cuh"
#include "qwen2/Qwen2Loader.h"
#include "qwen35/Qwen35Loader.h"
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"
#include "BPE.h"

template<Qwen2Size QWEN2_SIZE>
class MainRunner {
public:
    int32_t max_seq_len{};
    int32_t seq_len{0};
    bool interactive{};
    std::string system_prompt{};
    std::shared_ptr<Qwen2Model<QWEN2_SIZE>> model{};
    cudaStream_t stream{};

    std::shared_ptr<CudaBuffer> k_cache{};
    std::shared_ptr<CudaBuffer> v_cache{};

    std::shared_ptr<BPE> tokenizer;

    explicit MainRunner(argparse::ArgumentParser &program) {
        this->max_seq_len = program.get<int32_t>("max-seq-len");
        this->interactive = program.get<bool>("interactive");
        this->system_prompt = program.get<std::string>("system-prompt");

        auto model_dir = Qwen2Loader::get_model_dir();

        using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
        this->model = Qwen2Loader::load_qwen2<QWEN2_SIZE>(model_dir + "/model.safetensors", max_seq_len);
        this->tokenizer = std::make_shared<BPE>(model_dir);

        cudaStream_t stream{};
        checkCuda(cudaStreamCreate(&stream));

        this->k_cache = std::make_shared<CudaBuffer>(max_seq_len * Qwen2Config::num_layers() * Qwen2Config::keys_size() * sizeof(__nv_bfloat16));
        this->v_cache = std::make_shared<CudaBuffer>(max_seq_len * Qwen2Config::num_layers() * Qwen2Config::values_size() * sizeof(__nv_bfloat16));
    }

    void run() {
        if (this->interactive) {
            this->run_interactive();
        } else {
            this->run_autoregressive_test();
        }
    }

    void run_interactive() {
        // follows jinja2 chat template from tokenizer_config.json
        this->send_text("<|im_start|>system\n" + this->system_prompt);
        uint32_t im_end_token = tokenizer->inverse_vocab.at("<|im_end|>");
        while (true) {
            std::cout << "> ";
            std::string prompt;
            if (!std::getline(std::cin, prompt)) {
                break;
            }
            this->send_text("<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant");
            // autoregressive
            uint32_t latest_token = tokenizer->inverse_vocab_char.at('\n');
            while (true) {
                if (++this->seq_len > this->max_seq_len) {
                    throw std::runtime_error("max sequence length reached");
                }
                uint32_t new_token = this->model->forward(this->k_cache, this->v_cache, this->seq_len, latest_token, 0.0f);
                if (new_token == im_end_token) {
                    std::cout << std::endl;
                    break;
                }
                std::cout << this->tokenizer->decode(new_token) << std::flush;
                latest_token = new_token;
            }
        }
    }

    /**
     * Prefill a string through the model.
     * We don't implement a masked-matrix multiply operation, so internally this function calls the autoregressive
     * decoding function many times and ignores the predicted token.
     */
    void send_text(const std::string &text) {
        for (uint32_t user_token : tokenizer->encode(text)) {
            if (++seq_len > max_seq_len) {
                throw std::runtime_error("max sequence length reached");
            }
            model->forward(k_cache, v_cache, seq_len, user_token, 0.0f);
        }
    }

    void run_autoregressive_test() {
        // autoregressive generation test, matching python implementation
        uint32_t latest_token = 64; // sequence begins with "a" token
        for (size_t i = 0; i < max_seq_len; i++) {
            size_t seq_len = i + 1;
            if (seq_len == max_seq_len) {
                // add profiling range on last token
                nvtxRangePush("last_token");
            }
            uint32_t new_token = model->forward(k_cache, v_cache, seq_len, latest_token, 0.0f);
            std::cout << tokenizer->decode(new_token) << std::flush;
            latest_token = new_token;
            if (seq_len == max_seq_len) {
                nvtxRangePop();
            }
        }
    }
};

template<Qwen35Size QWEN35_SIZE>
class Qwen35MainRunner {
public:
    using Qwen35ModelT = Qwen35Model<QWEN35_SIZE>;

    int32_t max_seq_len{};
    int32_t seq_len{0};
    bool interactive{};
    std::string system_prompt{};
    std::shared_ptr<Qwen35ModelT> model{};
    Qwen35Cache cache{};
    std::shared_ptr<BPE> tokenizer;

    explicit Qwen35MainRunner(argparse::ArgumentParser &program) {
        this->max_seq_len = program.get<int32_t>("max-seq-len");
        this->interactive = program.get<bool>("interactive");
        this->system_prompt = program.get<std::string>("system-prompt");

        auto model_dir = Qwen35Loader::get_model_dir();
        this->model = Qwen35Loader::load_qwen35<QWEN35_SIZE>(model_dir);
        this->cache = this->model->allocate_cache(max_seq_len);
        this->tokenizer = std::make_shared<BPE>(model_dir);
    }

    void run() {
        if (this->interactive) {
            this->run_interactive();
        } else {
            this->run_autoregressive_test();
        }
    }

    void run_interactive() {
        this->send_text("<|im_start|>system\n" + this->system_prompt);
        uint32_t im_end_token = tokenizer->inverse_vocab.at("<|im_end|>");
        while (true) {
            std::cout << "> ";
            std::string prompt;
            if (!std::getline(std::cin, prompt)) {
                break;
            }
            this->send_text("<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant");
            uint32_t latest_token = tokenizer->inverse_vocab_char.at('\n');
            while (true) {
                if (this->cache.seq_len >= static_cast<size_t>(this->max_seq_len)) {
                    throw std::runtime_error("max sequence length reached");
                }
                uint32_t new_token = this->model->forward(this->cache, latest_token, 0.0f);
                if (new_token == im_end_token) {
                    std::cout << std::endl;
                    break;
                }
                std::cout << this->tokenizer->decode(new_token) << std::flush;
                latest_token = new_token;
            }
        }
    }

    void send_text(const std::string &text) {
        for (uint32_t user_token : tokenizer->encode(text)) {
            if (this->cache.seq_len >= static_cast<size_t>(this->max_seq_len)) {
                throw std::runtime_error("max sequence length reached");
            }
            model->forward(cache, user_token, 0.0f);
        }
    }

    void run_autoregressive_test() {
        uint32_t latest_token = 64;
        for (size_t i = 0; i < static_cast<size_t>(max_seq_len); i++) {
            if (i + 1 == static_cast<size_t>(max_seq_len)) {
                nvtxRangePush("last_token");
            }
            uint32_t new_token = model->forward(cache, latest_token, 0.0f);
            std::cout << tokenizer->decode(new_token) << std::flush;
            latest_token = new_token;
            if (i + 1 == static_cast<size_t>(max_seq_len)) {
                nvtxRangePop();
            }
        }
    }
};

int main(int argc, const char *argv[])
{
    argparse::ArgumentParser program("transformer");

    program.add_argument("--max-seq-len")
        .help("Maximum sequence length in test mode and interactive mode")
        .default_value(100)
    .scan<'d', int32_t>();

    program.add_argument("--interactive")
        .help("Ask questions on the command line")
        .flag();

    program.add_argument("--system-prompt")
        .help("System message for interactive mode")
        .default_value("You are a helpful assistant.");

    program.add_argument("--model")
        .help("Model implementation to run: qwen2 or qwen35")
        .default_value(std::string("qwen35"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto model_kind = program.get<std::string>("model");
    if (model_kind == "qwen35") {
        auto config_json_path = Qwen35Loader::get_model_dir() + "/config.json";
        std::ifstream config_file(config_json_path);
        if (!config_file.good()) {
            std::cerr << "Failed to open " << config_json_path << std::endl;
            return 1;
        }
        nlohmann::json root = nlohmann::json::parse(config_file);
        nlohmann::json config = root.contains("text_config") ? root["text_config"] : root;

        if (config["hidden_size"] == Qwen35Config<QWEN35_0_8B>::hidden_size() &&
            config["intermediate_size"] == Qwen35Config<QWEN35_0_8B>::intermediate_size()) {
            Qwen35MainRunner<QWEN35_0_8B>(program).run();
        } else if (config["hidden_size"] == Qwen35Config<QWEN35_4B>::hidden_size() &&
            config["intermediate_size"] == Qwen35Config<QWEN35_4B>::intermediate_size()) {
            Qwen35MainRunner<QWEN35_4B>(program).run();
        } else if (config["hidden_size"] == Qwen35Config<QWEN35_9B>::hidden_size() &&
                   config["intermediate_size"] == Qwen35Config<QWEN35_9B>::intermediate_size()) {
            Qwen35MainRunner<QWEN35_9B>(program).run();
        } else {
            std::cerr << "Unknown Qwen3.5 model size with hidden_size: " << config["hidden_size"]
                      << ", intermediate_size: " << config["intermediate_size"] << std::endl;
            return 1;
        }
        return 0;
    }
    if (model_kind != "qwen2") {
        std::cerr << "Unknown --model value: " << model_kind << std::endl;
        return 1;
    }

    auto config_json_path = Qwen2Loader::get_model_dir() + "/config.json";
    std::ifstream config_file(config_json_path);
    if (!config_file.good()) {
        std::cerr << "Failed to open " << config_json_path << std::endl;
        return 1;
    }
    nlohmann::json config = nlohmann::json::parse(config_file);

    // Verify this is a Qwen2 model
    if (config["architectures"][0] != "Qwen2ForCausalLM") {
        std::cerr << "Model is not Qwen2ForCausalLM, found: " << config["architectures"][0] << std::endl;
        return 1;
    }

    // Detect model config based on intermediate_size
    if (config["intermediate_size"] == Qwen2Config<QWEN2_0_5B>::intermediate_size()) {
        MainRunner<QWEN2_0_5B>(program).run();
    } else {
        std::cerr << "Unknown Qwen2 model size with intermediate_size: " << config["intermediate_size"] << std::endl;
        return 1;
    }

    return 0;
}
