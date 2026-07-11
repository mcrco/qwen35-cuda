#include <fstream>
#include <nvtx3/nvToolsExt.h>

#include "vendor/argparse.hpp"
#include "vendor/json.hpp"
#include "qwen35/Qwen35Loader.h"
#include "BPE.h"

template<Qwen35Size QWEN35_SIZE>
class Qwen35MainRunner {
public:
    using Qwen35ModelT = Qwen35Model<QWEN35_SIZE>;

    int32_t max_seq_len{};
    int32_t seq_len{0};
    float temperature{};
    bool interactive{};
    std::string system_prompt{};
    std::shared_ptr<Qwen35ModelT> model{};
    Qwen35Cache cache{};
    std::shared_ptr<BPE> tokenizer;

    explicit Qwen35MainRunner(argparse::ArgumentParser &program) {
        this->max_seq_len = program.get<int32_t>("max-seq-len");
        this->temperature = program.get<float>("temperature");
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
                uint32_t new_token = this->model->forward(this->cache, latest_token, this->temperature);
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

    program.add_argument("--temperature")
        .help("Sampling temperature for interactive assistant generation; 0 uses argmax")
        .default_value(1.0f)
        .scan<'g', float>();

    program.add_argument("--system-prompt")
        .help("System message for interactive mode")
        .default_value("You are a helpful assistant.");

    try {
        program.parse_args(argc, argv);
        if (program.get<float>("temperature") < 0.0f) {
            throw std::runtime_error("--temperature must be non-negative");
        }
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

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
