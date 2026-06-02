#include "../BPE.h"
#include "../qwen35/Qwen35Loader.h"
#include <iostream>

const char *TEST_STR1 = "Hello Hola Bonjour Hallo Ciao Olá Привет 你好 こんにちは 안녕하세요 नमस्ते السلام عليكم שלום Hej Hei Halló Merhaba Szia Sawubona Salam";
// reference values generated with Qwen3.5 tokenizer
const std::vector<uint32_t> TEST_STR1_REFERENCE{9419, 196716, 178499, 180752, 214743, 216958, 239029, 220, 109266, 184545, 191972, 149518, 84237, 150104, 153348, 175732, 217576, 151009, 173924, 1216, 73, 1216, 72, 10607, 1731, 8490, 10259, 64, 42721, 667, 35779, 386, 6510, 212440};

const char *TEST_STR2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2<|im_end|>\n<|im_start|>assistant\n2+2=4<|im_end|>\n";
const std::vector<uint32_t> TEST_STR2_REFERENCE{248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198, 248045, 846, 198, 3710, 369, 220, 17, 10, 17, 248046, 198, 248045, 74455, 198, 17, 10, 17, 28, 19, 248046, 198};

bool check_tokens(const std::vector<uint32_t> &encoded, const std::vector<uint32_t> &reference) {
    for (size_t i = 0; i < std::min(encoded.size(), reference.size()); i++) {
        if (encoded[i] != reference[i]) {
            std::cerr << "wrong token at position " << i << ", got " << encoded[i] << ", expected " << reference[i] << std::endl;
            return false;
        }
    }
    if (encoded.size() != reference.size()) {
        std::cerr << "wrong number of tokens, got " << encoded.size() << ", expected " << reference.size() << std::endl;
        return false;
    }
    return true;
}

void print_tokens(const BPE &bpe, const std::vector<uint32_t> &encoded) {
    for (size_t i = 0; i < encoded.size(); i++) {
        if (i > 0) {
            std::cerr << ", ";
        }
        std::cerr << encoded[i] << "(" << bpe.vocab[encoded[i]] << ")";
    }
    std::cerr << std::endl;
}

void check_and_print(const BPE &bpe, const char *str, const std::vector<uint32_t> &reference) {
    auto encoded = bpe.encode(str);
    if (!check_tokens(encoded, reference)) {
        std::cerr << "encoded:" << std::endl;
        print_tokens(bpe, encoded);
        std::cerr << "reference:" << std::endl;
        print_tokens(bpe, reference);
        std::exit(1);
    }
}

int main() {
    BPE bpe(Qwen35Loader::get_model_dir());

    check_and_print(bpe, TEST_STR1, TEST_STR1_REFERENCE);
    check_and_print(bpe, TEST_STR2, TEST_STR2_REFERENCE);
}
