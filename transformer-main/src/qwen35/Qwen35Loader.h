#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../CudaBuffer.cuh"
#include "../vendor/json.hpp"
#include "../vendor/safetensors.hh"
#include "Qwen35Model.cuh"

class Qwen35TensorIndex {
public:
    explicit Qwen35TensorIndex(const std::string &model_dir);
    bool contains(const std::string &name) const;

    template<typename storage_t>
    std::shared_ptr<CudaBuffer> load_tensor(const std::string &name, size_t expected_dim_0, size_t expected_dim_1 = 0, size_t expected_dim_2 = 0) const;

    template<typename storage_t>
    std::shared_ptr<CudaBuffer> load_tensor_slice_rows(const std::string &name, size_t row_start, size_t rows, size_t cols) const;

private:
    std::map<std::string, std::filesystem::path> tensor_to_file;
};

class Qwen35Loader {
public:
    static std::string get_model_dir();
    static nlohmann::json load_text_config(const std::string &model_dir);

    template<Qwen35Size QWEN35_SIZE>
    static void validate_config(const nlohmann::json &cfg);

    template<
        Qwen35Size QWEN35_SIZE,
        typename weight_t = float,
        typename hidden_t = float,
        typename compute_t = float>
    static std::shared_ptr<Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>> load_qwen35(const std::string &model_dir);
};
