#include "Qwen35Loader.h"

#include "../ErrorCheck.h"
#include "../gpu_ops/GpuFloat.cuh"
#include "Qwen35Layer.cuh"

#include <cuda_bf16.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TRANSFORMER_REPO_ROOT
#define TRANSFORMER_REPO_ROOT "."
#endif

Qwen35TensorIndex::Qwen35TensorIndex(const std::string &model_dir) {
    for (const auto &entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") {
            continue;
        }

        std::string warn, err;
        safetensors::safetensors_t st;
        bool ret = safetensors::mmap_from_file(entry.path().string(), &st, &warn, &err);
        if (!warn.empty()) {
            std::cerr << "safetensors warning: " << warn << std::endl;
        }
        if (!ret) {
            throw std::runtime_error("safetensors error while indexing " + entry.path().string() + ": " + err);
        }
        if (!safetensors::validate_data_offsets(st, err)) {
            throw std::runtime_error("safetensors invalid data offsets in " + entry.path().string() + ": " + err);
        }
        for (size_t i = 0; i < st.tensors.size(); i++) {
            tensor_to_file[st.tensors.keys()[i]] = entry.path();
        }
    }
    if (tensor_to_file.empty()) {
        throw std::runtime_error("no safetensors files found in " + model_dir);
    }
}

bool Qwen35TensorIndex::contains(const std::string &name) const {
    return tensor_to_file.contains(name);
}

static safetensors::tensor_t find_tensor(safetensors::safetensors_t &st, const std::string &name) {
    safetensors::tensor_t tensor;
    for (size_t i = 0; i < st.tensors.size(); i++) {
        if (st.tensors.keys()[i] == name) {
            st.tensors.at(i, &tensor);
            return tensor;
        }
    }
    throw std::runtime_error("failed to find tensor: " + name);
}

static float load_tensor_float_value(const uint8_t *data, safetensors::dtype dtype, size_t idx) {
    switch (dtype) {
        case safetensors::kBFLOAT16:
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(data)[idx]);
        case safetensors::kFLOAT32:
            return reinterpret_cast<const float *>(data)[idx];
        default:
            throw std::runtime_error("unsupported input tensor dtype");
    }
}

template<typename storage_t>
std::shared_ptr<CudaBuffer> Qwen35TensorIndex::load_tensor(const std::string &name, size_t expected_dim_0, size_t expected_dim_1, size_t expected_dim_2) const {
    auto it = tensor_to_file.find(name);
    if (it == tensor_to_file.end()) {
        throw std::runtime_error("failed to find tensor: " + name);
    }

    std::string warn, err;
    safetensors::safetensors_t st;
    bool ret = safetensors::mmap_from_file(it->second.string(), &st, &warn, &err);
    if (!ret) {
        throw std::runtime_error("safetensors error while loading " + name + ": " + err);
    }
    auto tensor = find_tensor(st, name);
    if (tensor.dtype != safetensors::kBFLOAT16 && tensor.dtype != safetensors::kFLOAT32) {
        throw std::runtime_error("unexpected dtype for " + name + ", only BF16 and F32 checkpoint tensors are supported");
    }
    if (tensor.shape.empty() || (expected_dim_0 != 0 && tensor.shape[0] != expected_dim_0)) {
        throw std::runtime_error("unexpected tensor shape for " + name);
    }
    if (expected_dim_1 != 0 && (tensor.shape.size() < 2 || tensor.shape[1] != expected_dim_1)) {
        throw std::runtime_error("unexpected tensor shape for " + name);
    }
    if (expected_dim_2 != 0 && (tensor.shape.size() < 3 || tensor.shape[2] != expected_dim_2)) {
        throw std::runtime_error("unexpected tensor shape for " + name);
    }
    size_t num_els = 1;
    for (auto dim : tensor.shape) {
        num_els *= dim;
    }
    size_t len_bytes = num_els * sizeof(storage_t);
    auto out = std::make_shared<CudaBuffer>(len_bytes);
    const uint8_t *tensor_data_host = st.databuffer_addr + tensor.data_offsets[0];
    auto out_ptr = static_cast<storage_t *>(out->data);
    for (size_t i = 0; i < num_els; i++) {
        out_ptr[i] = gpu_ops::write_from<storage_t>(load_tensor_float_value(tensor_data_host, tensor.dtype, i));
    }
    return out;
}

template<typename storage_t>
std::shared_ptr<CudaBuffer> Qwen35TensorIndex::load_tensor_slice_rows(const std::string &name, size_t row_start, size_t rows, size_t cols) const {
    auto full = load_tensor<storage_t>(name, 0, cols);
    auto out = std::make_shared<CudaBuffer>(rows * cols * sizeof(storage_t));
    auto full_ptr = static_cast<storage_t *>(full->data);
    std::memcpy(out->data, full_ptr + row_start * cols, rows * cols * sizeof(storage_t));
    return out;
}

std::string Qwen35Loader::get_model_dir() {
    const char *model_dir_env = std::getenv("TRANSFORMER_MODEL_DIR");
    if (model_dir_env) {
        return model_dir_env;
    }
    return std::string(TRANSFORMER_REPO_ROOT) + "/models/Qwen3.5-4B";
}

static void expect_config_value(const nlohmann::json &cfg, const std::string &name, size_t expected) {
    size_t got = cfg.at(name).get<size_t>();
    if (got != expected) {
        throw std::runtime_error("unexpected Qwen3.5 config value for " + name + ": got " + std::to_string(got) + ", expected " + std::to_string(expected));
    }
}

nlohmann::json Qwen35Loader::load_text_config(const std::string &model_dir) {
    std::ifstream config_file(model_dir + "/config.json");
    if (!config_file.good()) {
        throw std::runtime_error("failed to open " + model_dir + "/config.json");
    }
    nlohmann::json root = nlohmann::json::parse(config_file);
    return root.contains("text_config") ? root["text_config"] : root;
}

template<Qwen35Size QWEN35_SIZE>
void Qwen35Loader::validate_config(const nlohmann::json &cfg) {
    expect_config_value(cfg, "hidden_size", Qwen35Config<QWEN35_SIZE>::hidden_size());
    expect_config_value(cfg, "num_hidden_layers", Qwen35Config<QWEN35_SIZE>::num_layers());
    expect_config_value(cfg, "intermediate_size", Qwen35Config<QWEN35_SIZE>::intermediate_size());
    expect_config_value(cfg, "vocab_size", Qwen35Config<QWEN35_SIZE>::vocab_size());
    expect_config_value(cfg, "num_attention_heads", Qwen35Config<QWEN35_SIZE>::num_query_heads());
    expect_config_value(cfg, "num_key_value_heads", Qwen35Config<QWEN35_SIZE>::num_kv_heads());
    expect_config_value(cfg, "head_dim", Qwen35Config<QWEN35_SIZE>::head_size());
    expect_config_value(cfg, "linear_num_key_heads", Qwen35Config<QWEN35_SIZE>::linear_num_key_heads());
    expect_config_value(cfg, "linear_num_value_heads", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads());
    expect_config_value(cfg, "linear_key_head_dim", Qwen35Config<QWEN35_SIZE>::linear_key_head_dim());
    expect_config_value(cfg, "linear_value_head_dim", Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());
    expect_config_value(cfg, "linear_conv_kernel_dim", Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim());

    bool tying = cfg.value("tie_word_embeddings", false);
    if (tying != Qwen35Config<QWEN35_SIZE>::embedding_tying()) {
        throw std::runtime_error("unexpected Qwen3.5 config value for tie_word_embeddings");
    }

    auto rope = cfg.value("rope_parameters", nlohmann::json::object());
    size_t rotary_dim = static_cast<size_t>(static_cast<float>(cfg.at("head_dim").get<size_t>()) * rope.value("partial_rotary_factor", 1.0f));
    if (rotary_dim != Qwen35Config<QWEN35_SIZE>::rotary_dim()) {
        throw std::runtime_error("unexpected Qwen3.5 rotary_dim computed from partial_rotary_factor");
    }
}

template<Qwen35Size QWEN35_SIZE, typename weight_t, typename hidden_t, typename compute_t>
std::shared_ptr<Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>> Qwen35Loader::load_qwen35(const std::string &model_dir) {
    auto text_config = load_text_config(model_dir);
    validate_config<QWEN35_SIZE>(text_config);
    auto tensors = Qwen35TensorIndex(model_dir);
    auto model = std::make_shared<Qwen35Model<QWEN35_SIZE, weight_t, hidden_t, compute_t>>();
    std::string prefix = tensors.contains("model.language_model.embed_tokens.weight") ? "model.language_model" : "model";

    model->embedding_weight = tensors.template load_tensor<weight_t>(prefix + ".embed_tokens.weight", Qwen35Config<QWEN35_SIZE>::vocab_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
    model->final_layernorm.weights = tensors.template load_tensor<weight_t>(prefix + ".norm.weight", Qwen35Config<QWEN35_SIZE>::hidden_size());
    if (Qwen35Config<QWEN35_SIZE>::embedding_tying() || !tensors.contains("lm_head.weight")) {
        model->lm_head_weight = model->embedding_weight;
    } else {
        model->lm_head_weight = tensors.template load_tensor<weight_t>("lm_head.weight", Qwen35Config<QWEN35_SIZE>::vocab_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
    }

    for (size_t i = 0; i < Qwen35Config<QWEN35_SIZE>::num_layers(); i++) {
        std::string layer_prefix = prefix + ".layers." + std::to_string(i);
        std::shared_ptr<Qwen35Layer<QWEN35_SIZE, weight_t, hidden_t, compute_t>> layer;
        if (Qwen35Config<QWEN35_SIZE>::full_attention_layer(i)) {
            auto full = std::make_shared<Qwen35FullAttnLayer<QWEN35_SIZE, weight_t, hidden_t, compute_t>>(i);
            full->q_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.q_proj.weight", 2 * Qwen35Config<QWEN35_SIZE>::queries_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            if (tensors.contains(layer_prefix + ".self_attn.q_proj.bias")) full->q_proj_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.q_proj.bias", 2 * Qwen35Config<QWEN35_SIZE>::queries_size());
            full->k_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.k_proj.weight", Qwen35Config<QWEN35_SIZE>::keys_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            if (tensors.contains(layer_prefix + ".self_attn.k_proj.bias")) full->k_proj_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.k_proj.bias", Qwen35Config<QWEN35_SIZE>::keys_size());
            full->v_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.v_proj.weight", Qwen35Config<QWEN35_SIZE>::values_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            if (tensors.contains(layer_prefix + ".self_attn.v_proj.bias")) full->v_proj_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.v_proj.bias", Qwen35Config<QWEN35_SIZE>::values_size());
            full->o_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.o_proj.weight", Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::queries_size());
            if (tensors.contains(layer_prefix + ".self_attn.o_proj.bias")) full->o_proj_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.o_proj.bias", Qwen35Config<QWEN35_SIZE>::hidden_size());
            full->q_norm.weights = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.q_norm.weight", Qwen35Config<QWEN35_SIZE>::head_size());
            full->k_norm.weights = tensors.template load_tensor<weight_t>(layer_prefix + ".self_attn.k_norm.weight", Qwen35Config<QWEN35_SIZE>::head_size());
            layer = full;
        } else {
            auto linear = std::make_shared<Qwen35LinearAttentionLayer<QWEN35_SIZE, weight_t, hidden_t, compute_t>>(i);
            if (tensors.contains(layer_prefix + ".linear_attn.in_proj_qkvz.weight")) {
                linear->in_proj_qkv_weight = tensors.template load_tensor_slice_rows<weight_t>(layer_prefix + ".linear_attn.in_proj_qkvz.weight", 0, Qwen35Config<QWEN35_SIZE>::linear_conv_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
                linear->in_proj_z_weight = tensors.template load_tensor_slice_rows<weight_t>(layer_prefix + ".linear_attn.in_proj_qkvz.weight", Qwen35Config<QWEN35_SIZE>::linear_conv_size(), Qwen35Config<QWEN35_SIZE>::linear_values_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            } else {
                linear->in_proj_qkv_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.in_proj_qkv.weight", Qwen35Config<QWEN35_SIZE>::linear_conv_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
                linear->in_proj_z_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.in_proj_z.weight", Qwen35Config<QWEN35_SIZE>::linear_values_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            }
            if (tensors.contains(layer_prefix + ".linear_attn.in_proj_ba.weight")) {
                linear->in_proj_b_weight = tensors.template load_tensor_slice_rows<weight_t>(layer_prefix + ".linear_attn.in_proj_ba.weight", 0, Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size());
                linear->in_proj_a_weight = tensors.template load_tensor_slice_rows<weight_t>(layer_prefix + ".linear_attn.in_proj_ba.weight", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            } else {
                linear->in_proj_b_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.in_proj_b.weight", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size());
                linear->in_proj_a_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.in_proj_a.weight", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads(), Qwen35Config<QWEN35_SIZE>::hidden_size());
            }
            linear->conv1d_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.conv1d.weight", Qwen35Config<QWEN35_SIZE>::linear_conv_size(), 1, Qwen35Config<QWEN35_SIZE>::linear_conv_kernel_dim());
            if (tensors.contains(layer_prefix + ".linear_attn.conv1d.bias")) linear->conv1d_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.conv1d.bias", Qwen35Config<QWEN35_SIZE>::linear_conv_size());
            linear->dt_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.dt_bias", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads());
            linear->A_log = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.A_log", Qwen35Config<QWEN35_SIZE>::linear_num_value_heads());
            linear->norm.weights = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.norm.weight", Qwen35Config<QWEN35_SIZE>::linear_value_head_dim());
            linear->out_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.out_proj.weight", Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::linear_values_size());
            if (tensors.contains(layer_prefix + ".linear_attn.out_proj.bias")) linear->out_proj_bias = tensors.template load_tensor<weight_t>(layer_prefix + ".linear_attn.out_proj.bias", Qwen35Config<QWEN35_SIZE>::hidden_size());
            layer = linear;
        }

        layer->input_layernorm.weights = tensors.template load_tensor<weight_t>(layer_prefix + ".input_layernorm.weight", Qwen35Config<QWEN35_SIZE>::hidden_size());
        layer->post_attention_layernorm.weights = tensors.template load_tensor<weight_t>(layer_prefix + ".post_attention_layernorm.weight", Qwen35Config<QWEN35_SIZE>::hidden_size());
        layer->up_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".mlp.up_proj.weight", Qwen35Config<QWEN35_SIZE>::intermediate_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
        layer->gate_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".mlp.gate_proj.weight", Qwen35Config<QWEN35_SIZE>::intermediate_size(), Qwen35Config<QWEN35_SIZE>::hidden_size());
        layer->down_proj_weight = tensors.template load_tensor<weight_t>(layer_prefix + ".mlp.down_proj.weight", Qwen35Config<QWEN35_SIZE>::hidden_size(), Qwen35Config<QWEN35_SIZE>::intermediate_size());
        model->layers.push_back(layer);
    }

    return model;
}

template void Qwen35Loader::validate_config<QWEN35_0_8B>(const nlohmann::json &cfg);
template void Qwen35Loader::validate_config<QWEN35_4B>(const nlohmann::json &cfg);
template void Qwen35Loader::validate_config<QWEN35_9B>(const nlohmann::json &cfg);
template std::shared_ptr<Qwen35Model<QWEN35_0_8B, float, float, float>> Qwen35Loader::load_qwen35<QWEN35_0_8B, float, float, float>(const std::string &model_dir);
template std::shared_ptr<Qwen35Model<QWEN35_4B, float, float, float>> Qwen35Loader::load_qwen35<QWEN35_4B, float, float, float>(const std::string &model_dir);
template std::shared_ptr<Qwen35Model<QWEN35_9B, float, float, float>> Qwen35Loader::load_qwen35<QWEN35_9B, float, float, float>(const std::string &model_dir);
