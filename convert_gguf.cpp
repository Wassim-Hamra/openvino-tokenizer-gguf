#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <tbb/tbb.h>
#include "openvino/openvino.hpp"
#include "convert_tokenizer.hpp"
#include <nlohmann/json.hpp>

using namespace std;
using namespace openvino;
const std::string OV_XML_FILE_NAME = "openvino_model.xml";
const std::string OV_TOKENIZER_FILE_NAME = "openvino_tokenizer.xml";
const std::string OV_DETOKENIZER_FILE_NAME = "openvino_detokenizer.xml";


// save_pretrained_config in Python equiv
using json = nlohmann::json;
void save_pretrained_config(const std::string& model_id, const std::string& ov_model_path) {
    json config;

    // some basic config that might be required for OpenVINO
    config["model_id"] = model_id;
    config["batch_size"] = 1;
    config["input_name"] = "input_ids";
    config["output_name"] = "logits";
    config["max_seq_length"] = 512;
    config["num_layers"] = 12;

    std::filesystem::create_directories(ov_model_path);
    std::string config_file_path = ov_model_path + "/config.json";

    std::ofstream config_file(config_file_path);
    if (config_file.is_open()) {
        config_file << config.dump(4);
        config_file.close();
        std::cout << "Configuration saved at: " << config_file_path << std::endl;
    } else {
        std::cerr << "Error saving configuration file." << std::endl;
    }
}


// Function to save tokenizer vocabulary and configuration to a file - HF AutoTokenizer equiv
void save_original_tokenizer(const std::string& orig_model_path, const std::string& ov_model_path) {
    // Open a file to write the tokenizer configuration (vocabulary, special tokens, etc.)
    std::ofstream vocab_file(ov_model_path + "/vocab.txt");
    if (!vocab_file.is_open()) {
        std::cerr << "Error opening file for writing tokenizer vocab." << std::endl;
        return;
    }

    // Ex: hardcoded tokenizer vocab
    std::vector<std::string> vocab = {"<pad>", "<s>", "</s>", "hello", "world"};
    for (const auto& token : vocab) {
        vocab_file << token << std::endl;
    }
    vocab_file.close();

    // Ex: Save special tokens
    std::ofstream special_tokens_file(ov_model_path + "/special_tokens.txt");
    if (!special_tokens_file.is_open()) {
        std::cerr << "Error opening file for writing special tokens." << std::endl;
        return;
    }

    std::unordered_map<std::string, int> special_tokens = {
        {"<pad>", 0},
        {"<s>", 1},
        {"</s>", 2}
    };

    for (const auto& token : special_tokens) {
        special_tokens_file << token.first << " " << token.second << std::endl;
    }
    special_tokens_file.close();

    std::cout << "Tokenizer configuration saved to " << ov_model_path << std::endl;
}

// openvino.runtime equiv
void serialize(const Model& model, const std::string& file_path) {
    try {
        model.serialize(file_path);
        cout << "Model serialized to " << file_path << endl;
    } catch (const std::exception& e) {
        cerr << "Error during serialization: " << e.what() << endl;
    }
}


void show_model(const Model& model) {
    cout << "Inputs of the model:" << endl;
    for (size_t port = 0; port < model.inputs().size(); ++port) {
        cout << "\t[" << port << "] " << model.inputs()[port].get_node()->get_friendly_name() << endl;
    }

    cout << "Outputs of the model:" << endl;
    for (size_t port = 0; port < model.outputs().size(); ++port) {
        cout << "\t[" << port << "] " << model.outputs()[port].get_node()->get_friendly_name() << endl;
    }
}



Model create_model(const ConfigMap& configs, const WeightsMap& consts) {
    cout << "Start generating OpenVINO model..." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    auto input_ids = opset::parameter({-1, -1}, Type::i64, "input_ids");
    auto attention_mask = opset::parameter({-1, -1}, Type::i64, "attention_mask");
    auto position_ids = opset::parameter({-1, -1}, Type::i64, "position_ids");
    auto beam_idx = opset::parameter({-1}, Type::i32, "beam_idx");

    auto [inputs_embeds, embeddings] = make_embedding("model.embed_tokens", input_ids, consts, configs.at("qtype"));
    auto hidden_states = inputs_embeds;

    auto rope_const = init_rope(configs.at("head_size"), configs.at("max_position_embeddings"), configs.at("rope_freq_base"));

    auto input_shape = opset::shape_of(input_ids);
    auto batch_size = opset::gather(input_shape, opset::constant({0}, Type::i64), opset::constant({0}, Type::i64));
    auto hidden_dim = opset::constant({3}, Type::i64);

    vector<shared_ptr<opset::Node>> sinks;
    shared_ptr<opset::Node> causal_mask = nullptr;
    shared_ptr<opset::Node> cos_sin_cached = nullptr;
    shared_ptr<opset::Node> output_shape = nullptr;

    int layer_num = stoi(configs.at("layer_num"));

    // Loop through each layer
    for (int i = 0; i < layer_num; ++i) {
        auto [layer_hidden_states, layer_sinks, new_causal_mask, new_cos_sin_cached, new_output_shape] = layer(configs, consts, i, hidden_states, attention_mask, causal_mask, position_ids, rope_const, beam_idx, batch_size, hidden_dim, cos_sin_cached, output_shape);
        hidden_states = layer_hidden_states;
        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
        causal_mask = new_causal_mask;
        cos_sin_cached = new_cos_sin_cached;
        output_shape = new_output_shape;
    }

    auto final_layernorm = make_rms_norm("model.norm", hidden_states, consts, configs.at("rms_norm_eps"));

    auto embed_out = make_lm_head("lm_head", final_layernorm, consts, embeddings, configs.at("qtype"));

    auto logits = opset::result(embed_out, "logits");
    logits->set_friendly_name("logits");

    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Model generation done, took " << duration << " seconds." << endl;

    Model model({logits}, sinks, {input_ids, attention_mask, position_ids, beam_idx});
    model.outputs()[0].get_tensor().set_names({"logits"});

    model.set_rt_info("f16", {"runtime_options", "KV_CACHE_PRECISION"});
    model.set_rt_info("8.0", {"runtime_options", "ACTIVATIONS_SCALE_FACTOR"});

    return model;
}





struct GGUFModel {
    std::unordered_map<std::string, std::vector<float>> weights;
    std::unordered_map<std::string, std::string> metadata;
};

GGUFModel load_gguf(const std::string& model_path);

std::string get_quantization_type(int file_type) {
    switch (file_type) {
        case 0: return "FP32";
        case 1: return "FP16";
        case 2: return "INT8";
        default: return "Unknown";
    }
}

// Data Structures
using ConfigMap = std::unordered_map<std::string, std::string>;
using WeightsMap = std::unordered_map<std::string, std::vector<float>>;
using TokenizerConfig = std::unordered_map<std::string, std::string>;

std::tuple<ConfigMap, WeightsMap, TokenizerConfig> load_gguf_model(const std::string& model_path) {
    std::cout << "Extracting from GGUF model '" << model_path << "'..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    GGUFModel gguf_data = load_gguf(model_path);
    auto& weights = gguf_data.weights;
    auto& metadata = gguf_data.metadata;

    std::cout << "Metadata Keys:" << std::endl;
    for (const auto& pair : metadata) {
        std::cout << pair.first << std::endl;
    }

    std::string model_id;
    try {
        std::string url = metadata["general.source.url"];
        size_t last_slash = url.find_last_of('/');
        size_t second_last_slash = url.find_last_of('/', last_slash - 1);
        model_id = url.substr(second_last_slash + 1);
    } catch (...) {
        std::cerr << "Cannot get model_id to retrieve config.json and tokenizer" << std::endl;
        model_id = "";
    }

    ConfigMap config;
    try {
        config["layer_num"] = metadata.at("llama.block_count");
        config["head_num"] = metadata.at("llama.attention.head_count");
        config["head_size"] = std::to_string(std::stoi(metadata.at("llama.embedding_length")) / std::stoi(metadata.at("llama.attention.head_count")));
        config["head_num_kv"] = metadata.count("llama.attention.head_count_kv") ? metadata.at("llama.attention.head_count_kv") : metadata.at("llama.attention.head_count");
        config["hidden_size"] = metadata.at("llama.embedding_length");
        config["max_position_embeddings"] = metadata.count("llama.context_length") ? metadata.at("llama.context_length") : "2048";
        config["rotary_dims"] = metadata.at("llama.rope.dimension_count");
        config["rms_norm_eps"] = metadata.at("llama.attention.layer_norm_rms_epsilon");
        config["rope_freq_base"] = metadata.count("llama.rope.freq_base") ? metadata.at("llama.rope.freq_base") : "10000";
        config["qtype"] = get_quantization_type(std::stoi(metadata.at("general.file_type")));
        config["model_id"] = model_id;
    } catch (const std::exception& e) {
        std::cerr << "Error reading metadata: " << e.what() << std::endl;
    }

    std::cout << "Config:\n";
    for (const auto& pair : config) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    WeightsMap consts;
    consts["model.embed_tokens.weight"] = weights["token_embd.weight"];
    consts["model.norm.weight"] = weights["output_norm.weight"];

    if (weights.count("output.weight")) {
        consts["lm_head.weight"] = weights["output.weight"];
    }

    if (weights.count("token_embd.scales")) {
        consts["model.embed_tokens.scales"] = weights["token_embd.scales"];
        consts["model.embed_tokens.biases"] = weights["token_embd.biases"];
    }

    if (weights.count("output.scales")) {
        consts["lm_head.scales"] = weights["output.scales"];
        consts["lm_head.biases"] = weights["output.biases"];
    }

    std::vector<WeightsMap> layers;
    int layer_num = std::stoi(config["layer_num"]);

    for (int i = 0; i < layer_num; i++) {
        WeightsMap layer_weights;
        std::string prefix = "blk." + std::to_string(i) + ".";

        layer_weights["model.layers.input_layernorm.weight"] = weights[prefix + "attn_norm.weight"];
        layer_weights["model.layers.post_attention_layernorm.weight"] = weights[prefix + "ffn_norm.weight"];

        layer_weights["model.layers.self_attn.q_proj.weight"] = weights[prefix + "attn_q.weight"];
        layer_weights["model.layers.self_attn.k_proj.weight"] = weights[prefix + "attn_k.weight"];
        layer_weights["model.layers.self_attn.v_proj.weight"] = weights[prefix + "attn_v.weight"];
        layer_weights["model.layers.self_attn.o_proj.weight"] = weights[prefix + "attn_output.weight"];

        layer_weights["model.layers.mlp.gate_proj.weight"] = weights[prefix + "ffn_gate.weight"];
        layer_weights["model.layers.mlp.up_proj.weight"] = weights[prefix + "ffn_up.weight"];
        layer_weights["model.layers.mlp.down_proj.weight"] = weights[prefix + "ffn_down.weight"];

        if (config["qtype"] != "FP16") {
            layer_weights["model.layers.self_attn.q_proj.scales"] = weights[prefix + "attn_q.scales"];
            layer_weights["model.layers.self_attn.k_proj.scales"] = weights[prefix + "attn_k.scales"];
            layer_weights["model.layers.self_attn.v_proj.scales"] = weights[prefix + "attn_v.scales"];
            layer_weights["model.layers.self_attn.o_proj.scales"] = weights[prefix + "attn_output.scales"];

            layer_weights["model.layers.mlp.gate_proj.scales"] = weights[prefix + "ffn_gate.scales"];
            layer_weights["model.layers.mlp.up_proj.scales"] = weights[prefix + "ffn_up.scales"];
            layer_weights["model.layers.mlp.down_proj.scales"] = weights[prefix + "ffn_down.scales"];
        }

        layers.push_back(layer_weights);
    }

    consts["layers"] = layers;

    TokenizerConfig tokenizer_config;
    for (const auto& [key, value] : metadata) {
        if (key.find("tokenizer") == 0) {
            tokenizer_config[key.substr(9)] = value;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Extraction complete in " << duration << " seconds.\n";

    return {config, consts, tokenizer_config};
}




int main(int argc, char* argv[]) {
    std::string org_model_path = (argc > 1) ? argv[1] : "Model ID (can be a Hugging Face Hub ID, or a local directory)";
    std::string ov_model_path = (argc > 2) ? argv[2] : "./gen/llama-2-7b-chat/";
    std::string model_id = (argc > 3) ? argv[3] : "";

    std::filesystem::create_directories(ov_model_path);
    auto [config, weights, tokenizer_config] = load_gguf_model(org_model_path);

    Model openvino_model = create_model(config, weights);
    show_model(openvino_model);

    std::cout << "Serializing OpenVINO model to '" << ov_model_path << "'..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    serialize(config, ov_model_path + "/" + OV_XML_FILE_NAME);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Serialization done in "
              << std::chrono::duration<double>(end_time - start_time).count() << " seconds.\n";

    std::cout << "Creating tokenizer and detokenizer..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::string tokenizer, detokenizer;
    create_tokenizer_from_config(tokenizer_config, tokenizer, detokenizer);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Tokenizer and detokenizer created in "
              << std::chrono::duration<double>(end_time - start_time).count() << " seconds.\n";

    std::cout << "Tokenizer:" << std::endl;
    show_model(tokenizer);
    std::cout << "Detokenizer:" << std::endl;
    show_model(detokenizer);

    std::cout << "Serializing OpenVINO tokenizer and detokenizer to '" << ov_model_path << "'..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    serialize(tokenizer, ov_model_path + "/" + OV_TOKENIZER_FILE_NAME);
    serialize(detokenizer, ov_model_path + "/" + OV_DETOKENIZER_FILE_NAME);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Serialization done in "
              << std::chrono::duration<double>(end_time - start_time).count() << " seconds.\n";

    if (!model_id.empty()) {
        std::cout << "Saving original tokenizer to '" << ov_model_path << "'..." << std::endl;
        save_original_tokenizer(model_id, ov_model_path);
        save_pretrained_config(model_id, ov_model_path);
    } else {
        std::cout << "[WARNING]: config.json was not saved because model_id was not found or provided as an option.\n";
    }

    return 0;
}
