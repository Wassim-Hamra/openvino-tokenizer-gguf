#include <openvino/openvino.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/opsets/opset.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <regex>

const int MIN_CACHE_CAPACITY = 20000;
const float VOCAB_SIZE_CACHE_PROPORTION = 0.2;
const int MAX_LENGTH = 8192;

std::map<std::string, std::vector<std::string>> split_regex_mapping = {
    {"smollm", {
        "\\p{N}",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    }}
};

std::unordered_map<std::string, std::string> split_behaviour_mapping = {
    {"\\p{N}", "isolate"},
    {"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)", "isolate"}
};

std::vector<std::string> DEFAULT_BPE_SPLIT_RE = {
    R"([\\p{P}\\$\\+<=>\\^~\\|]+)",
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S))",
    R"(\\p{N}+)",
    R"([0-9][0-9][0-9])"
};


bool is_special(int token_type) {
    return token_type == 3 || token_type == 4;
}

std::vector<uint8_t> to_bytes(int number) {
    std::vector<uint8_t> bytes(4);
    bytes[0] = static_cast<uint8_t>(number & 0xFF);
    bytes[1] = static_cast<uint8_t>((number >> 8) & 0xFF);
    bytes[2] = static_cast<uint8_t>((number >> 16) & 0xFF);
    bytes[3] = static_cast<uint8_t>((number >> 24) & 0xFF);
    return bytes;
}


std::vector<uint8_t> string_to_bytes(const std::string& str) {
    return std::vector<uint8_t>(str.begin(), str.end());
}

std::vector<std::shared_ptr<ov::Output<ov::Node>>> create_unpacked_string(const std::vector<std::string>& strings) {
    std::vector<uint8_t> begins, ends, chars;
    int offset = 0;

    for (const auto& string : strings) {
        std::vector<uint8_t> byte_string(string.begin(), string.end());
        int length = byte_string.size();

        auto begin_bytes = to_bytes(offset);
        begins.insert(begins.end(), begin_bytes.begin(), begin_bytes.end());
        offset += length;

        auto end_bytes = to_bytes(offset);
        ends.insert(ends.end(), end_bytes.begin(), end_bytes.end());

        chars.insert(chars.end(), byte_string.begin(), byte_string.end());
    }

    auto begin_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{begins.size() / 4}, begins.data());
    auto end_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ends.size() / 4}, ends.data());
    auto chars_const = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{chars.size()}, chars.data());

    return {begin_const->output(0), end_const->output(0), chars_const->output(0)};
}

std::vector<std::shared_ptr<ov::Output<ov::Node>>> create_string_constant(const std::vector<std::string>& strings) {
    std::vector<uint8_t> data;
    for (const auto& str : strings) {
        std::vector<uint8_t> byte_string(str.begin(), str.end());
        data.insert(data.end(), byte_string.begin(), byte_string.end());
    }

    auto const_tensor = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{data.size()}, data.data());
    return {const_tensor->output(0)};
}

std::unordered_map<char, int> unicode_to_bytes() {
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) {
        bs.push_back(i);
    }
    for (int i = static_cast<int>('¡'); i <= static_cast<int>('¬'); ++i) {
        bs.push_back(i);
    }
    for (int i = static_cast<int>('®'); i <= static_cast<int>('ÿ'); ++i) {
        bs.push_back(i);
    }

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<char, int> mapping;
    for (size_t i = 0; i < cs.size(); ++i) {
        mapping[static_cast<char>(cs[i])] = bs[i];
    }

    return mapping;
}


std::vector<unsigned char> apply_unicode_to_bytes(const std::string& token) {
    std::unordered_map<char, int> bytes_encoder = unicode_to_bytes();
    std::vector<unsigned char> result;

    for (const char& c : token) {
        auto it = bytes_encoder.find(c);
        if (it != bytes_encoder.end()) {
            result.push_back(static_cast<unsigned char>(it->second));  // Character is found in the map
        } else {
            result.push_back(static_cast<unsigned char>(c));  // Character not found, use UTF-8 encoding (default behavior)
        }
    }

    return result;
}

namespace ov {
    class Output {};
    class Model {};
    std::shared_ptr<ov::Node> create_node(
    const std::string& node_name,
    const std::vector<ov::Output<ov::Node>>& inputs,
    const std::unordered_map<std::string, std::string>& params
    ) {
    std::cout << "Creating node: " << node_name << std::endl;

    if (node_name == "Add") {
        return std::make_shared<ov::op::v1::Add>(inputs[0], inputs[1]);
    } else if (node_name == "Multiply") {
        return std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[1]);
    } else if (node_name == "Constant") {
        int value = std::stoi(params.at("value"));
        return std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int>{value});
    } else {
        throw std::runtime_error("Unsupported node type: " + node_name);
    }
    }
std::vector<std::vector<unsigned char>> parse_bbpe_vocab(const std::vector<std::string>& tokens) {
    std::vector<std::vector<unsigned char>> result;
    for (const std::string& token : tokens) {
        result.push_back(apply_unicode_to_bytes(token));
    }
    return result;
}

std::vector<ov::Output<ov::Node>> parse_bbpe_config(
    const std::unordered_map<std::string, std::vector<std::string>>& config,
    std::vector<ov::Output<ov::Node>>& inputs) {

    std::vector<std::string> vocab = config.at("tokens");
    auto vocab_constants = create_string_constant(vocab);
    inputs.insert(inputs.end(), vocab_constants.begin(), vocab_constants.end());

    std::vector<std::vector<unsigned char>> left_merges, right_merges;
    for (const auto& merge : config.at("merges")) {
        std::vector<std::string> split_merge;
    std::istringstream iss(merge);
    std::string token;
    while (iss >> token) {
    split_merge.push_back(token);
    }
        left_merges.push_back( apply_unicode_to_bytes(split_merge[0]));
        right_merges.push_back(apply_unicode_to_bytes(split_merge[1]));
    }

    auto left_merge_constants = create_string_constant({left_merges.begin(), left_merges.end()});
    auto right_merge_constants = create_string_constant({right_merges.begin(), right_merges.end()});
    inputs.insert(inputs.end(), left_merge_constants.begin(), left_merge_constants.end());
    inputs.insert(inputs.end(), right_merge_constants.begin(), right_merge_constants.end());

    std::vector<std::string> special_tokens;
    std::vector<int> special_tokens_idx;
    for (size_t idx = 0; idx < vocab.size(); ++idx) {
        if (is_special(config.at("token_type")[idx])) {
            special_tokens.push_back(vocab[idx]);
            special_tokens_idx.push_back(idx);
        }
    }

    auto special_token_constants = create_string_constant(special_tokens);
    inputs.insert(inputs.end(), special_token_constants.begin(), special_token_constants.end());

    std::unordered_map<std::string, std::string> params = {
        {"unk_token", vocab[std::stoi(config.at("unknown_token_id")[0])]},
        {"fuse_unk", "True"},
        {"suffix_indicator", ""},
        {"end_suffix", ""},
        {"byte_fallback", "True"},
        {"cache_capacity", std::to_string(std::max(int(vocab.size() * 0.2), 20000))}
    };

    return create_node("BPETokenizer", inputs, params);
}

std::unordered_map<std::string, std::function<std::vector<ov::Output>(const std::unordered_map<std::string, std::vector<std::string>>&,
                                                               std::vector<ov::Output>&, ov::Core&)>> tokenizer_node_parser_mapping;


std::unordered_map<std::string, std::function<std::vector<std::vector<unsigned char>>(const std::vector<std::string>&)>> vocab_parser_mapping;

void initialize_mappings() {
    tokenizer_node_parser_mapping["gpt2"] = parse_bbpe_config;
    vocab_parser_mapping["gpt2"] = parse_bbpe_vocab;
}

std::vector<ov::Output> add_ragged_dimension(const std::vector<ov::Output>& input_node) {
    auto shape = ov::op::v0::ShapeOf::create(input_node[0]);

    auto batch_size = ov::op::v0::Gather::create(shape, ov::op::v0::Constant::create(ov::element::i32, {}, {0}), 0);

    auto ragged_begins = ov::op::v0::Range::create(ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                                                   batch_size, ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                                                   ov::element::i32);

    auto ragged_ends = ov::op::v0::Range::create(ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                                                 ov::op::v0::Add::create(batch_size, ov::op::v0::Constant::create(ov::element::i64, {}, {1})),
                                                 ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                                                 ov::element::i32);

    std::vector<ov::Output> result;
    result.push_back(ragged_begins);
    result.push_back(ragged_ends);
    result.insert(result.end(), input_node.begin(), input_node.end());

    return result;
}

std::tuple<ov::Model, ov::Model> create_tokenizer_from_config(const std::unordered_map<std::string, std::any>& tokenizer_config) {
    std::unordered_map<std::string, std::any> config = tokenizer_config;

    //mimicking tokenizer_config manip in python
    for (auto& pair : config) {
        auto& key = pair.first;
        size_t pos = key.find_last_of('.');
        if (pos != std::string::npos) {
            key = key.substr(pos + 1);
        }
    }

    ov::Core core;  //  Openvino core object

    const std::string ov_tokenizers_extension_path = "...";
    core.add_extension(ov_tokenizers_extension_path);

    std::vector<ov::Output<ov::Node>> tokenizer_inputs = {ov::op::v0::Parameter::create(ov::element::str, ov::PartialShape::dynamic())};

    auto outputs = opset::string_tensor_unpack(tokenizer_inputs[0]).outputs();
    outputs = add_ragged_dimension(outputs);

    std::vector<std::string> special_tokens;
    auto tokens = std::any_cast<std::vector<std::string>>(config["tokens"]);
    auto token_types = std::any_cast<std::vector<int>>(config["token_type"]);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special(token_types[i])) {
            special_tokens.push_back(tokens[i]);
        }
    }

    std::string special_tokens_re = std::accumulate(special_tokens.begin(), special_tokens.end(), std::string(),
        [](const std::string& a, const std::string& b) { return a.empty() ? b : a + "|" + b; });

    std::vector<ov::Output> special_tokens_re_outputs = create_string_constant(special_tokens_re);

    std::vector<ov::Output> outputs_with_split = outputs;
    outputs_with_split.insert(outputs_with_split.end(), special_tokens_re_outputs.begin(), special_tokens_re_outputs.end());

    std::shared_ptr<ov::Node> special_tokens_split_node = std::make_shared<SpecialTokensSplit>(outputs_with_split);


    model.add_node(special_tokens_split_node);

    std::vector<ov::Output> final_outputs = special_tokens_split_node->outputs();


    std::vector<ov::Output> outputs;

    auto split_res = split_regex_mapping.at(std::any_cast<std::string>(config["pre"]), DEFAULT_BPE_SPLIT_RE);
    for (const auto& split_re : split_res) {
    // 6 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8], skips[bool]
    auto split_constant = create_string_constant(split_re);
    outputs.push_back(split_constant);

    std::shared_ptr<ov::Node> regex_split_node = std::make_shared<ov::op::v0::SomeOp>(outputs);

    regex_split_node->set_parameter("behaviour", split_behaviour_mapping.at(split_re));
    regex_split_node->set_parameter("invert", false);
    regex_split_node->set_parameter("max_splits", -1);

    model.add_node(regex_split_node);

    outputs = regex_split_node->outputs();
}


    //tokenization step


    std::unordered_map<std::string, std::function<std::vector<ov::Output>(
    const std::unordered_map<std::string, std::vector<std::string>>&,
    std::vector<ov::Output>&, ov::NodeFactory&)>> tokenizer_node_parser_mapping;

    // posttokenization step

    auto max_length = opset::minimum(
        opset::subtract(outputs[1], outputs[0]),
        create_node("Constant", {}, {{"value", std::to_string(MAX_LENGTH)}})
    );

    outputs[0] = opset::subtract(outputs[1], max_length).output(0);

    // Left padding
    max_length = opset::reduce_max(
        opset::subtract(outputs[1], outputs[0]),
        create_node("Constant", {}, {{"value", "0"}})
    );

    outputs = create_node("RaggedToDense",
        outputs + max_length.outputs() + create_node("Constant", {}, {{"value", "0"}}).outputs(),
        {
            {"pad_right", "false"},
            {"pad_max_length", "false"}
        }).outputs();


    for (size_t idx = 0; idx < 2; ++idx) {
        outputs[idx] = opset::convert(outputs[idx], ov::element::i64).output(0);
        outputs[idx].tensor().add_names({idx == 0 ? "input_ids" : "attention_mask"});
    }

    ov::Model tokenizer_model(outputs, tokenizer_inputs, "tokenizer");


    // #detokenization model

    auto detokenizer_input = ov::op::v0::Parameter::create(ov::element::i32, ov::PartialShape({"?", "?"}));

    auto vocab = vocab_parser_mapping.at(std::any_cast<std::string>(config["model"]))(config["tokens"]);
    outputs = detokenizer_input.outputs() + create_string_constant(vocab);

    const auto& token_types = std::any_cast<std::vector<int>>(tokenizer_config.at("token_type"));

    std::vector<int32_t> special_token_ids;
    for (size_t idx = 0; idx < token_types.size(); ++idx) {
        if (is_special(token_types[idx])) {
            special_token_ids.push_back(static_cast<int32_t>(idx));
        }
    }

    auto special_token_ids_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32,
        ov::Shape{special_token_ids.size()},
        special_token_ids.data()
    );

    auto stop_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32,
        ov::Shape{},
        std::vector<int32_t>{std::numeric_limits<int32_t>::max()}
    );

    auto zero_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32,
        ov::Shape{},
        std::vector<int32_t>{0}
    );

    auto one_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32,
        ov::Shape{},
        std::vector<int32_t>{1}
    );
    auto sliced_skips = opset::slice(special_token_ids, zero_const, stop_const, one_const).outputs();
    outputs.push_back(sliced_skips);

    outputs = create_node("VocabDecoder", outputs).outputs();
    outputs = create_node("FuzeRagged", outputs).outputs();


    // TODO: add clean_up_tokenization_spaces step, utf-8 check

    outputs = opset::string_tensor_pack(outputs).outputs();

    ov::Model detokenizer_model(outputs, {detokenizer_input}, "detokenizer");
    detokenizer_model.output().tensor().add_names({"string_output"});

    return std::make_tuple(tokenizer_model, detokenizer_model);
}