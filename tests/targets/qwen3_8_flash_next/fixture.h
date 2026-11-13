#pragma once

#include "artifact/binder.h"
#include "artifact/reader.h"
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer_flash_next_fixture {

using Json = nlohmann::json;

constexpr std::uint64_t kAlignment   = 4096;
constexpr std::uint64_t kHeaderBytes = 32;

struct FixtureObject {
    std::string id;
    std::string kind;
    std::string format;
    std::string layout;
    std::vector<std::uint64_t> shape;
    std::string encoding;
    std::uint64_t bytes  = 0;
    std::uint64_t offset = 0;
};

struct FixturePart {
    std::string object;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct FixtureBinding {
    std::string name;
    bool whole = false;
    std::string object;
    std::vector<FixturePart> parts;
};

struct FixtureArtifact {
    std::vector<FixtureObject> objects;
    std::vector<FixtureBinding> bindings;
    std::vector<Json> uses;
    Json config = Json::object();
    Json component_resources = Json::object();
};

struct Axes {
    std::uint64_t matrices = 1;
    std::uint64_t rows     = 1;
    std::uint64_t columns  = 1;
};

inline Axes axes_of(const std::vector<std::uint64_t>& shape) {
    Axes axes;
    if (shape.size() >= 3) {
        axes.matrices = shape[0];
        axes.rows     = shape[1];
        axes.columns  = shape[2];
    } else if (shape.size() == 2) {
        axes.rows    = shape[0];
        axes.columns = shape[1];
    } else {
        axes.rows = shape[0];
    }
    return axes;
}

class FixtureBuilder {
public:
    FixtureArtifact artifact;

    std::string add_object(const std::string& format, const std::string& layout,
                           std::vector<std::uint64_t> shape) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "weight/%06zu", artifact.objects.size());
        FixtureObject object;
        object.id     = buffer;
        object.kind   = "tensor";
        object.format = format;
        object.layout = layout;
        object.shape  = std::move(shape);
        artifact.objects.push_back(std::move(object));
        return artifact.objects.back().id;
    }

    std::string add_auxiliary(const std::string& id, const std::string& format,
                              const std::string& layout, std::vector<std::uint64_t> shape) {
        FixtureObject object;
        object.id     = id;
        object.kind   = "tensor";
        object.format = format;
        object.layout = layout;
        object.shape  = std::move(shape);
        artifact.objects.push_back(std::move(object));
        return artifact.objects.back().id;
    }

    std::string add_resource(const std::string& id, std::uint64_t bytes) {
        FixtureObject object;
        object.id       = id;
        object.kind     = "resource";
        object.encoding = "raw_bytes_v1";
        object.bytes    = bytes;
        artifact.objects.push_back(std::move(object));
        return artifact.objects.back().id;
    }

    void add_resource_role(const std::string& component, const std::string& role,
                           const std::string& object_id) {
        artifact.component_resources[component][role] = object_id;
    }

    void bind_whole(const std::string& name, const std::string& object,
                    const std::string& policy = "A16Only", bool declares_use = true) {
        FixtureBinding binding;
        binding.name   = name;
        binding.whole  = true;
        binding.object = object;
        artifact.bindings.push_back(std::move(binding));
        if (declares_use) { add_use(name, policy); }
    }

    void bind_rows(const std::string& name, const std::string& object, std::uint64_t row_begin,
                   std::uint64_t row_count, const std::string& policy = "A16Only",
                   bool declares_use = true) {
        const Axes axes = axes_of(shape_of(object));
        FixtureBinding binding;
        binding.name        = name;
        binding.parts.push_back(FixturePart{object, row_begin * axes.columns,
                                            (row_begin + row_count) * axes.columns});
        artifact.bindings.push_back(std::move(binding));
        if (declares_use) { add_use(name, policy); }
    }

    void bind_matrix_rows(const std::string& name, const std::string& object,
                          std::uint64_t row_begin, std::uint64_t row_count,
                          const std::string& policy = "A16Only") {
        const Axes axes = axes_of(shape_of(object));
        FixtureBinding binding;
        binding.name = name;
        for (std::uint64_t matrix = 0; matrix < axes.matrices; ++matrix) {
            const auto begin = matrix * axes.rows * axes.columns + row_begin * axes.columns;
            binding.parts.push_back(FixturePart{object, begin, begin + row_count * axes.columns});
        }
        artifact.bindings.push_back(std::move(binding));
        add_use(name, policy);
    }

    void bind_whole_objects(const std::string& name, const std::vector<std::string>& objects,
                            bool declares_use = true) {
        FixtureBinding binding;
        binding.name = name;
        for (const auto& object : objects) {
            const Axes axes = axes_of(shape_of(object));
            binding.parts.push_back(FixturePart{object, 0, axes.matrices * axes.rows * axes.columns});
        }
        artifact.bindings.push_back(std::move(binding));
        if (declares_use) { add_use(name, "A16Only"); }
    }

    void bind_activation_divisor(const std::string& name, const std::string& auxiliary,
                                 std::uint64_t row, std::uint64_t column) {
        const Axes axes = axes_of(shape_of(auxiliary));
        const auto begin = row * axes.columns + column;
        for (auto& use : artifact.uses) {
            if (use.at("parameter").get<std::string>() != name) { continue; }
            use["activation_policy"] = "AllowA4";
            use["auxiliaries"]["activation_input_divisor"]["parts"].push_back(
                Json{{"object", auxiliary}, {"range", Json::array({begin, begin + 1})}});
        }
    }

private:
    const std::vector<std::uint64_t>& shape_of(const std::string& id) const {
        for (const auto& object : artifact.objects) {
            if (object.id == id) { return object.shape; }
        }
        throw std::runtime_error("the fixture has no object " + id);
    }

    void add_use(const std::string& name, const std::string& policy) {
        artifact.uses.push_back(Json{{"parameter", name},
                                     {"input", name + "_input"},
                                     {"activation_policy", policy}});
    }
};

inline Json text_config() {
    Json layer_types = Json::array();
    for (int layer = 0; layer < 48; ++layer) {
        layer_types.push_back(layer % 4 == 3 ? "indexed_attention" : "linear_attention");
    }
    return Json{
        {"hidden_size", 2560},
        {"vocab_size", 248320},
        {"num_hidden_layers", 48},
        {"max_position_embeddings", 262144},
        {"hc_count", 4},
        {"hc_lowrank", 320},
        {"num_attention_heads", 24},
        {"num_key_value_heads", 2},
        {"head_dim", 256},
        {"linear_num_key_heads", 16},
        {"linear_num_value_heads", 48},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
        {"indexer_n_heads", 4},
        {"indexer_kv_heads", 1},
        {"indexer_head_dim", 128},
        {"indexer_compress_ratio", 4},
        {"indexer_budget", 2048},
        {"num_experts", 512},
        {"num_experts_per_tok", 10},
        {"moe_intermediate_size", 640},
        {"shared_expert_intermediate_size", 640},
        {"ple_embed_dim", 2560},
        {"ngram_size", 3},
        {"heads_per_ngram", 8},
        {"split_ngram_parts", 128},
        {"ple_conv_kernel_size", 4},
        {"ple_conv_dilation", 3},
        {"architectures", Json::array({"Qwen4ExpForCausalLM"})},
        {"model_type", "qwen4_exp_text"},
        {"hidden_act", "silu"},
        {"output_gate_type", "sigmoid"},
        {"tie_word_embeddings", false},
        {"attention_bias", false},
        {"mamba_ssm_dtype", "float32"},
        {"rms_norm_eps", 1e-06},
        {"norm_topk_prob", true},
        {"layer_types", layer_types},
        {"ple_layer_ids", Json::array({1})},
        {"eos_token_id", 248044},
        {"image_token_id", 248056},
        {"video_token_id", 248057},
        {"vision_start_token_id", 248053},
        {"vision_end_token_id", 248054},
        {"ple_shard_shape", Json::array({2500012, 160})},
        {"rope_parameters",
         Json{{"mrope_interleaved", true},
              {"mrope_section", Json::array({11, 11, 10})},
              {"partial_rotary_factor", 0.25},
              {"rope_theta", 10000000},
              {"rope_type", "default"}}},
    };
}

inline FixtureArtifact build_flash_next_fixture() {
    FixtureBuilder builder;
    builder.artifact.config = text_config();

    builder.bind_whole("text/token_embedding",
                       builder.add_object("bf16", "contiguous_le_v1", {248320, 2560}),
                       "A16Only", false);
    builder.bind_whole("text/output_head",
                       builder.add_object("bf16", "contiguous_le_v1", {248320, 2560}));
    builder.bind_whole("text/final_hc/norm",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}),
                       "A16Only", false);
    builder.bind_whole("text/final_hc/down",
                       builder.add_object("bf16", "contiguous_le_v1", {320, 10240}));
    builder.bind_whole("text/final_hc/up",
                       builder.add_object("bf16", "contiguous_le_v1", {10240, 320}));

    const auto auxiliary =
        builder.add_auxiliary("auxiliary/shared/expert_activation_divisors", "fp32",
                              "contiguous_le_v1", {48, 2});

    for (int layer = 0; layer < 48; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";

        const auto attention_hc_norm =
            builder.add_object("bf16", "contiguous_le_v1", {10240});
        builder.bind_whole(prefix + "attention_hc/norm", attention_hc_norm, "A16Only", false);
        const auto attention_hc_down = builder.add_object("bf16", "contiguous_le_v1", {324, 10240});
        builder.bind_rows(prefix + "attention_hc/down", attention_hc_down, 0, 320);
        builder.bind_rows(prefix + "attention_hc/inject", attention_hc_down, 320, 4);
        const auto attention_hc_up = builder.add_object("bf16", "contiguous_le_v1", {10240, 320});
        builder.bind_whole(prefix + "attention_hc/up", attention_hc_up);

        const auto moe_hc_norm = builder.add_object("bf16", "contiguous_le_v1", {10240});
        builder.bind_whole(prefix + "moe_hc/norm", moe_hc_norm, "A16Only", false);
        const auto moe_hc_down = builder.add_object("bf16", "contiguous_le_v1", {324, 10240});
        builder.bind_rows(prefix + "moe_hc/down", moe_hc_down, 0, 320);
        builder.bind_rows(prefix + "moe_hc/inject", moe_hc_down, 320, 4);
        const auto moe_hc_up = builder.add_object("bf16", "contiguous_le_v1", {10240, 320});
        builder.bind_whole(prefix + "moe_hc/up", moe_hc_up);

        if (layer % 4 == 3) {
            const auto query_gate = builder.add_object("q8_g32_fp16", "row_split_k128_v1",
                                                       {12288, 2560});
            builder.bind_rows(prefix + "attention/query", query_gate, 0, 6144);
            builder.bind_rows(prefix + "attention/output_gate", query_gate, 6144, 6144);
            const auto key_value = builder.add_object("bf16", "contiguous_le_v1", {1664, 2560});
            builder.bind_rows(prefix + "attention/key", key_value, 0, 512);
            builder.bind_rows(prefix + "attention/value", key_value, 512, 512);
            builder.bind_rows(prefix + "attention/index_query", key_value, 1024, 512);
            builder.bind_rows(prefix + "attention/index_key", key_value, 1536, 128);
            builder.bind_whole(prefix + "attention/output",
                               builder.add_object("q8_g32_fp16", "row_split_k128_v1",
                                                  {2560, 6144}));
            builder.bind_whole(prefix + "attention/query_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {256}),
                               "A16Only", false);
            builder.bind_whole(prefix + "attention/key_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {256}),
                               "A16Only", false);
            builder.bind_whole(prefix + "attention/index_query_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}),
                               "A16Only", false);
            builder.bind_whole(prefix + "attention/index_key_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}),
                               "A16Only", false);
        } else {
            const auto query_value =
                builder.add_object("q8_g32_fp16", "row_split_k128_v1", {16384, 2560});
            builder.bind_rows(prefix + "gdn/query", query_value, 0, 2048);
            builder.bind_rows(prefix + "gdn/key", query_value, 2048, 2048);
            builder.bind_rows(prefix + "gdn/value", query_value, 4096, 6144);
            builder.bind_rows(prefix + "gdn/z", query_value, 10240, 6144);
            const auto control = builder.add_object("bf16", "contiguous_le_v1", {96, 2560});
            builder.bind_rows(prefix + "gdn/a_projection", control, 0, 48);
            builder.bind_rows(prefix + "gdn/b_projection", control, 48, 48);
            builder.bind_whole(prefix + "gdn/a_log",
                               builder.add_object("fp32", "contiguous_le_v1", {48}),
                               "A16Only", false);
            builder.bind_whole(prefix + "gdn/dt_bias",
                               builder.add_object("fp32", "contiguous_le_v1", {48}),
                               "A16Only", false);
            builder.bind_whole(prefix + "gdn/convolution",
                               builder.add_object("bf16", "contiguous_le_v1", {4, 10240}),
                               "A16Only", false);
            builder.bind_whole(prefix + "gdn/norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}),
                               "A16Only", false);
            builder.bind_whole(prefix + "gdn/output",
                               builder.add_object("q8_g32_fp16", "row_split_k128_v1",
                                                  {2560, 6144}));
        }

        const auto control = builder.add_object("bf16", "contiguous_le_v1", {1793, 2560});
        builder.bind_rows(prefix + "moe/router", control, 0, 512);
        builder.bind_rows(prefix + "moe/shared_score", control, 512, 1);
        builder.bind_rows(prefix + "moe/shared/gate", control, 513, 640);
        builder.bind_rows(prefix + "moe/shared/up", control, 1153, 640);
        builder.bind_whole(prefix + "moe/shared/down",
                           builder.add_object("bf16", "contiguous_le_v1", {2560, 640}));
        const auto bank = builder.add_object("nvfp4", "batched_block_scale_k16_m128x4_v1",
                                             {512, 1280, 2560});
        builder.bind_matrix_rows(prefix + "moe/experts/gate", bank, 0, 640, "AllowA4");
        builder.bind_matrix_rows(prefix + "moe/experts/up", bank, 640, 640, "AllowA4");
        const auto down_bank = builder.add_object("nvfp4", "batched_block_scale_k16_m128x4_v1",
                                                  {512, 2560, 640});
        builder.bind_matrix_rows(prefix + "moe/experts/down", down_bank, 0, 2560, "AllowA4");
        builder.bind_activation_divisor(prefix + "moe/experts/gate", auxiliary, layer, 0);
        builder.bind_activation_divisor(prefix + "moe/experts/up", auxiliary, layer, 0);
        builder.bind_activation_divisor(prefix + "moe/experts/down", auxiliary, layer, 1);
    }

    std::vector<std::string> shards;
    for (int shard = 0; shard < 128; ++shard) {
        shards.push_back(builder.add_object("nf4_g16_e4m3fn", "row_paged_g16_4k_v1",
                                           {2500012, 160}));
    }
    builder.bind_whole_objects("text/layers/1/ple/table", shards, false);
    const auto ple_key_value = builder.add_object("bf16", "contiguous_le_v1", {12800, 2560});
    builder.bind_rows("text/layers/1/ple/key", ple_key_value, 0, 10240);
    builder.bind_rows("text/layers/1/ple/value", ple_key_value, 10240, 2560);
    builder.bind_whole("text/layers/1/ple/norm_key",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}),
                       "A16Only", false);
    builder.bind_whole("text/layers/1/ple/norm_query",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}),
                       "A16Only", false);
    builder.bind_whole("text/layers/1/ple/norm_conv",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}),
                       "A16Only", false);
    builder.bind_whole("text/layers/1/ple/convolution",
                       builder.add_object("bf16", "contiguous_le_v1", {4, 10240}),
                       "A16Only", false);

    const auto mtp = builder.add_object("bf16", "contiguous_le_v1", {248320, 2560});
    builder.bind_whole("mtp/token_embedding", mtp);
    builder.bind_whole("mtp/output_head", mtp);
    const auto vision = builder.add_object("bf16", "contiguous_le_v1", {2560, 2560});
    builder.bind_whole("vision/patch_embedding", vision);
    const auto proposal = builder.add_object("q4_g64_fp16", "row_split_k128_v1", {248320, 2560});
    builder.bind_whole("proposal/head", proposal);

    const auto text_tokenizer = builder.add_resource("resource/text/tokenizer.json", 1);
    const auto text_tokenizer_config = builder.add_resource("resource/text/tokenizer_config.json", 1);
    const auto text_chat_template = builder.add_resource("resource/text/chat_template.jinja", 1);
    const auto text_generation_config = builder.add_resource("resource/text/generation_config.json", 1);
    const auto vision_preprocessor = builder.add_resource("resource/vision/preprocessor_config.json", 1);
    const auto vision_video_preprocessor =
        builder.add_resource("resource/vision/video_preprocessor_config.json", 1);

    builder.add_resource_role("text", "tokenizer.json", text_tokenizer);
    builder.add_resource_role("text", "tokenizer_config.json", text_tokenizer_config);
    builder.add_resource_role("text", "chat_template.jinja", text_chat_template);
    builder.add_resource_role("text", "generation_config.json", text_generation_config);
    builder.add_resource_role("vision", "preprocessor_config.json", vision_preprocessor);
    builder.add_resource_role("vision", "video_preprocessor_config.json", vision_video_preprocessor);

    return std::move(builder.artifact);
}

inline bool has_frontend_resources(const ninfer::artifact::Binder& binder) {
    const auto& resources = binder.directory().component_resources;
    auto has = [&](const std::string& component, const std::string& role) {
        const auto component_it = resources.find(component);
        if (component_it == resources.end()) { return false; }
        return component_it->second.find(role) != component_it->second.end();
    };

    return has("text", "tokenizer.json") &&
           has("text", "tokenizer_config.json") &&
           has("text", "chat_template.jinja") &&
           has("text", "generation_config.json") &&
           has("vision", "preprocessor_config.json") &&
           has("vision", "video_preprocessor_config.json");
}

inline std::filesystem::path fixture_path() {
    return std::filesystem::temp_directory_path() /
           ("ninfer_flash_next_fixture_" + std::to_string(static_cast<long long>(::getpid())) +
            ".ninfer");
}

class TemporaryArtifact {
public:
    explicit TemporaryArtifact(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    TemporaryArtifact(const TemporaryArtifact&)            = delete;
    TemporaryArtifact& operator=(const TemporaryArtifact&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

inline std::filesystem::path write_flash_next_fixture(FixtureArtifact& artifact) {
    for (auto& object : artifact.objects) {
        if (object.kind == "tensor") {
            const auto format = ninfer::artifact::parse_v3_format(object.format);
            const auto layout = ninfer::artifact::parse_v3_layout(object.layout);
            object.bytes = ninfer::artifact::tensor_encoded_size(layout, format, object.shape);
        }
    }

    std::uint64_t cursor = 0;
    for (auto& object : artifact.objects) {
        cursor = (cursor + kAlignment - 1) / kAlignment * kAlignment;
        object.offset = cursor;
        cursor += object.bytes;
    }

    Json objects = Json::array();
    for (const auto& object : artifact.objects) {
        objects.push_back(Json{{"id", object.id},
                               {"kind", object.kind},
                               {"encoding", object.encoding},
                               {"format", object.format},
                               {"layout", object.layout},
                               {"shape", object.shape},
                               {"bytes", object.bytes},
                               {"offset", object.offset}});
    }
    Json bindings = Json::object();
    for (const auto& binding : artifact.bindings) {
        if (binding.whole) {
            bindings[binding.name] = Json{{"object", binding.object}};
            continue;
        }
        Json parts = Json::array();
        for (const auto& part : binding.parts) {
            parts.push_back(Json{{"object", part.object},
                                 {"range", Json::array({part.begin, part.end})}});
        }
        bindings[binding.name] = Json{{"parts", parts}};
    }
    Json components{
        {"text", Json{{"config", artifact.config}, {"resources", artifact.component_resources["text"]}}},
        {"vision", Json{{"resources", artifact.component_resources["vision"]}}},
    };
    Json directory{
        {"metadata", Json{{"name", "qwen3.8-flash-next"}}},
        {"files", Json::array({Json{{"path", nullptr}, {"payload_bytes", cursor}}})},
        {"objects", objects},
        {"bindings", bindings},
        {"uses", artifact.uses},
        {"components", components},
    };

    const std::string text = directory.dump();
    const auto payload_start = (kHeaderBytes + text.size() + kAlignment - 1) / kAlignment * kAlignment;
    const auto path          = fixture_path();

    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { throw std::runtime_error("cannot create the fixture artifact"); }
    std::vector<std::byte> header(kHeaderBytes + text.size());
    const std::array<char, 8> magic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    std::memcpy(header.data(), magic.data(), magic.size());
    const auto json_bytes = static_cast<std::uint64_t>(text.size());
    for (unsigned index = 0; index < 8; ++index) {
        header[8 + index] = std::byte(static_cast<std::uint8_t>((json_bytes >> (index * 8)) & 0xff));
    }
    std::memcpy(header.data() + kHeaderBytes, text.data(), text.size());
    if (::pwrite(fd, header.data(), header.size(), 0) !=
        static_cast<ssize_t>(header.size())) {
        ::close(fd);
        throw std::runtime_error("cannot write the fixture directory");
    }
    if (::ftruncate(fd, static_cast<off_t>(payload_start + cursor)) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot size the fixture artifact");
    }
    ::close(fd);
    return path;
}

} // namespace ninfer_flash_next_fixture
