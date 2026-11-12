#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Binder;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::StorageLayout;
using Json = nlohmann::json;

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint64_t kAlignment   = 4096;
constexpr std::uint64_t kHeaderBytes = 32;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct FixtureObject {
    std::string id;
    std::string format;
    std::string layout;
    std::vector<std::uint64_t> shape;
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
};

struct Axes {
    std::uint64_t matrices = 1;
    std::uint64_t rows     = 1;
    std::uint64_t columns  = 1;
};

Axes axes_of(const std::vector<std::uint64_t>& shape) {
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
        object.format = format;
        object.layout = layout;
        object.shape  = std::move(shape);
        artifact.objects.push_back(std::move(object));
        return artifact.objects.back().id;
    }

    void bind_whole(const std::string& name, const std::string& object,
                    const std::string& policy = "A16Only") {
        FixtureBinding binding;
        binding.name   = name;
        binding.whole  = true;
        binding.object = object;
        artifact.bindings.push_back(std::move(binding));
        add_use(name, policy);
    }

    void bind_rows(const std::string& name, const std::string& object, std::uint64_t row_begin,
                   std::uint64_t row_count, const std::string& policy = "A16Only") {
        const Axes axes = axes_of(shape_of(object));
        FixtureBinding binding;
        binding.name        = name;
        binding.parts.push_back(FixturePart{object, row_begin * axes.columns,
                                             (row_begin + row_count) * axes.columns});
        artifact.bindings.push_back(std::move(binding));
        add_use(name, policy);
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

    void bind_whole_objects(const std::string& name, const std::vector<std::string>& objects) {
        FixtureBinding binding;
        binding.name = name;
        for (const auto& object : objects) {
            const Axes axes = axes_of(shape_of(object));
            binding.parts.push_back(FixturePart{object, 0, axes.matrices * axes.rows * axes.columns});
        }
        artifact.bindings.push_back(std::move(binding));
        add_use(name, "A16Only");
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

Json text_config() {
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

FixtureArtifact build_fixture() {
    FixtureBuilder builder;
    builder.artifact.config = text_config();

    builder.bind_whole("text/token_embedding",
                       builder.add_object("bf16", "contiguous_le_v1", {248320, 2560}));
    builder.bind_whole("text/output_head",
                       builder.add_object("bf16", "contiguous_le_v1", {248320, 2560}));
    builder.bind_whole("text/final_hc/norm",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}));
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
        builder.bind_whole(prefix + "attention_hc/norm", attention_hc_norm);
        const auto attention_hc_down = builder.add_object("bf16", "contiguous_le_v1", {324, 10240});
        builder.bind_rows(prefix + "attention_hc/down", attention_hc_down, 0, 320);
        builder.bind_rows(prefix + "attention_hc/inject", attention_hc_down, 320, 4);
        const auto attention_hc_up = builder.add_object("bf16", "contiguous_le_v1", {10240, 320});
        builder.bind_whole(prefix + "attention_hc/up", attention_hc_up);

        const auto moe_hc_norm = builder.add_object("bf16", "contiguous_le_v1", {10240});
        builder.bind_whole(prefix + "moe_hc/norm", moe_hc_norm);
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
                               builder.add_object("bf16", "contiguous_le_v1", {256}));
            builder.bind_whole(prefix + "attention/key_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {256}));
            builder.bind_whole(prefix + "attention/index_query_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}));
            builder.bind_whole(prefix + "attention/index_key_norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}));
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
                               builder.add_object("fp32", "contiguous_le_v1", {48}));
            builder.bind_whole(prefix + "gdn/dt_bias",
                               builder.add_object("fp32", "contiguous_le_v1", {48}));
            builder.bind_whole(prefix + "gdn/convolution",
                               builder.add_object("bf16", "contiguous_le_v1", {4, 10240}));
            builder.bind_whole(prefix + "gdn/norm",
                               builder.add_object("bf16", "contiguous_le_v1", {128}));
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
    builder.bind_whole_objects("text/layers/1/ple/table", shards);
    const auto ple_key_value = builder.add_object("bf16", "contiguous_le_v1", {12800, 2560});
    builder.bind_rows("text/layers/1/ple/key", ple_key_value, 0, 10240);
    builder.bind_rows("text/layers/1/ple/value", ple_key_value, 10240, 2560);
    builder.bind_whole("text/layers/1/ple/norm_key",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}));
    builder.bind_whole("text/layers/1/ple/norm_query",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}));
    builder.bind_whole("text/layers/1/ple/norm_conv",
                       builder.add_object("bf16", "contiguous_le_v1", {10240}));
    builder.bind_whole("text/layers/1/ple/convolution",
                       builder.add_object("bf16", "contiguous_le_v1", {4, 10240}));

    const auto mtp = builder.add_object("bf16", "contiguous_le_v1", {248320, 2560});
    builder.bind_whole("mtp/token_embedding", mtp);
    builder.bind_whole("mtp/output_head", mtp);
    const auto vision = builder.add_object("bf16", "contiguous_le_v1", {2560, 2560});
    builder.bind_whole("vision/patch_embedding", vision);
    const auto proposal = builder.add_object("q4_g64_fp16", "row_split_k128_v1", {248320, 2560});
    builder.bind_whole("proposal/head", proposal);

    return std::move(builder.artifact);
}

std::filesystem::path fixture_path() {
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

std::filesystem::path write_fixture(FixtureArtifact& artifact) {
    std::uint64_t cursor = 0;
    for (auto& object : artifact.objects) {
        const auto format = ninfer::artifact::parse_v3_format(object.format);
        const auto layout = ninfer::artifact::parse_v3_layout(object.layout);
        object.bytes = ninfer::artifact::tensor_encoded_size(layout, format, object.shape);
        cursor       = align_up(cursor, kAlignment);
        object.offset = cursor;
        cursor += object.bytes;
    }

    Json objects = Json::array();
    for (const auto& object : artifact.objects) {
        objects.push_back(Json{{"id", object.id},
                               {"kind", "tensor"},
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
    Json directory{
        {"metadata", Json{{"name", "qwen3.8-flash-next"}}},
        {"files", Json::array({Json{{"path", nullptr}, {"payload_bytes", cursor}}})},
        {"objects", objects},
        {"bindings", bindings},
        {"uses", artifact.uses},
        {"components", Json{{"text", Json{{"config", artifact.config}}}}},
    };

    const std::string text = directory.dump();
    const auto payload_start = align_up(kHeaderBytes + text.size(), kAlignment);
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

const flash::BoundObject& object_of(const flash::BindingPlan& plan, const flash::BoundPart& part) {
    return plan.objects.at(part.object);
}

void require_rectangle(const flash::BoundParameter& parameter, std::uint64_t row_begin,
                       std::uint64_t row_count, std::uint64_t column_count,
                       const std::string& what) {
    require(!parameter.parts.empty(), what + " has no part");
    for (const auto& part : parameter.parts) {
        require(part.view.matrix_count == 1, what + " spans more than one matrix");
        require(part.view.matrix_row_begin == row_begin,
                what + " starts at row " + std::to_string(part.view.matrix_row_begin) +
                    "; it starts at " + std::to_string(row_begin));
        require(part.view.matrix_row_count == row_count,
                what + " spans " + std::to_string(part.view.matrix_row_count) + " rows; it spans " +
                    std::to_string(row_count));
        require(part.view.column_count == column_count,
                what + " spans " + std::to_string(part.view.column_count) + " columns; it spans " +
                    std::to_string(column_count));
    }
}

void test_identity() {
    require(Package::model_id == "qwen3.8-flash-next", "the package's model id is wrong");
    require(Package::weights_id == "nvfp4", "the package's weights id is wrong");
    require(Package::target_key == "qwen3_8_flash_next", "the package's target key is wrong");
    require(Package::resolve_weights("qwen3.8-flash-next", "nvfp4").has_value(),
            "the package does not resolve its own identity");
    require(!Package::resolve_weights("qwen3.8-27b", "nvfp4").has_value(),
            "the package resolves another target's identity");
    require(!Package::resolve_weights("qwen3.8-flash-next", "gguf-q4-k-m").has_value(),
            "the package resolves another weights recipe");
    require(!Package::resolve_weights("qwen3.8-flash-next", "").has_value(),
            "the package resolves an empty weights id");

    const auto defaults = Package::sampling_defaults();
    require(defaults.thinking.temperature == 1.0F, "the thinking temperature is wrong");
    require(defaults.thinking.top_k == 20, "the thinking top-k is wrong");
    require(defaults.thinking.top_p == 0.95F, "the thinking top-p is wrong");
    require(defaults.non_thinking.temperature == 1.0F, "the non-thinking temperature is wrong");
    require(defaults.non_thinking.top_k == 20, "the non-thinking top-k is wrong");
    require(defaults.non_thinking.top_p == 0.95F, "the non-thinking top-p is wrong");
}

void check_coverage(const flash::BindingPlan& plan, const Reader& reader) {
    const auto& directory = reader.v3_directory();
    std::set<std::string> artifact_text;
    for (const auto& entry : directory.bindings) {
        if (entry.first.starts_with("text/")) { artifact_text.insert(entry.first); }
    }
    for (const auto& entry : directory.part_bindings) {
        if (entry.first.starts_with("text/")) { artifact_text.insert(entry.first); }
    }

    const auto parameters = plan.parameters();
    require(parameters.size() == roles::kTextNameCount,
            "the plan binds " + std::to_string(parameters.size()) + " text parameters; it binds " +
                std::to_string(roles::kTextNameCount));
    require(artifact_text.size() == roles::kTextNameCount,
            "the artifact holds " + std::to_string(artifact_text.size()) +
                " text parameters; the role table expects " +
                std::to_string(roles::kTextNameCount));

    std::set<std::string> bound;
    for (const auto* parameter : parameters) {
        require(!parameter->parts.empty(), "the parameter " + parameter->parameter + " has no part");
        require(artifact_text.count(parameter->parameter) == 1,
                "the plan binds " + parameter->parameter + ", which the artifact does not hold");
        bound.insert(parameter->parameter);
        const auto role = roles::role_of(parameter->parameter);
        require(role.has_value() && *role == parameter->role,
                "the parameter " + parameter->parameter + " does not denote its role");
        if (roles::is_layer_scoped(parameter->role)) {
            const auto layer = roles::layer_of(parameter->parameter);
            require(layer.has_value() && *layer == parameter->layer,
                    "the parameter " + parameter->parameter + " does not denote its layer");
        }
    }
    require(bound == artifact_text,
            "the plan binds a different set of text parameters than the artifact holds");
}

void check_unclaimed(const flash::BindingPlan& plan, const std::map<std::string, std::size_t>& expected) {
    require(plan.unclaimed.by_namespace.size() == expected.size(),
            "the plan reports " + std::to_string(plan.unclaimed.by_namespace.size()) +
                " unclaimed namespaces; it reports " + std::to_string(expected.size()));
    for (const auto& [space, count] : expected) {
        const auto found = plan.unclaimed.by_namespace.find(space);
        require(found != plan.unclaimed.by_namespace.end(),
                "the plan does not report the unclaimed namespace " + space);
        require(found->second.size() == count,
                "the plan reports " + std::to_string(found->second.size()) +
                    " unclaimed " + space + " parameters; the artifact holds " +
                    std::to_string(count));
    }
}

void check_layers(const flash::BindingPlan& plan) {
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
        const auto& layer_plan = plan.layers[layer];
        require(layer_plan.attention_hc.norm.role == roles::Role::AttentionHcNorm,
                "layer " + std::to_string(layer) + " has no attention HC norm");
        require(layer_plan.moe_hc.norm.role == roles::Role::MoeHcNorm,
                "layer " + std::to_string(layer) + " has no MoE HC norm");
        require(layer_plan.moe.router.role == roles::Role::MoeRouter,
                "layer " + std::to_string(layer) + " has no router");
        require(layer_plan.moe.routed_gate.role == roles::Role::MoeRoutedGate,
                "layer " + std::to_string(layer) + " has no routed gate");
        if (layer % 4 == 3) {
            require(std::holds_alternative<flash::QsaPlan>(layer_plan.mixer),
                    "layer " + std::to_string(layer) + " has no QSA mixer");
            const auto& qsa = std::get<flash::QsaPlan>(layer_plan.mixer);
            require(qsa.query.role == roles::Role::QsaQuery,
                    "layer " + std::to_string(layer) + " has no QSA query");
            require(qsa.index_key.role == roles::Role::QsaIndexKey,
                    "layer " + std::to_string(layer) + " has no QSA index key");
        } else {
            require(std::holds_alternative<flash::GdnPlan>(layer_plan.mixer),
                    "layer " + std::to_string(layer) + " has no GDN mixer");
            const auto& gdn = std::get<flash::GdnPlan>(layer_plan.mixer);
            require(gdn.query.role == roles::Role::GdnQuery,
                    "layer " + std::to_string(layer) + " has no GDN query");
            require(gdn.z.role == roles::Role::GdnZ,
                    "layer " + std::to_string(layer) + " has no GDN z");
        }
    }
}

void check_routed_bank(const flash::BindingPlan& plan, std::uint32_t layer) {
    const auto& layer_plan = plan.layers[layer];
    for (const auto* parameter : {&layer_plan.moe.routed_gate, &layer_plan.moe.routed_up,
                                  &layer_plan.moe.routed_down}) {
        require(parameter->parts.size() == 512,
                "the routed bank of layer " + std::to_string(layer) + " holds " +
                    std::to_string(parameter->parts.size()) + " parts; it holds 512");
        const auto& object = object_of(plan, parameter->parts.front());
        require(object.format == NumericFormat::NVFP4,
                "the routed bank of layer " + std::to_string(layer) + " is not NVFP4");
        require(object.layout == StorageLayout::BatchedBlockScaleK16M128x4V1,
                "the routed bank of layer " + std::to_string(layer) +
                    " is not a batched block scale library");
        for (const auto& part : parameter->parts) {
            require(part.object == parameter->parts.front().object,
                    "the routed bank of layer " + std::to_string(layer) +
                        " is spread over several objects");
            require(part.view.matrix_count == 1,
                    "the routed bank of layer " + std::to_string(layer) +
                        " has a part that spans several matrices");
        }
        require(parameter->parts.front().view.matrix_begin == 0,
                "the routed bank of layer " + std::to_string(layer) + " starts at matrix 1");
        require(parameter->parts.back().view.matrix_begin == 511,
                "the routed bank of layer " + std::to_string(layer) + " ends before matrix 512");
    }
    require(layer_plan.moe.routed_gate.parts.front().view.matrix_row_count == 640,
            "the routed gate of layer " + std::to_string(layer) + " does not span 640 rows");
    require(layer_plan.moe.routed_up.parts.front().view.matrix_row_begin == 640,
            "the routed up projection of layer " + std::to_string(layer) +
                " does not start at row 640");
    require(layer_plan.moe.routed_up.parts.front().view.matrix_row_count == 640,
            "the routed up projection of layer " + std::to_string(layer) + " does not span 640 rows");
    require(layer_plan.moe.routed_down.parts.front().view.matrix_row_count == 2560,
            "the routed down projection of layer " + std::to_string(layer) +
                " does not span 2560 rows");
}

void check_activation_divisor(const flash::BindingPlan& plan, std::uint32_t layer) {
    const auto& layer_plan = plan.layers[layer];
    const auto check = [&](const flash::BoundParameter& parameter, std::uint64_t column) {
        const auto found = parameter.auxiliaries.find("activation_input_divisor");
        require(found != parameter.auxiliaries.end(),
                "the parameter " + parameter.parameter + " has no activation input divisor");
        require(found->second.size() == 1,
                "the parameter " + parameter.parameter + " has " +
                    std::to_string(found->second.size()) + " activation input divisors");
        const auto& part = found->second.front();
        require(part.view.matrix_row_begin == layer,
                "the activation input divisor of " + parameter.parameter + " is not the divisor "
                "of layer " + std::to_string(layer));
        require(part.view.column_begin == column,
                "the activation input divisor of " + parameter.parameter + " is the wrong "
                "divisor of its layer");
    };
    check(layer_plan.moe.routed_gate, 0);
    check(layer_plan.moe.routed_up, 0);
    check(layer_plan.moe.routed_down, 1);
}

void check_geometry(const flash::BindingPlan& plan) {
    require_rectangle(plan.token_embedding, 0, 248320, 2560, "the token embedding");
    require_rectangle(plan.output_head, 0, 248320, 2560, "the output head");
    require_rectangle(plan.final_hc.norm, 0, 10240, 1, "the final HC norm");
    require_rectangle(plan.final_hc.down, 0, 320, 10240, "the final HC down projection");
    require_rectangle(plan.final_hc.up, 0, 10240, 320, "the final HC up projection");

    const auto& layer = plan.layers[0];
    require_rectangle(layer.attention_hc.norm, 0, 10240, 1, "the layer 0 attention HC norm");
    require_rectangle(layer.attention_hc.down, 0, 320, 10240,
                      "the layer 0 attention HC down projection");
    require_rectangle(layer.attention_hc.inject, 320, 4, 10240,
                      "the layer 0 attention HC injection projection");
    require_rectangle(layer.attention_hc.up, 0, 10240, 320, "the layer 0 attention HC up projection");
    require_rectangle(layer.moe_hc.down, 0, 320, 10240, "the layer 0 MoE HC down projection");
    require_rectangle(layer.moe_hc.inject, 320, 4, 10240,
                      "the layer 0 MoE HC injection projection");
    require_rectangle(layer.moe.router, 0, 512, 2560, "the layer 0 router");
    require_rectangle(layer.moe.shared_score, 512, 1, 2560, "the layer 0 shared logit");
    require_rectangle(layer.moe.shared_gate, 513, 640, 2560, "the layer 0 shared gate");
    require_rectangle(layer.moe.shared_up, 1153, 640, 2560, "the layer 0 shared up projection");
    require_rectangle(layer.moe.shared_down, 0, 2560, 640, "the layer 0 shared down projection");

    const auto& gdn = std::get<flash::GdnPlan>(layer.mixer);
    require_rectangle(gdn.query, 0, 2048, 2560, "the layer 0 GDN query");
    require_rectangle(gdn.key, 2048, 2048, 2560, "the layer 0 GDN key");
    require_rectangle(gdn.value, 4096, 6144, 2560, "the layer 0 GDN value");
    require_rectangle(gdn.z, 10240, 6144, 2560, "the layer 0 GDN z");
    require_rectangle(gdn.a_projection, 0, 48, 2560, "the layer 0 GDN a projection");
    require_rectangle(gdn.b_projection, 48, 48, 2560, "the layer 0 GDN b projection");
    require_rectangle(gdn.a_log, 0, 48, 1, "the layer 0 GDN a log");
    require_rectangle(gdn.dt_bias, 0, 48, 1, "the layer 0 GDN dt bias");
    require_rectangle(gdn.convolution, 0, 4, 10240, "the layer 0 GDN convolution");
    require_rectangle(gdn.norm, 0, 128, 1, "the layer 0 GDN norm");
    require_rectangle(gdn.output, 0, 2560, 6144, "the layer 0 GDN output projection");

    const auto& qsa = std::get<flash::QsaPlan>(plan.layers[3].mixer);
    require_rectangle(qsa.query, 0, 6144, 2560, "the layer 3 QSA query");
    require_rectangle(qsa.output_gate, 6144, 6144, 2560, "the layer 3 QSA output gate");
    require_rectangle(qsa.key, 0, 512, 2560, "the layer 3 QSA key");
    require_rectangle(qsa.value, 512, 512, 2560, "the layer 3 QSA value");
    require_rectangle(qsa.index_query, 1024, 512, 2560, "the layer 3 QSA index query");
    require_rectangle(qsa.index_key, 1536, 128, 2560, "the layer 3 QSA index key");
    require_rectangle(qsa.output, 0, 2560, 6144, "the layer 3 QSA output projection");
    require_rectangle(qsa.query_norm, 0, 256, 1, "the layer 3 QSA query norm");
    require_rectangle(qsa.key_norm, 0, 256, 1, "the layer 3 QSA key norm");
    require_rectangle(qsa.index_query_norm, 0, 128, 1, "the layer 3 QSA index query norm");
    require_rectangle(qsa.index_key_norm, 0, 128, 1, "the layer 3 QSA index key norm");

    require_rectangle(plan.ple.table, 0, 2500012, 160, "the PLE table");
    require_rectangle(plan.ple.key, 0, 10240, 2560, "the PLE key projection");
    require_rectangle(plan.ple.value, 10240, 2560, 2560, "the PLE value projection");
    require_rectangle(plan.ple.norm_key, 0, 10240, 1, "the PLE key norm");
    require_rectangle(plan.ple.norm_query, 0, 10240, 1, "the PLE query norm");
    require_rectangle(plan.ple.norm_conv, 0, 10240, 1, "the PLE convolution norm");
    require_rectangle(plan.ple.convolution, 0, 4, 10240, "the PLE convolution");
    require(plan.ple.table.parts.size() == 128,
            "the PLE table holds " + std::to_string(plan.ple.table.parts.size()) +
                " shards; it holds 128");
}

template <typename Function>
void expect_rejection(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("the package accepted " + what);
}

void test_synthetic_artifact() {
    auto artifact = build_fixture();
    const auto path = write_fixture(artifact);
    const TemporaryArtifact temporary(path);

    Reader reader(path);
    Binder binder(reader, 1);
    const auto plan = Package::plan_load(binder, Package::WeightsProfile::Nvfp4, 1);
    const auto& bindings = plan.bindings();

    check_coverage(bindings, reader);
    check_layers(bindings);
    check_geometry(bindings);
    check_routed_bank(bindings, 0);
    check_activation_divisor(bindings, 0);
    check_unclaimed(bindings, {{"mtp", 2}, {"vision", 1}, {"proposal", 1}});

    require(bindings.objects.size() == 955,
            "the plan binds " + std::to_string(bindings.objects.size()) +
                " objects; it binds 955");

    bool rejected = false;
    try {
        Binder wider(reader, 1);
        (void)Package::plan_load(wider, Package::WeightsProfile::Nvfp4, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "the package accepts a tensor-parallel plan it does not own");

    std::cout << "flash-next load plan: " << bindings.parameters().size()
              << " text parameters bound to " << bindings.objects.size() << " objects\n";
}

void test_unclaimed_text_parameter_is_rejected() {
    auto artifact = build_fixture();
    FixtureBinding extra;
    extra.name   = "text/layers/0/gdn/unknown";
    extra.whole  = true;
    extra.object = artifact.objects.front().id;
    artifact.bindings.push_back(std::move(extra));

    const auto path = write_fixture(artifact);
    const TemporaryArtifact temporary(path);
    Reader reader(path);
    expect_rejection(
        [&] {
            Binder binder(reader, 1);
            (void)Package::plan_load(binder, Package::WeightsProfile::Nvfp4, 1);
        },
        "a text parameter it does not claim");
}

void test_foreign_configuration_is_rejected() {
    auto artifact = build_fixture();
    artifact.config["hidden_size"] = 2561;

    const auto path = write_fixture(artifact);
    const TemporaryArtifact temporary(path);
    Reader reader(path);
    expect_rejection(
        [&] {
            Binder binder(reader, 1);
            (void)Package::plan_load(binder, Package::WeightsProfile::Nvfp4, 1);
        },
        "a configuration this package does not expect");
}

void test_real_artifact(const char* path) {
    Reader reader(path);
    Binder binder(reader, 1);
    const auto plan = Package::plan_load(binder, Package::WeightsProfile::Nvfp4, 1);
    const auto& bindings = plan.bindings();

    check_coverage(bindings, reader);
    check_layers(bindings);
    check_geometry(bindings);
    check_routed_bank(bindings, 0);
    check_activation_divisor(bindings, 0);
    check_unclaimed(bindings, {{"mtp", 36}, {"vision", 441}, {"proposal", 1}});

    const auto object_name = [&](const flash::BoundParameter& parameter) {
        return bindings.objects.at(parameter.parts.front().object).name;
    };
    require(object_name(bindings.token_embedding) == "weight/000525",
            "the token embedding is not weight/000525");
    require(object_name(bindings.output_head) == "weight/000526",
            "the output head is not weight/000526");
    require(object_name(bindings.final_hc.norm) == "weight/000527",
            "the final HC norm is not weight/000527");
    require(object_name(bindings.final_hc.down) == "weight/000528",
            "the final HC down projection is not weight/000528");
    require(object_name(bindings.final_hc.up) == "weight/000529",
            "the final HC up projection is not weight/000529");

    const auto& layer0 = bindings.layers[0];
    require(object_name(layer0.attention_hc.norm) == "weight/000530",
            "the layer 0 attention HC norm is not weight/000530");
    require(object_name(layer0.attention_hc.down) == "weight/000226",
            "the layer 0 attention HC down projection is not weight/000226");
    require(object_name(layer0.moe_hc.down) == "weight/000227",
            "the layer 0 MoE HC down projection is not weight/000227");
    require(object_name(layer0.moe.router) == "weight/000228",
            "the layer 0 router is not weight/000228");
    require(object_name(layer0.moe.shared_down) == "weight/000534",
            "the layer 0 shared down projection is not weight/000534");
    const auto& gdn0 = std::get<flash::GdnPlan>(layer0.mixer);
    require(object_name(gdn0.query) == "weight/000229",
            "the layer 0 GDN query is not weight/000229");
    require(object_name(gdn0.key) == "weight/000229",
            "the layer 0 GDN key is not weight/000229");
    require(object_name(gdn0.a_projection) == "weight/000230",
            "the layer 0 GDN a projection is not weight/000230");
    require(object_name(gdn0.output) == "weight/000535",
            "the layer 0 GDN output projection is not weight/000535");
    require(object_name(gdn0.a_log) == "weight/000537",
            "the layer 0 GDN a log is not weight/000537");
    require(object_name(gdn0.dt_bias) == "weight/000538",
            "the layer 0 GDN dt bias is not weight/000538");
    require(object_name(gdn0.convolution) == "weight/000536",
            "the layer 0 GDN convolution is not weight/000536");
    require(object_name(gdn0.norm) == "weight/000539",
            "the layer 0 GDN norm is not weight/000539");

    const auto& qsa3 = std::get<flash::QsaPlan>(bindings.layers[3].mixer);
    require(object_name(qsa3.query) == "weight/000245",
            "the layer 3 QSA query is not weight/000245");
    require(object_name(qsa3.output_gate) == "weight/000245",
            "the layer 3 QSA output gate is not weight/000245");
    require(object_name(qsa3.key) == "weight/000246",
            "the layer 3 QSA key is not weight/000246");
    require(object_name(qsa3.index_key) == "weight/000246",
            "the layer 3 QSA index key is not weight/000246");
    require(object_name(qsa3.output) == "weight/000569",
            "the layer 3 QSA output projection is not weight/000569");
    require(object_name(qsa3.query_norm) == "weight/000570",
            "the layer 3 QSA query norm is not weight/000570");
    require(object_name(qsa3.key_norm) == "weight/000571",
            "the layer 3 QSA key norm is not weight/000571");
    require(object_name(qsa3.index_query_norm) == "weight/000572",
            "the layer 3 QSA index query norm is not weight/000572");
    require(object_name(qsa3.index_key_norm) == "weight/000573",
            "the layer 3 QSA index key norm is not weight/000573");

    require(object_name(bindings.layers[0].moe.routed_gate) == "weight/000000",
            "the layer 0 routed gate is not weight/000000");
    require(object_name(bindings.layers[0].moe.routed_down) == "weight/000001",
            "the layer 0 routed down projection is not weight/000001");
    require(object_name(bindings.layers[1].moe.routed_gate) == "weight/000002",
            "the layer 1 routed gate is not weight/000002");
    require(object_name(bindings.ple.key) == "weight/000236",
            "the PLE key projection is not weight/000236");
    require(bindings.objects.at(bindings.ple.table.parts.front().object).name == "weight/000098",
            "the PLE table does not start at weight/000098");
    require(bindings.objects.at(bindings.ple.table.parts.back().object).name == "weight/000225",
            "the PLE table does not end at weight/000225");

    std::cout << "flash-next load plan: " << bindings.parameters().size()
              << " text parameters bound to " << bindings.objects.size() << " objects, "
              << bindings.unclaimed.by_namespace.at("mtp").size() << " mtp / "
              << bindings.unclaimed.by_namespace.at("vision").size() << " vision / "
              << bindings.unclaimed.by_namespace.at("proposal").size() << " proposal unclaimed\n";
}

} // namespace

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    try {
        test_identity();
        if (!artifact_mode) {
            test_synthetic_artifact();
            test_unclaimed_text_parameter_is_rejected();
            test_foreign_configuration_is_rejected();
            return 0;
        }
        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }
        test_real_artifact(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
