#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::array<std::byte, 8> kV3Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::array<std::byte, 8> kV3PartMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'P'},
    std::byte{'R'}, std::byte{'T'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kV3HeaderBytes    = 32;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

const Json& require_member(const Json& value, const char* member, std::string_view label) {
    const auto it = value.find(member);
    if (it == value.end()) {
        throw ArtifactError(std::string(label) + " is missing its " + member + " member");
    }
    return *it;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP16") { return NumericFormat::FP16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    if (name == "GGML_K") { return NumericFormat::GGML_K; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    if (name == "ggml-k256-v1") { return StorageLayout::GgmlK256V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape, stored_size);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

struct V3CompatibilityDirectory {
    Json directory;
    std::map<std::string, std::vector<std::byte>, std::less<>> payload_overrides;
};

std::uint64_t require_v3_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw ArtifactError(std::string(label) + " must be a nonnegative integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

std::string v3_metadata_name(const Json& directory) {
    if (!directory.is_object() || !directory.contains("metadata") ||
        !directory.at("metadata").is_object() || !directory.at("metadata").contains("name") ||
        !directory.at("metadata").at("name").is_string()) {
        return {};
    }
    return directory.at("metadata").at("name").get<std::string>();
}

bool v3_is_scalar(const Json& value) {
    return value.is_string() || value.is_boolean() || value.is_number();
}

// One element index of a binding range: a nonnegative integer.
bool v3_is_index(const Json& value) {
    return value.is_number_unsigned() ||
           (value.is_number_integer() && value.get<std::int64_t>() >= 0);
}

// The text form of one configuration scalar: a number is its shortest round trip form and a
// boolean is "true" or "false".
std::string v3_scalar_text(const Json& value) {
    if (value.is_string()) { return value.get_ref<const std::string&>(); }
    if (value.is_boolean()) { return value.get<bool>() ? "true" : "false"; }
    if (value.is_number_unsigned()) { return std::to_string(value.get<std::uint64_t>()); }
    if (value.is_number_integer()) { return std::to_string(value.get<std::int64_t>()); }
    const double number = value.get<double>();
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), number);
    if (result.ec != std::errc{}) {
        throw ArtifactError("v3 configuration number has no text form");
    }
    return std::string(buffer.data(), result.ptr);
}

// Flattens a configuration object into dot joined key paths. A nested object becomes a path
// prefix, an array is not a scalar: its elements are joined into one comma separated value when
// every element is a scalar, and the array is skipped otherwise.
void v3_flatten_config(const Json& value, const std::string& path,
                        std::map<std::string, std::string>& out) {
    if (value.is_object()) {
        for (const auto& [key, member] : value.items()) {
            v3_flatten_config(member, path.empty() ? key : path + "." + key, out);
        }
        return;
    }
    if (value.is_array()) {
        if (value.empty()) { return; }
        std::string joined;
        bool first = true;
        for (const auto& element : value) {
            if (!v3_is_scalar(element)) { return; }
            if (!first) { joined.push_back(','); }
            first = false;
            joined += v3_scalar_text(element);
        }
        out.emplace(path, std::move(joined));
        return;
    }
    if (v3_is_scalar(value)) { out.emplace(path, v3_scalar_text(value)); }
}

void v3_parse_components(const Json& directory, V3Directory& out) {
    const auto components = directory.find("components");
    if (components == directory.end() || !components->is_object()) { return; }
    for (const auto& [name, component] : components->items()) {
        if (!component.is_object()) { continue; }
        const auto config = component.find("config");
        if (config == component.end() || !config->is_object()) { continue; }
        std::map<std::string, std::string> flattened;
        v3_flatten_config(*config, "", flattened);
        out.component_config.emplace(name, std::move(flattened));
    }
}

void v3_parse_bindings(const Json& directory, V3Directory& out) {
    const auto bindings = directory.find("bindings");
    if (bindings == directory.end() || !bindings->is_object()) { return; }
    for (const auto& [name, binding] : bindings->items()) {
        if (!binding.is_object()) { continue; }
        const auto object = binding.find("object");
        if (object != binding.end() && object->is_string()) {
            out.bindings.emplace(name, object->get<std::string>());
            continue;
        }
        // A binding that names no single object names parts of objects instead. Its parts are
        // reported separately, so that a parameter is never dropped silently.
        const auto parts = binding.find("parts");
        if (parts == binding.end() || !parts->is_array() || parts->empty()) { continue; }
        std::vector<V3BindingPart> resolved;
        bool complete = true;
        for (const auto& part : *parts) {
            if (!part.is_object()) {
                complete = false;
                break;
            }
            const auto id    = part.find("object");
            const auto range = part.find("range");
            if (id == part.end() || !id->is_string() || range == part.end() ||
                !range->is_array() || range->size() != 2 || !v3_is_index((*range)[0]) ||
                !v3_is_index((*range)[1])) {
                complete = false;
                break;
            }
            resolved.push_back(V3BindingPart{id->get<std::string>(),
                                             (*range)[0].get<std::uint64_t>(),
                                             (*range)[1].get<std::uint64_t>()});
        }
        if (!complete) { continue; }
        out.part_bindings.emplace(name, std::move(resolved));
    }
}

void v3_parse_uses(const Json& directory, V3Directory& out) {
    const auto uses = directory.find("uses");
    if (uses == directory.end() || !uses->is_array()) { return; }
    for (const auto& entry : *uses) {
        if (!entry.is_object()) { continue; }
        const auto parameter = entry.find("parameter");
        if (parameter == entry.end() || !parameter->is_string()) { continue; }
        V3Use use;
        use.parameter = parameter->get<std::string>();
        if (const auto input = entry.find("input"); input != entry.end() && input->is_string()) {
            use.input = input->get<std::string>();
        }
        if (const auto policy = entry.find("activation_policy");
            policy != entry.end() && policy->is_string()) {
            use.activation_policy = policy->get<std::string>();
        }
        out.uses.push_back(std::move(use));
    }
}

V3Directory v3_parse_directory(const Json& directory) {
    V3Directory out;
    v3_parse_components(directory, out);
    v3_parse_bindings(directory, out);
    v3_parse_uses(directory, out);
    return out;
}

class Qwen38Nvfp4V3Adapter {
public:
    Qwen38Nvfp4V3Adapter(const Json& directory, const MappedFile& file,
                         std::uint64_t payload_start)
        : directory_(directory), file_(file), payload_start_(payload_start) {
        if (!directory_.is_object() || !directory_.contains("metadata") ||
            !directory_.at("metadata").is_object()) {
            throw ArtifactError("NInfer v3 directory is missing its metadata");
        }
        // The native NVFP4 27B package and the published Coder390 W4A4/W8A8 package share one
        // v3 object/binding schema; the only differences are the vision quantization recipe
        // (native FP16 vs the Q4/Q5/Q6/W8 set) and the DFlash2 optional component, so both map
        // through this adapter to their registered 27B identities.
        const auto meta_name = directory_.at("metadata").value("name", "");
        if (meta_name == "qwen3.8-27b") {
            model_id_   = "qwen3.8-27b";
            weights_id_ = "nvfp4";
        } else if (meta_name == "qwen3.8-27b-coder390-w4a4-w8a8-mtp-dflash2") {
            model_id_   = "qwen3.8-27b-coder390";
            weights_id_ = "nvfp4-qvis";
        } else {
            throw ArtifactError(
                "NInfer v3 compatibility is limited to qwen3.8-27b and "
                "qwen3.8-27b-coder390-w4a4-w8a8-mtp-dflash2");
        }
        if (!directory_.contains("files") || !directory_.at("files").is_array() ||
            directory_.at("files").size() != 1 ||
            !directory_.at("files")[0].at("path").is_null()) {
            throw ArtifactError("qwen3.8-27b v3 compatibility requires a single-file artifact");
        }
        const auto payload_bytes = require_v3_unsigned(
            directory_.at("files")[0].at("payload_bytes"), "payload_bytes", true);
        if (checked_add(payload_start_, payload_bytes, "v3 artifact length") != file_.size()) {
            throw ArtifactError("v3 artifact length differs from its directory");
        }
        if (!directory_.contains("objects") || !directory_.at("objects").is_array() ||
            !directory_.contains("bindings") || !directory_.at("bindings").is_object() ||
            !directory_.contains("uses") || !directory_.at("uses").is_array() ||
            !directory_.contains("components") || !directory_.at("components").is_object()) {
            throw ArtifactError("qwen3.8-27b v3 directory is incomplete");
        }
        for (const auto& object : directory_.at("objects")) {
            const auto& id = require_string(object.at("id"), "v3 object id");
            if (!objects_.emplace(id, &object).second) {
                throw ArtifactError("duplicate v3 object id: " + id);
            }
        }
    }

    V3CompatibilityDirectory build() {
        for (const auto& [role, component] : std::array{
                 std::pair{"tokenizer.json", "text"},
                 std::pair{"tokenizer_config.json", "text"},
                 std::pair{"chat_template.jinja", "text"},
                 std::pair{"generation_config.json", "text"},
                 std::pair{"preprocessor_config.json", "vision"},
                 std::pair{"video_preprocessor_config.json", "vision"},
             }) {
            add_resource("frontend/" + std::string(role), component, role);
        }

        for (const char* name : {"text/token_embedding", "text/output_head", "text/final_norm"}) {
            add(name, {name});
        }
        add("text/draft_head", {"proposal/head"});
        add("text/draft_head_token_ids", {"proposal/token_ids"});

        std::size_t nvfp4_mlp_layers = 0;
        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
            add(prefix + "input_norm", {prefix + "input_norm"});
            if (layer >= 3 && (layer - 3) % 4 == 0) {
                add(prefix + "attention/query_key_gate_value",
                    {prefix + "attention/query", prefix + "attention/key",
                     prefix + "attention/gate", prefix + "attention/value"});
                for (const char* role : {"query_norm", "key_norm", "output"}) {
                    add(prefix + "attention/" + role, {prefix + "attention/" + role});
                }
            } else {
                for (const char* role : {"a_log", "dt_bias", "convolution"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
                add(prefix + "gdn/a_b_projection",
                    {prefix + "gdn/a_projection", prefix + "gdn/b_projection"});
                add(prefix + "gdn/query_key_value_z",
                    {prefix + "gdn/query", prefix + "gdn/key", prefix + "gdn/value",
                     prefix + "gdn/z"});
                for (const char* role : {"norm", "output"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
            }
            add(prefix + "post_attention_norm", {prefix + "post_attention_norm"});
            const std::string gate = prefix + "mlp/gate";
            const std::string up   = prefix + "mlp/up";
            const auto& gate_up = add(prefix + "mlp/gate_up", {gate, up});
            add(prefix + "mlp/down", {prefix + "mlp/down"});
            if (require_string(gate_up.at("format"), "v3 tensor format") == "nvfp4") {
                ++nvfp4_mlp_layers;
                add_input_divisor(prefix + "mlp/gate_up_projection/input_scale_divisor",
                                  {gate, up});
                add_input_divisor(prefix + "mlp/down_projection/input_scale_divisor",
                                  {prefix + "mlp/down"});
            }
        }

        for (const char* name : {"input_projection", "embedding_norm", "hidden_norm", "final_norm"}) {
            add("mtp/" + std::string(name), {"mtp/" + std::string(name)});
        }
        const std::string old_mtp = "mtp/layer/";
        const std::string new_mtp = "mtp/layers/0/";
        for (const char* role : {"input_norm", "post_attention_norm"}) {
            add(old_mtp + role, {new_mtp + role});
        }
        add(old_mtp + "attention/query_key_gate_value",
            {new_mtp + "attention/query", new_mtp + "attention/key",
             new_mtp + "attention/gate", new_mtp + "attention/value"});
        for (const char* role : {"query_norm", "key_norm", "output"}) {
            add(old_mtp + "attention/" + role, {new_mtp + "attention/" + role});
        }
        add(old_mtp + "mlp/gate_up", {new_mtp + "mlp/gate", new_mtp + "mlp/up"});
        add(old_mtp + "mlp/down", {new_mtp + "mlp/down"});

        for (const char* name : {"patch_embedding", "patch_embedding_bias", "position_embedding"}) {
            add("vision/" + std::string(name), {"vision/" + std::string(name)});
        }
        for (int layer = 0; layer < 27; ++layer) {
            const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
            add(prefix + "attention/qkv",
                {prefix + "attention/query", prefix + "attention/key", prefix + "attention/value"});
            add(prefix + "attention/qkv_bias",
                {prefix + "attention/query_bias", prefix + "attention/key_bias",
                 prefix + "attention/value_bias"});
            for (const char* role : {"output", "output_bias"}) {
                add(prefix + "attention/" + role, {prefix + "attention/" + role});
            }
            for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
                add(prefix + "mlp/" + role, {prefix + "mlp/" + role});
            }
            for (const char* norm : {"norm1", "norm2"}) {
                for (const char* part : {"weight", "bias"}) {
                    add(prefix + norm + "/" + part, {prefix + norm + "_" + part});
                }
            }
        }
        for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
            add("vision/merger/" + std::string(role), {"vision/merger/" + std::string(role)});
        }
        for (const char* part : {"weight", "bias"}) {
            add("vision/merger/norm/" + std::string(part),
                {"vision/merger/norm_" + std::string(part)});
        }

        const bool has_dflash2 = directory_.at("components").contains("dflash2");
        if (has_dflash2) { add_dflash2(); }

        constexpr std::size_t kStructuralObjects = 1012;
        constexpr std::size_t kDFlash2Objects    = 66; // alias context-K/V package
        const std::size_t expected =
            kStructuralObjects + 2 * nvfp4_mlp_layers +
            (has_dflash2 ? kDFlash2Objects + dflash_context_kv_objects_ : 0);
        if (selected_.size() != expected) {
            throw ArtifactError("qwen3.8-27b v3 projection produced " +
                                std::to_string(selected_.size()) + " objects; expected " +
                                std::to_string(expected));
        }
        Json objects = Json::array();
        for (const auto& [old_name, object] : selected_) {
            Json converted{
                {"name", old_name},
                {"kind", object->at("kind")},
                {"offset", require_v3_unsigned(object->at("offset"), "v3 object offset", false)},
                {"bytes", require_v3_unsigned(object->at("bytes"), "v3 object bytes", true)}};
            const auto kind = require_string(object->at("kind"), "v3 object kind");
            if (kind == "tensor") {
                static const std::map<std::string, std::string, std::less<>> formats = {
                    {"bf16", "BF16"}, {"fp16", "FP16"}, {"fp32", "FP32"}, {"int32", "I32"},
                    {"q4_g64_fp16", "Q4G64_F16S"}, {"q5_g64_fp16", "Q5G64_F16S"},
                    {"q6_g64_fp16", "Q6G64_F16S"}, {"q8_g32_fp16", "W8G32_F16S"},
                    {"nvfp4", "NVFP4"}, {"fp8_e4m3fn_row_bf16", "FP8_E4M3FN_ROW_BF16S"},
                };
                static const std::map<std::string, std::string, std::less<>> layouts = {
                    {"contiguous_le_v1", "contiguous-le-v1"},
                    {"row_split_k128_v1", "row-split-k128-v1"},
                    {"block_scale_k16_m128x4_v1", "blockscale-k16-m128x4-v1"},
                    {"row_scale_v1", "row-scale-v1"},
                };
                converted["shape"] = Json::array();
                for (const auto& dimension : object->at("shape")) {
                    converted["shape"].push_back(
                        require_v3_unsigned(dimension, "v3 shape dimension", true));
                }
                if (old_name.starts_with("dflash/") && old_name.ends_with("conv/base_kernel")) {
                    // v3 describes the checkpoint's [side,tap,channel] row-major tensor;
                    // the native Tensor descriptor is ne0-fast [channel,tap,side].
                    // These are the SAME bytes, not a transpose or weight repack.
                    if (converted["shape"] != Json::array({2, 2, 5120})) {
                        throw ArtifactError(old_name + ": invalid DFlash2 convolution shape");
                    }
                    converted["shape"] = {std::uint64_t{5120}, std::uint64_t{2}, std::uint64_t{2}};
                }
                converted["format"] = formats.at(require_string(object->at("format"), "v3 format"));
                converted["layout"] = layouts.at(require_string(object->at("layout"), "v3 layout"));
            } else if (kind == "resource" &&
                       require_string(object->at("encoding"), "v3 resource encoding") ==
                           "raw_bytes_v1") {
                converted["encoding"] = "raw-bytes-v1";
            } else {
                throw ArtifactError("unsupported qwen3.8-27b v3 object representation");
            }
            objects.push_back(std::move(converted));
        }
        std::sort(objects.begin(), objects.end(), [](const Json& a, const Json& b) {
            return a.at("offset").get<std::uint64_t>() < b.at("offset").get<std::uint64_t>();
        });

        // The template ships either as its own raw-bytes resource or embedded in
        // tokenizer_config.json. In the resource form the resource is authoritative: the projected
        // config is made to carry exactly those bytes, replacing whatever (possibly stale) field the
        // published package shipped, because the native directory contract requires the two to
        // agree. The embedded form instead shrinks the jinja allocation to the config's bytes.
        const auto resource_names = directory_.at("components").at("text").value("resources",
                                                                                Json::object());
        V3CompatibilityDirectory result;
        if (resource_names.contains("chat_template.jinja")) {
            // add_resource already projected the resource under its native name with true extent;
            // verify the converted entry exists and leave its allocation untouched.
            const auto template_it = std::find_if(objects.begin(), objects.end(),
                                                  [](const Json& object) {
                                                      return object.at("name") ==
                                                             "frontend/chat_template.jinja";
                                                  });
            if (template_it == objects.end()) {
                throw ArtifactError("chat_template.jinja resource did not project to its object");
            }
            const auto template_span = object_payload(*template_it);
            const std::string template_text(
                reinterpret_cast<const char*>(template_span.data()), template_span.size());
            const auto config_it = std::find_if(
                objects.begin(), objects.end(), [](const Json& object) {
                    return object.at("name") == "frontend/tokenizer_config.json";
                });
            if (config_it == objects.end()) {
                throw ArtifactError(
                    "tokenizer_config.json resource did not project to its object");
            }
            const auto config_span = object_payload(*config_it);
            Json config;
            try {
                const auto* begin = reinterpret_cast<const char*>(config_span.data());
                config = Json::parse(begin, begin + config_span.size());
            } catch (const Json::exception& error) {
                throw ArtifactError(std::string("invalid tokenizer_config.json: ") + error.what());
            }
            config["chat_template"] = template_text;
            // The override fully replaces the read bytes, so the physical allocation stays at its
            // declared extent (it abuts the next object); only the payload changes.
            const auto patched = config.dump(-1, ' ', /*ensure_ascii=*/false);
            const auto raw = std::as_bytes(std::span(patched.data(), patched.size()));
            result.payload_overrides.emplace("frontend/tokenizer_config.json",
                                             std::vector<std::byte>(raw.begin(), raw.end()));
        } else {
            const auto template_bytes = embedded_chat_template();
            auto template_it =
                std::find_if(objects.begin(), objects.end(), [](const Json& object) {
                    return object.at("name") == "frontend/chat_template.jinja";
                });
            if (template_it == objects.end() ||
                template_bytes.size() > template_it->at("bytes").get<std::uint64_t>()) {
                throw ArtifactError("embedded chat template does not fit its resource allocation");
            }
            (*template_it)["bytes"] = template_bytes.size();
            result.payload_overrides.emplace("frontend/chat_template.jinja", template_bytes);
        }
        result.directory = {{"identity", {{"model_id", model_id_}, {"weights_id", weights_id_}}},
                            {"objects", std::move(objects)}};
        return result;
    }

private:
    void add_dflash2() {
        const auto& component = directory_.at("components").at("dflash2");
        const auto& config = component.at("config");
        const auto& draft = config.at("dflash_config");
        if (component.at("target") != "text" ||
            config.at("num_hidden_layers") != 5 || config.at("intermediate_size") != 17408 ||
            config.at("num_attention_heads") != 32 || config.at("num_key_value_heads") != 8 ||
            config.at("head_dim") != 128 || config.at("sliding_window") != 2048 ||
            config.at("layer_types") != Json::array({"sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "sliding_attention"}) ||
            config.at("rope_parameters").at("rope_theta") != 10000000.0 ||
            std::abs(config.at("rms_norm_eps").get<double>() - 1e-6) > 1e-13 ||
            draft.at("target_layer_ids") != Json::array({5, 19, 33, 47, 61}) ||
            draft.at("mask_token_id") != 248070 || draft.at("conv_kernel_size") != 2 ||
            draft.at("conv_group_size") != 16 || draft.at("selector_rank") != 256 ||
            draft.at("selector_top_k") != 16) {
            throw ArtifactError("unsupported qwen3.8-27b v3 DFlash2 configuration");
        }
        for (const char* role : {"feature_projection", "context_norm", "final_norm"}) {
            add("dflash/" + std::string(role), {"dflash2/" + std::string(role)});
        }
        for (int layer = 0; layer < 5; ++layer) {
            const std::string suffix = "layers/" + std::to_string(layer) + "/";
            const std::string old_prefix = "dflash/" + suffix;
            const std::string prefix = "dflash2/" + suffix;
            add(old_prefix + "attention/query_key_value",
                {prefix + "attention/query", prefix + "attention/key", prefix + "attention/value"});
// Context K/V come from one of two published DFlash2 packages. The single-matrix package aliases
                // them onto the QKV rows (the draft and context streams share the projection); the
                // later fine-tuned package carries dedicated context matrices of the same [N,K]
                // extent. Both map through this adapter: the alias form adds no native objects,
                // the separate form adds two per layer. A mixed form would mis-decode the draft
                // context, so it is rejected.
                const auto context_key   = binding_parts(prefix + "attention/context_key");
                const auto context_value = binding_parts(prefix + "attention/context_value");
                const bool key_alias     = context_key == binding_parts(prefix + "attention/key");
                const bool value_alias   = context_value == binding_parts(prefix + "attention/value");
                if (key_alias != value_alias) {
                    throw ArtifactError(prefix + "attention/context_key: mixed alias and "
                        "independent context projections are unsupported");
                }
                if (!key_alias) {
                    add(old_prefix + "attention/context_key",
                        {prefix + "attention/context_key"});
                    add(old_prefix + "attention/context_value",
                        {prefix + "attention/context_value"});
                    dflash_context_kv_objects_ += 2;
                }
            add(old_prefix + "mlp/gate_up", {prefix + "mlp/gate", prefix + "mlp/up"});
            for (const char* role : {"input_norm", "post_attention_norm", "attention/query_norm",
                    "attention/key_norm", "attention/output", "mlp/down",
                    "attention_conv/base_kernel", "attention_conv/kernel_projection",
                    "mlp_conv/base_kernel", "mlp_conv/kernel_projection"}) {
                add(old_prefix + role, {prefix + role});
            }
        }
        for (const char* role : {"hidden_projection", "predecessor_codebook", "successor_codebook"}) {
            add("dflash/candidate_selector/" + std::string(role),
                {"dflash2/candidate_selector/" + std::string(role)});
        }
    }

    const Json& object(std::string_view id) const {
        const auto found = objects_.find(id);
        if (found == objects_.end()) { throw ArtifactError("missing v3 object: " + std::string(id)); }
        return *found->second;
    }

    std::vector<Json> binding_parts(const std::string& logical) const {
        const auto& bindings = directory_.at("bindings");
        if (!bindings.contains(logical)) { throw ArtifactError("missing v3 binding: " + logical); }
        const auto& binding = bindings.at(logical);
        if (binding.contains("object")) {
            const auto id = require_string(binding.at("object"), "v3 binding object");
            const auto& value = object(id);
            std::uint64_t elements = 1;
            for (const auto& dimension : value.at("shape")) {
                const auto dim = require_v3_unsigned(dimension, "v3 shape dimension", true);
                if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
                    throw ArtifactError("v3 tensor shape overflows u64");
                }
                elements *= dim;
            }
            return {Json{{"object", id}, {"range", {0, elements}}}};
        }
        if (!binding.contains("parts") || !binding.at("parts").is_array()) {
            throw ArtifactError("invalid v3 binding: " + logical);
        }
        return binding.at("parts").get<std::vector<Json>>();
    }

    const Json& add(const std::string& old_name, std::initializer_list<std::string> logical_names) {
        std::vector<Json> parts;
        for (const auto& logical : logical_names) {
            auto binding = binding_parts(logical);
            parts.insert(parts.end(), binding.begin(), binding.end());
        }
        std::string object_id;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& part : parts) {
            const auto id = require_string(part.at("object"), "v3 binding object");
            if (object_id.empty()) { object_id = id; }
            if (id != object_id) { throw ArtifactError(old_name + ": binding spans physical objects"); }
            const auto& range = part.at("range");
            if (!range.is_array() || range.size() != 2) {
                throw ArtifactError(old_name + ": invalid binding range");
            }
            ranges.emplace_back(require_v3_unsigned(range[0], "v3 binding range", false),
                                require_v3_unsigned(range[1], "v3 binding range", true));
        }
        std::sort(ranges.begin(), ranges.end());
        std::uint64_t cursor = 0;
        for (const auto [begin, end] : ranges) {
            if (begin != cursor || end <= begin) {
                throw ArtifactError(old_name + ": binding ranges are not one packed object");
            }
            cursor = end;
        }
        const auto& value = object(object_id);
        std::uint64_t elements = 1;
        for (const auto& dimension : value.at("shape")) {
            elements *= require_v3_unsigned(dimension, "v3 shape dimension", true);
        }
        if (cursor != elements) { throw ArtifactError(old_name + ": binding coverage is incomplete"); }
        select(old_name, object_id);
        return value;
    }

    void add_input_divisor(const std::string& old_name,
                           std::initializer_list<std::string> logical_names) {
        std::string selected_id;
        std::span<const std::byte> selected_bytes;
        for (const auto& logical : logical_names) {
            std::set<std::string> candidates;
            for (const auto& use : directory_.at("uses")) {
                if (use.value("parameter", "") != logical || !use.contains("auxiliaries")) { continue; }
                const auto& auxiliaries = use.at("auxiliaries");
                if (auxiliaries.contains("activation_input_divisor")) {
                    candidates.insert(require_string(
                        auxiliaries.at("activation_input_divisor").at("object"),
                        "activation input divisor"));
                }
            }
            if (candidates.size() != 1) {
                throw ArtifactError(logical + ": expected one activation input divisor");
            }
            const auto& id = *candidates.begin();
            const auto& value = object(id);
            if (value.at("format") != "fp32" || value.at("layout") != "contiguous_le_v1" ||
                !value.at("shape").empty()) {
                throw ArtifactError(logical + ": invalid activation input divisor");
            }
            const auto bytes = object_payload(value);
            if (selected_id.empty()) {
                selected_id = id;
                selected_bytes = bytes;
            } else if (!std::equal(bytes.begin(), bytes.end(), selected_bytes.begin(), selected_bytes.end())) {
                throw ArtifactError(old_name + ": fused parameters have different input divisors");
            }
        }
        select(old_name, selected_id);
    }

    void add_resource(const std::string& old_name, const std::string& component,
                      const std::string& role) {
        const auto& components = directory_.at("components");
        const auto id = require_string(components.at(component).at("resources").at(role),
                                       "v3 component resource");
        select(old_name, id);
    }

    void select(const std::string& old_name, const std::string& object_id) {
        if (!selected_.emplace(old_name, &object(object_id)).second) {
            throw ArtifactError("duplicate projected object name: " + old_name);
        }
        if (!physical_names_.emplace(object_id, old_name).second) {
            throw ArtifactError("v3 physical object is selected more than once: " + object_id);
        }
    }

    std::span<const std::byte> object_payload(const Json& value) const {
        const auto offset = require_v3_unsigned(value.at("offset"), "v3 object offset", false);
        const auto bytes  = require_v3_unsigned(value.at("bytes"), "v3 object bytes", true);
        const auto begin  = checked_add(payload_start_, offset, "v3 object payload");
        const auto end    = checked_add(begin, bytes, "v3 object payload");
        if (end > file_.size()) { throw ArtifactError("v3 object extends beyond the file"); }
        return {file_.data() + begin, static_cast<std::size_t>(bytes)};
    }

    std::vector<std::byte> embedded_chat_template() const {
        const auto& components = directory_.at("components");
        const auto id = require_string(
            components.at("text").at("resources").at("tokenizer_config.json"),
            "tokenizer_config.json resource");
        const auto bytes = object_payload(object(id));
        Json config;
        try {
            const auto* begin = reinterpret_cast<const char*>(bytes.data());
            config = Json::parse(begin, begin + bytes.size());
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid tokenizer_config.json: ") + error.what());
        }
        const auto& value = require_string(config.at("chat_template"), "chat_template");
        const auto raw = std::as_bytes(std::span(value.data(), value.size()));
        return {raw.begin(), raw.end()};
    }

    std::string model_id_;
    std::string weights_id_;
    // Separate-package DFlash2 layers each add a context K/V pair; zero for the alias form.
    std::size_t dflash_context_kv_objects_ = 0;
    const Json& directory_;
    const MappedFile& file_;
    std::uint64_t payload_start_;
    std::map<std::string, const Json*, std::less<>> objects_;
    std::map<std::string, const Json*, std::less<>> selected_;
    std::map<std::string, std::string, std::less<>> physical_names_;
};

// An upstream v3 artifact carries no product identity: the reader synthesizes one from the
// directory's architecture name. Each name is bound to exactly one registered identity.
struct V3Identity {
    std::string_view metadata_name;
    std::string_view model_id;
    std::string_view weights_id;
};

constexpr std::array<V3Identity, 1> kContainerV3Identities = {{
    {"qwen3.8-flash-next", "qwen3.8-flash-next", "nvfp4"},
}};

using ObjectIndex =
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>;

// Upstream v3 is a multi-volume container: an entry file holding the 32-byte header and the
// directory, plus NINPRT continuation files. The directory's file list gives each volume's
// payload extent, and the logical payload space is the entry payload followed by the
// continuation payloads in file order. Object offsets are absolute in that logical space, whose
// origin is the first byte of the entry payload.
class V3MultiPartReader {
public:
    V3MultiPartReader(const std::filesystem::path& entry_path, const MappedFile& entry,
                      const Json& directory, std::uint64_t entry_payload_start)
        : entry_(entry) {
        resolve_identity(directory);
        open_volumes(entry_path, directory, entry_payload_start);
        parse_objects(directory);
        directory_ = v3_parse_directory(directory);
    }

    static bool recognizes(std::string_view metadata_name) noexcept {
        return find_identity(metadata_name) != nullptr;
    }

    const ArtifactIdentity& identity() const noexcept { return identity_; }

    const V3Directory& directory() const noexcept { return directory_; }

    const std::vector<ObjectDescriptor>& objects() const noexcept { return objects_; }

    const ObjectDescriptor* find(std::string_view name) const noexcept {
        const auto it = index_.find(name);
        return it == index_.end() ? nullptr : &objects_[it->second];
    }

    std::uint64_t file_bytes() const noexcept { return file_bytes_; }

    PayloadSpan payload(const ObjectDescriptor& object) const {
        const auto offset = object_offset(object);
        const auto bytes  = object_bytes(object);
        const auto pieces = segments(offset, bytes);
        if (pieces.size() != 1) {
            throw ArtifactError("object " + std::string(object_name(object)) + " spans volumes");
        }
        const auto& piece = pieces.front();
        const auto& volume = volumes_[piece.volume];
        if (piece.file_offset + piece.bytes > volume.file->size()) {
            throw ArtifactError("object " + std::string(object_name(object)) +
                                " extends beyond its volume");
        }
        return {offset,
                std::span<const std::byte>(volume.file->data() + piece.file_offset,
                                           static_cast<std::size_t>(piece.bytes))};
    }

    std::vector<PayloadSegment> payload_segments(std::uint64_t offset, std::uint64_t bytes) const {
        std::vector<PayloadSegment> result;
        for (const auto& piece : segments(offset, bytes)) {
            result.push_back(PayloadSegment{piece.logical_offset, piece.bytes});
        }
        return result;
    }

    std::size_t read_direct(std::uint64_t logical_offset, std::span<std::byte> destination) const {
        std::size_t copied = 0;
        for (const auto& piece : segments(logical_offset, destination.size())) {
            const auto read = read_volume(piece.volume, piece.file_offset,
                                          destination.subspan(copied, piece.bytes));
            copied += read;
            if (read < piece.bytes) { break; }
        }
        return copied;
    }

private:
    struct Volume {
        const MappedFile* file      = nullptr;
        std::uint64_t payload_bytes = 0;
        std::uint64_t logical_begin = 0;
        std::uint64_t file_base     = 0;
    };

    // One piece of a logical range: the volume that holds it, the byte offset inside that
    // volume's file, and the piece's length.
    struct Piece {
        std::size_t volume = 0;
        std::uint64_t file_offset = 0;
        std::uint64_t logical_offset = 0;
        std::uint64_t bytes = 0;
    };

    // The volume-contiguous pieces of a logical range. Upstream cuts the logical payload space
    // into volumes by byte budget, so an object may straddle a boundary.
    std::vector<Piece> segments(std::uint64_t logical_offset, std::uint64_t bytes) const {
        if (bytes == 0) { return {}; }
        if (logical_offset > total_payload_) {
            throw ArtifactError("read range exceeds the logical payload");
        }
        // A request may reach past the payload: the caller rounds its length up to a block
        // boundary, and the bytes past the end are slack the caller never reads.
        const auto available = std::min(bytes, total_payload_ - logical_offset);
        std::vector<Piece> result;
        std::uint64_t copied = 0;
        while (copied < available) {
            const std::size_t index = volume_index_for(logical_offset + copied);
            const auto& volume = volumes_[index];
            const auto local   = logical_offset + copied - volume.logical_begin;
            const auto count   = std::min(available - copied, volume.payload_bytes - local);
            result.push_back(Piece{index, volume.file_base + local, logical_offset + copied, count});
            copied += count;
        }
        return result;
    }

    // Reads one volume's bytes. A request whose file offset, length and destination are all
    // 4096-aligned goes straight to the direct-I/O path, which the kernel serves in whole blocks.
    // Anything else is served by the mapping: the direct-I/O path refuses an unaligned request.
    std::size_t read_volume(std::size_t volume_index, std::uint64_t file_offset,
                            std::span<std::byte> destination) const {
        const auto& file = *volumes_[volume_index].file;
        const auto end   = file_offset + destination.size();
        if (end > file.size()) { throw ArtifactError("read range exceeds the volume file"); }
        const bool aligned =
            file_offset % kPayloadAlignment == 0 && destination.size() % kPayloadAlignment == 0 &&
            reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment == 0;
        if (aligned) {
            return file.read_direct(file_offset, destination);
        }
        std::memcpy(destination.data(), file.data() + file_offset, destination.size());
        return destination.size();
    }

    void resolve_identity(const Json& directory) {
        if (!directory.is_object() || !directory.contains("metadata") ||
            !directory.at("metadata").is_object() ||
            !directory.at("metadata").contains("name") ||
            !directory.at("metadata").at("name").is_string()) {
            throw ArtifactError("NInfer v3 directory is missing its metadata");
        }
        const auto metadata_name = directory.at("metadata").at("name").get<std::string>();
        const auto* identity      = find_identity(metadata_name);
        if (identity == nullptr) {
            throw ArtifactError("unsupported NInfer v3 artifact name: " + metadata_name +
                                " (supported: qwen3.8-flash-next)");
        }
        identity_.model_id   = identity->model_id;
        identity_.weights_id = identity->weights_id;
    }

    void open_volumes(const std::filesystem::path& entry_path, const Json& directory,
                      std::uint64_t entry_payload_start) {
        if (!directory.contains("files") || !directory.at("files").is_array() ||
            directory.at("files").empty()) {
            throw ArtifactError("NInfer v3 directory is missing its file list");
        }
        if (!directory.contains("bindings") || !directory.at("bindings").is_object() ||
            !directory.contains("uses") || !directory.at("uses").is_array() ||
            !directory.contains("components") || !directory.at("components").is_object()) {
            throw ArtifactError("NInfer v3 directory is incomplete");
        }
        const auto& files = directory.at("files");

        const auto& entry_record = files[0];
        if (!entry_record.is_object() || !entry_record.contains("path") ||
            !entry_record.at("path").is_null()) {
            throw ArtifactError("NInfer v3 files[0].path must be null");
        }
        const auto entry_payload_bytes = require_v3_unsigned(
            require_member(entry_record, "payload_bytes", "v3 file record"),
            "v3 payload_bytes", true);
        if (checked_add(entry_payload_start, entry_payload_bytes, "v3 entry length") !=
            static_cast<std::uint64_t>(entry_.size())) {
            throw ArtifactError("NInfer v3 entry length differs from its directory");
        }
        std::array<std::byte, 16> artifact_id{};
        std::copy(entry_.data() + 16, entry_.data() + kV3HeaderBytes, artifact_id.begin());

        volumes_.push_back(Volume{&entry_, entry_payload_bytes, 0, entry_payload_start});
        std::uint64_t logical_begin = entry_payload_bytes;
        std::uint64_t total_bytes   = entry_.size();
        for (std::size_t index = 1; index < files.size(); ++index) {
            const auto& record = files[index];
            const auto path     = require_string(
                require_member(record, "path", "v3 file record"), "v3 continuation path");
            if (path == "." || path == ".." || path.find('/') != std::string::npos ||
                path.find('\\') != std::string::npos) {
                throw ArtifactError("NInfer v3 continuation path is not a sibling file: " + path);
            }
            const auto payload_bytes = require_v3_unsigned(
                require_member(record, "payload_bytes", "v3 file record"),
                "v3 payload_bytes", true);
            auto file = std::make_unique<MappedFile>(entry_path.parent_path() / path);
            if (file->size() < kV3HeaderBytes) {
                throw ArtifactError(path + ": continuation header is truncated");
            }
            if (!std::equal(kV3PartMagic.begin(), kV3PartMagic.end(), file->data())) {
                throw ArtifactError(path + ": continuation magic is not NInfer v3");
            }
            const auto part_index = read_u64_le(file->data() + 8);
            if (part_index != index) {
                throw ArtifactError(path + ": continuation declares part index " +
                                    std::to_string(part_index) + " but the directory expects " +
                                    std::to_string(index));
            }
            if (!std::equal(artifact_id.begin(), artifact_id.end(), file->data() + 16)) {
                throw ArtifactError(path + ": continuation belongs to a different artifact");
            }
            if (file->size() !=
                checked_add(kPayloadAlignment, payload_bytes, "v3 continuation length")) {
                throw ArtifactError(path + ": continuation length differs from its directory");
            }
            volumes_.push_back(Volume{file.get(), payload_bytes, logical_begin, kPayloadAlignment});
            owned_.push_back(std::move(file));
            logical_begin += payload_bytes;
            total_bytes += static_cast<std::uint64_t>(owned_.back()->size());
        }
        file_bytes_     = total_bytes;
        total_payload_  = logical_begin;
    }

    void parse_objects(const Json& directory) {
        if (!directory.contains("objects") || !directory.at("objects").is_array() ||
            directory.at("objects").empty()) {
            throw ArtifactError("NInfer v3 directory has no objects");
        }
        std::uint64_t cursor = 0;
        for (const auto& raw_object : directory.at("objects")) {
            auto object = parse_v3_object(raw_object);
            const auto name   = object_name(object);
            const auto offset = object_offset(object);
            const auto bytes  = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);
            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > total_payload_) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the payload");
            }
            const auto object_index = objects_.size();
            auto [_, inserted]      = index_.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            objects_.push_back(std::move(object));
            cursor = end;
        }
    }

    static const V3Identity* find_identity(std::string_view metadata_name) noexcept {
        for (const auto& identity : kContainerV3Identities) {
            if (identity.metadata_name == metadata_name) { return &identity; }
        }
        return nullptr;
    }

    static ObjectDescriptor parse_v3_object(const Json& value) {
        if (!value.is_object()) {
            throw ArtifactError("each v3 object entry must be a JSON object");
        }
        const auto id = require_string(require_member(value, "id", "v3 object entry"),
                                       "v3 object id");
        const auto& kind = require_string(require_member(value, "kind", "v3 object entry"),
                                          "v3 object kind");
        const auto offset = require_v3_unsigned(require_member(value, "offset", "v3 object entry"),
                                                 "v3 object offset", false);
        const auto bytes  = require_v3_unsigned(require_member(value, "bytes", "v3 object entry"),
                                                "v3 object bytes", true);
        if (kind == "tensor") {
            const auto format = parse_v3_format(require_string(
                require_member(value, "format", "v3 tensor entry"), "v3 tensor format"));
            const auto layout = parse_v3_layout(require_string(
                require_member(value, "layout", "v3 tensor entry"), "v3 tensor layout"));
            const auto& raw_shape = require_member(value, "shape", "v3 tensor entry");
            if (!raw_shape.is_array()) { throw ArtifactError("v3 tensor shape must be an array"); }
            std::vector<std::uint64_t> shape;
            shape.reserve(raw_shape.size());
            for (const auto& dimension : raw_shape) {
                shape.push_back(require_v3_unsigned(dimension, "v3 shape dimension", true));
            }
            const auto expected = tensor_encoded_size(layout, format, shape, bytes);
            if (bytes != expected) {
                throw ArtifactError("tensor " + id + " stores " + std::to_string(bytes) +
                                    " bytes; layout requires " + std::to_string(expected));
            }
            return TensorDescriptor{id, std::move(shape), format, layout, offset, bytes};
        }
        if (kind == "resource") {
            return ResourceDescriptor{
                id,
                parse_v3_encoding(
                    require_string(require_member(value, "encoding", "v3 resource entry"),
                                   "v3 resource encoding")),
                offset,
                bytes,
            };
        }
        throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
    }

    std::size_t volume_index_for(std::uint64_t logical_offset) const {
        for (std::size_t index = 0; index < volumes_.size(); ++index) {
            const auto& volume = volumes_[index];
            if (logical_offset < volume.logical_begin + volume.payload_bytes) { return index; }
        }
        throw ArtifactError("v3 payload offset is beyond the artifact payload");
    }

    const MappedFile& entry_;
    std::vector<std::unique_ptr<MappedFile>> owned_;
    std::vector<Volume> volumes_;
    ArtifactIdentity identity_;
    V3Directory directory_;
    std::vector<ObjectDescriptor> objects_;
    ObjectIndex index_;
    std::uint64_t file_bytes_    = 0;
    std::uint64_t total_payload_ = 0;
};

} // namespace

// The upstream v3 container spells its numeric formats, storage layouts and resource encodings in
// lower case snake case; the v2 container spells the same concepts differently. The two spellings
// are separate contracts that denote the same enum values, so the v3 reader keeps its own table
// instead of widening the v2 one. The table accepts every spelling an upstream v3 artifact uses,
// so it is a superset of the 27B narrow adapter's set: one spelling must not be accepted in one
// place and rejected in the other.
NumericFormat parse_v3_format(std::string_view name) {
    if (name == "bf16") { return NumericFormat::BF16; }
    if (name == "fp16") { return NumericFormat::FP16; }
    if (name == "fp32") { return NumericFormat::FP32; }
    if (name == "int32") { return NumericFormat::I32; }
    if (name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "nf4_g16_e4m3fn") { return NumericFormat::NF4_G16_E4M3S; }
    if (name == "fp8_e4m3fn_block128x128_bf16") { return NumericFormat::FP8_E4M3FN_BLOCK128_BF16S; }
    if (name == "fp8_e4m3fn_row_bf16") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown v3 tensor format: " + std::string(name));
}

StorageLayout parse_v3_layout(std::string_view name) {
    if (name == "contiguous_le_v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row_split_k128_v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    if (name == "row_paged_g16_4k_v1") { return StorageLayout::RowPagedG16x4KV1; }
    if (name == "block_scale_k16_m128x4_v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "batched_block_scale_k16_m128x4_v1") {
        return StorageLayout::BatchedBlockScaleK16M128x4V1;
    }
    if (name == "block_scale_m128_k128_v1") { return StorageLayout::BlockScaleM128K128V1; }
    throw ArtifactError("unknown v3 tensor layout: " + std::string(name));
}

ResourceEncoding parse_v3_encoding(std::string_view name) {
    if (name == "raw_bytes_v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown v3 resource encoding: " + std::string(name));
}

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the NInfer prefix");
        }
        if (std::equal(kV1Magic.begin(), kV1Magic.end(), file.data())) {
            throw ArtifactError("NInfer artifact v1 is no longer supported; migrate it with: "
                                "python3 -m tools.artifact.migrate_v1_to_v2 <artifact>");
        }
        const bool v2 = std::equal(kMagic.begin(), kMagic.end(), file.data());
        const bool v3 = std::equal(kV3Magic.begin(), kV3Magic.end(), file.data());
        if (!v2 && !v3) { throw ArtifactError("artifact magic is not NInfer v2 or supported v3"); }
        const std::uint64_t header_bytes = v3 ? kV3HeaderBytes : kPrefixBytes;
        if (file.size() < header_bytes) { throw ArtifactError("artifact header is truncated"); }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(header_bytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + header_bytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        std::map<std::string, std::vector<std::byte>, std::less<>> named_overrides;
        if (v3) {
            // A v3 artifact that names an identity this reader serves directly is read by the
            // container path. Every other v3 artifact is the 27B package's single-file artifact,
            // which the narrow adapter projects and rejects when it is not one.
            if (V3MultiPartReader::recognizes(v3_metadata_name(directory))) {
                multipart =
                    std::make_unique<V3MultiPartReader>(path, file, directory, payload_start);
                identity  = multipart->identity();
                return;
            }
            auto compatible = Qwen38Nvfp4V3Adapter(directory, file, payload_start).build();
            directory       = std::move(compatible.directory);
            named_overrides = std::move(compatible.payload_overrides);
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            if (const auto* tensor = std::get_if<TensorDescriptor>(&object);
                tensor != nullptr && tensor->format == NumericFormat::GGML_K) {
                validate_ggml_k_payload(tensor->shape,
                    std::span<const std::byte>(file.data() + payload_start + offset, bytes));
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            if (auto override = named_overrides.find(std::string(name));
                override != named_overrides.end()) {
                payload_overrides.emplace(std::string(name), std::move(override->second));
            }
            cursor = end;
        }
        if (payload_overrides.size() != named_overrides.size()) {
            throw ArtifactError("v3 payload override does not name a projected object");
        }
    }

    MappedFile file;
    std::unique_ptr<V3MultiPartReader> multipart;
    ArtifactIdentity identity;
    V3Directory directory;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::unordered_map<std::string, std::vector<std::byte>, TransparentStringHash, std::equal_to<>>
        payload_overrides;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept {
    return impl_->multipart ? impl_->multipart->objects() : impl_->entries;
}

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    if (impl_->multipart) { return impl_->multipart->find(name); }
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

// The v3 container path reads the directory metadata; the v2 path and the 27B narrow adapter
// carry none, so they report an empty directory.
const V3Directory& Reader::v3_directory() const noexcept {
    if (impl_->multipart) { return impl_->multipart->directory(); }
    return impl_->directory;
}

std::uint64_t Reader::file_bytes() const noexcept {
    return impl_->multipart ? impl_->multipart->file_bytes() : impl_->file.size();
}

// The payload coordinate space of a multi-volume artifact is the concatenation of its volume
// payloads, so its origin is zero; a single-file artifact's payload space is the file's byte
// space, whose origin is the payload start.
std::uint64_t Reader::payload_offset() const noexcept {
    return impl_->multipart ? 0 : impl_->payload_start;
}

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    if (impl_->multipart) { return impl_->multipart->payload(object); }
    if (const auto override = impl_->payload_overrides.find(object_name(object));
        override != impl_->payload_overrides.end()) {
        return {0, override->second};
    }
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::vector<PayloadSegment> Reader::payload_segments(std::uint64_t offset,
                                                      std::uint64_t bytes) const {
    if (impl_->multipart) { return impl_->multipart->payload_segments(offset, bytes); }
    const auto end = checked_add(offset, bytes, "payload range");
    if (end > impl_->file.size()) { throw ArtifactError("read range exceeds the file"); }
    if (bytes == 0) { return {}; }
    return {PayloadSegment{offset, bytes}};
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    if (impl_->multipart) { return impl_->multipart->read_direct(absolute_offset, destination); }
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
