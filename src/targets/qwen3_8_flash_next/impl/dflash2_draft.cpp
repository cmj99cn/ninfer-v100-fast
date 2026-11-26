#include "targets/qwen3_8_flash_next/impl/dflash2_draft.h"

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dflash2_attn_input.h"
#include "ninfer/ops/dflash2_linear_swiglu.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using artifact::ArtifactError;
using artifact::NumericFormat;
using artifact::StorageLayout;

constexpr std::string_view kComponent = "dflash2";
constexpr std::string_view kPrefix    = "dflash2/";

std::string_view require_value(const std::map<std::string, std::string>& config,
                               std::string_view key) {
    const auto found = config.find(std::string(key));
    if (found == config.end()) {
        throw ArtifactError("DFlash2 companion config is missing '" + std::string(key) + "'");
    }
    return found->second;
}

std::uint64_t parse_integer(std::string_view text, std::string_view key) {
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw ArtifactError("DFlash2 companion config key '" + std::string(key) +
                            "' is not an unsigned integer: " + std::string(text));
    }
    return value;
}

std::uint64_t require_integer(const std::map<std::string, std::string>& config,
                              std::string_view key, std::uint64_t expected) {
    const auto value = parse_integer(require_value(config, key), key);
    if (value != expected) {
        throw ArtifactError("DFlash2 companion config key '" + std::string(key) + "' is " +
                            std::to_string(value) + "; this package expects " +
                            std::to_string(expected));
    }
    return value;
}

std::vector<std::uint32_t> parse_u32_list(std::string_view text, std::string_view key) {
    std::vector<std::uint32_t> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string_view::npos ? end : end - begin);
        if (part.empty()) {
            throw ArtifactError("DFlash2 companion config key '" + std::string(key) +
                                "' has an empty element");
        }
        const auto value = parse_integer(part, key);
        if (value > 0xFFFFFFFFULL) {
            throw ArtifactError("DFlash2 companion config key '" + std::string(key) +
                                "' has an element outside uint32");
        }
        values.push_back(static_cast<std::uint32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return values;
}

// The companion's quantized matrices bind the target vocabulary of projections the drafter
// actually runs. Deriving each row from the companion's own dimensions keeps the checkpoint's
// 12800-wide feature projection (five tapped target blocks) distinct from its 2560-wide hidden.
std::vector<std::uint64_t> expected_quantized_shape(std::string_view logical, std::uint64_t hidden,
                                                    std::uint64_t taps,
                                                    std::uint64_t intermediate,
                                                    std::uint64_t query_width,
                                                    std::uint64_t key_width) {
    if (logical == "dflash2/feature_projection") { return {hidden, taps * hidden}; }
    const auto ends_with = [&](std::string_view suffix) { return logical.ends_with(suffix); };
    if (ends_with("/attention/query") || ends_with("/attention/key") ||
        ends_with("/attention/value") || ends_with("/attention/context_key") ||
        ends_with("/attention/context_value")) {
        return {query_width + 2 * key_width, hidden};
    }
    if (ends_with("/attention/output")) { return {hidden, query_width}; }
    if (ends_with("/mlp/gate") || ends_with("/mlp/up")) { return {2 * intermediate, hidden}; }
    if (ends_with("/mlp/down")) { return {hidden, intermediate}; }
    throw ArtifactError("DFlash2 quantized object " + std::string(logical) +
                        " has no registered geometry");
}

std::map<std::string, std::string> sub_config(const std::map<std::string, std::string>& config,
                                              std::string_view prefix) {
    std::map<std::string, std::string> out;
    for (const auto& [key, value] : config) {
        if (key.starts_with(prefix)) { out.emplace(key.substr(prefix.size()), value); }
    }
    return out;
}

void validate_dflash2_draft_config(const std::map<std::string, std::string>& draft_config,
                                   const std::vector<std::uint32_t>& target_layer_ids) {
    require_integer(draft_config, "mask_token_id", 248077);
    require_integer(draft_config, "conv_kernel_size", 2);
    require_integer(draft_config, "conv_group_size", 16);
    require_integer(draft_config, "selector_rank", 256);
    require_integer(draft_config, "selector_top_k", 16);
    if (target_layer_ids != std::vector<std::uint32_t>({5, 15, 25, 35, 45})) {
        throw ArtifactError("DFlash2 companion 'dflash_config.target_layer_ids' is not [5,15,25,35,45]");
    }
}

} // namespace

void validate_dflash2_target_config(const std::map<std::string, std::string>& config) {
    const auto architecture = require_value(config, "architectures");
    if (architecture != "Qwen4ExpForCausalLM") {
        throw ArtifactError("DFlash2 companion target_config 'architectures' is '" +
                            std::string(architecture) + "'; the main artifact is "
                            "'Qwen4ExpForCausalLM'");
    }
    const auto hidden = require_value(config, "hidden_size");
    if (hidden != "2560") {
        throw ArtifactError("DFlash2 companion target_config 'hidden_size' is " + std::string(hidden) +
                            "; the main artifact uses 2560");
    }
    const auto vocab = require_value(config, "vocab_size");
    if (vocab != "248320") {
        throw ArtifactError("DFlash2 companion target_config 'vocab_size' is " + std::string(vocab) +
                            "; the main artifact uses 248320");
    }
    require_integer(config, "num_hidden_layers", 48);
}

DFlash2Draft::DFlash2Draft(const std::filesystem::path& path, DFlash2Placement placement)
    : reader_(path) {
    const auto& identity = reader_.identity();
    if (identity.model_id != "qwen3.8-flash-next-dflash2" || identity.weights_id != "dflash2") {
        throw ArtifactError("the artifact is not a DFlash2 companion (got " + identity.model_id +
                            "/" + identity.weights_id + ")");
    }
    const auto& directory = reader_.v3_directory();
    const auto component = directory.component_config.find(std::string(kComponent));
    if (component == directory.component_config.end()) {
        throw ArtifactError("the DFlash2 companion carries no 'dflash2' component config");
    }
    const auto& config       = component->second;
    const auto target_config = sub_config(config, "target_config.");
    const auto draft_config  = sub_config(config, "dflash_config.");
    validate_dflash2_target_config(target_config);

    target_layer_ids_ = parse_u32_list(require_value(draft_config, "target_layer_ids"),
                                       "dflash_config.target_layer_ids");
    validate_dflash2_draft_config(draft_config, target_layer_ids_);
    drafter_layers_ = static_cast<std::uint32_t>(require_integer(config, "num_hidden_layers", 5));
    const auto intermediate = require_integer(config, "intermediate_size", 7680);
    const auto heads        = require_integer(config, "num_attention_heads", 32);
    const auto kv_heads     = require_integer(config, "num_key_value_heads", 8);
    const auto head_dim     = require_integer(config, "head_dim", 128);
    require_integer(config, "sliding_window", 2048);

    target_.architecture = std::string(require_value(target_config, "architectures"));
    target_.hidden = static_cast<std::uint32_t>(parse_integer(require_value(target_config, "hidden_size"),
                                                              "target_config.hidden_size"));
    target_.vocab  = static_cast<std::uint32_t>(parse_integer(require_value(target_config, "vocab_size"),
                                                             "target_config.vocab_size"));

    const std::uint64_t hidden      = target_.hidden;
    const std::uint64_t taps        = target_layer_ids_.size();
    const std::uint64_t query_width = heads * head_dim;
    const std::uint64_t key_width   = kv_heads * head_dim;

    artifact::Binder binder(reader_);
    std::set<std::string> bound_objects;
    const auto bind_object = [&](const std::string& object, const std::string& logical) {
        if (!bound_objects.insert(object).second) { return; }
        const auto* descriptor = reader_.find(object);
        if (descriptor == nullptr) {
            throw ArtifactError("DFlash2 binding '" + logical + "' names a missing object " + object);
        }
        const auto* tensor = std::get_if<artifact::TensorDescriptor>(descriptor);
        if (tensor == nullptr) {
            throw ArtifactError("DFlash2 binding '" + logical + "' does not name a tensor");
        }
        if (tensor->format != NumericFormat::W8G32_F16S && tensor->format != NumericFormat::BF16) {
            throw ArtifactError("DFlash2 object " + object + " uses an unexpected numeric format");
        }
        if (tensor->format == NumericFormat::W8G32_F16S) {
            if (tensor->layout != StorageLayout::RowSplitK128V1 || tensor->shape.size() != 2) {
                throw ArtifactError("DFlash2 quantized object " + object +
                                    " does not have a row-split recipe");
            }
            const auto expected = expected_quantized_shape(logical, hidden, taps, intermediate,
                                                           query_width, key_width);
            if (tensor->shape != expected) {
                throw ArtifactError("DFlash2 quantized object " + object +
                                    " does not have the registered DFlash2 geometry");
            }
        } else if (tensor->layout != StorageLayout::ContiguousLeV1) {
            throw ArtifactError("DFlash2 BF16 object " + object + " is not contiguous");
        }
        const auto handle = binder.require_tensor(object, tensor->format, tensor->layout, tensor->shape);
        if (placement == DFlash2Placement::Device) {
            binder.materialize_on_device(handle);
        } else {
            binder.validate_only(handle);
        }
        tensors_.push_back(BoundDFlash2Tensor{logical, object, handle, tensor->format,
                                              tensor->layout, tensor->shape});
    };

    for (const auto& [logical, object] : directory.bindings) {
        if (logical.starts_with(kPrefix)) { bind_object(object, logical); }
    }
    for (const auto& [logical, parts] : directory.part_bindings) {
        if (!logical.starts_with(kPrefix)) { continue; }
        for (const auto& part : parts) { bind_object(part.object, logical); }
    }
    if (tensors_.size() != reader_.objects().size()) {
        throw ArtifactError("DFlash2 companion bound " + std::to_string(tensors_.size()) + " of " +
                            std::to_string(reader_.objects().size()) + " objects");
    }
    const std::vector<std::uint64_t> feature_shape{hidden, taps * hidden};
    const auto feature = std::find_if(tensors_.begin(), tensors_.end(), [](const auto& tensor) {
        return tensor.name == "dflash2/feature_projection";
    });
    if (feature == tensors_.end() || feature->shape != feature_shape) {
        throw ArtifactError("DFlash2 feature_projection does not have the registered geometry");
    }
    plan_ = binder.finish();
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::int32_t kHidden        = DFlash2Geometry::hidden;
constexpr std::int32_t kIntermediate  = DFlash2Geometry::intermediate;
constexpr std::int32_t kQueryWidth    = DFlash2Geometry::query_width;
constexpr std::int32_t kKeyWidth      = DFlash2Geometry::key_width;
constexpr std::int32_t kHeadDim       = DFlash2Geometry::head_dim;
constexpr std::int32_t kQueryHeads    = DFlash2Geometry::query_heads;
constexpr std::int32_t kKvHeads       = DFlash2Geometry::kv_heads;
constexpr std::int32_t kGroups        = DFlash2Geometry::conv_groups;
constexpr std::int32_t kRank          = DFlash2Geometry::selector_rank;
constexpr std::int32_t kCandidates    = DFlash2Geometry::candidates;
constexpr std::uint32_t kWindow       = DFlash2Geometry::window;
constexpr std::size_t kSlack          = 1U << 20;

std::size_t plane(std::uint64_t rows, std::uint64_t columns, std::size_t element) {
    return rows * columns * element + 256;
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("DFlash2 drafter: " + message); }
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n) {
        throw std::logic_error("DFlash2 drafter: invalid weight row view");
    }
    if (block.layout == QuantLayout::Contiguous && block.qtype == QType::BF16_CTRL) {
        Weight out = block;
        const auto offset =
            static_cast<std::size_t>(row_begin) * block.k * sizeof(std::uint16_t);
        out.qdata         = static_cast<const std::byte*>(block.qdata) + offset;
        out.payload       = out.qdata;
        out.payload_bytes = static_cast<std::uint64_t>(row_count) * block.k * sizeof(std::uint16_t);
        out.n = out.shape[0] = out.padded_shape[0] = row_count;
        return out;
    }
    if (block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("DFlash2 drafter: row view needs a row-split weight");
    }
    const std::uint64_t groups   = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_row  = groups * static_cast<std::uint64_t>(block.group);
    const std::uint64_t scale_row = groups * 2;
    Weight out = block;
    out.qdata  = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

const BoundDFlash2Tensor& bound_tensor(const DFlash2Draft& draft, std::string_view name) {
    for (const auto& tensor : draft.tensors()) {
        if (tensor.name == name) { return tensor; }
    }
    throw ArtifactError("DFlash2 companion has no bound tensor '" + std::string(name) + "'");
}

// A fused parent object is recorded under whichever alias the companion's sorted binding table
// visits first (the query/key/value/context aliases all name one object, as do gate/up).
const BoundDFlash2Tensor& bound_tensor_any(
    const DFlash2Draft& draft, std::initializer_list<std::string_view> names) {
    for (const auto& tensor : draft.tensors()) {
        for (const auto name : names) {
            if (tensor.name == name) { return tensor; }
        }
    }
    throw ArtifactError("DFlash2 companion has no bound fused parent tensor");
}

Weight bound_weight(const DFlash2Draft& draft, const artifact::MaterializedArtifact& materialized,
                    std::string_view name, std::int32_t rows, std::int32_t columns, int device) {
    const auto& tensor = bound_tensor(draft, name);
    return artifact::materialized_weight(materialized, tensor.handle, tensor.format, rows, columns,
                                         device);
}

Weight bound_weight_any(const DFlash2Draft& draft,
                        const artifact::MaterializedArtifact& materialized,
                        std::initializer_list<std::string_view> names, std::int32_t rows,
                        std::int32_t columns, int device) {
    const auto& tensor = bound_tensor_any(draft, names);
    return artifact::materialized_weight(materialized, tensor.handle, tensor.format, rows, columns,
                                         device);
}

Tensor bound_dense(const DFlash2Draft& draft, const artifact::MaterializedArtifact& materialized,
                   std::string_view name, std::initializer_list<std::int32_t> shape, int device) {
    const auto& tensor = bound_tensor(draft, name);
    return artifact::materialized_tensor(materialized, tensor.handle, tensor.format, shape, device);
}

} // namespace

DFlash2Operands bind_dflash2_operands(const DFlash2Draft& draft,
                                      const artifact::MaterializedArtifact& materialized,
                                      int device) {
    DFlash2Operands out;
    out.feature_projection = bound_weight(draft, materialized, "dflash2/feature_projection",
                                          kHidden, DFlash2Geometry::feature_rows, device);
    out.context_norm =
        bound_dense(draft, materialized, "dflash2/context_norm", {kHidden}, device);
    out.final_norm =
        bound_dense(draft, materialized, "dflash2/final_norm", {kHidden}, device);
    out.selector_projection = bound_weight(draft, materialized,
                                          "dflash2/candidate_selector/hidden_projection", kRank,
                                          kHidden, device);
    out.predecessor_codebook = bound_dense(
        draft, materialized, "dflash2/candidate_selector/predecessor_codebook",
        {static_cast<std::int32_t>(draft.vocab()), kRank}, device);
    out.successor_codebook = bound_dense(
        draft, materialized, "dflash2/candidate_selector/successor_codebook",
        {static_cast<std::int32_t>(draft.vocab()), kRank}, device);

    out.layers.resize(DFlash2Geometry::layers);
    for (std::int32_t layer = 0; layer < DFlash2Geometry::layers; ++layer) {
        const std::string prefix = "dflash2/layers/" + std::to_string(layer) + "/";
        auto& target             = out.layers[static_cast<std::size_t>(layer)];
        target.input_norm =
            bound_dense(draft, materialized, prefix + "input_norm", {kHidden}, device);
        target.post_attention_norm =
            bound_dense(draft, materialized, prefix + "post_attention_norm", {kHidden}, device);
        target.query_norm =
            bound_dense(draft, materialized, prefix + "attention/query_norm", {kHeadDim}, device);
        target.key_norm =
            bound_dense(draft, materialized, prefix + "attention/key_norm", {kHeadDim}, device);
        target.attention_base = bound_dense(draft, materialized,
                                            prefix + "attention_conv/base_kernel",
                                            {kHidden, 2, 2}, device);
        target.mlp_base = bound_dense(draft, materialized, prefix + "mlp_conv/base_kernel",
                                      {kHidden, 2, 2}, device);
        target.query_key_value = bound_weight_any(
            draft, materialized,
            {prefix + "attention/query", prefix + "attention/context_key",
             prefix + "attention/context_value", prefix + "attention/key",
             prefix + "attention/value"},
            kQueryWidth + 2 * kKeyWidth, kHidden, device);
        target.context_key = row_view(target.query_key_value, kQueryWidth, kKeyWidth);
        target.context_value =
            row_view(target.query_key_value, kQueryWidth + kKeyWidth, kKeyWidth);
        target.attention_output = bound_weight(draft, materialized, prefix + "attention/output",
                                               kHidden, kQueryWidth, device);
        target.gate_up = bound_weight_any(draft, materialized,
                                          {prefix + "mlp/gate", prefix + "mlp/up"},
                                          2 * kIntermediate, kHidden, device);
        target.down = bound_weight(draft, materialized, prefix + "mlp/down", kHidden,
                                   kIntermediate, device);
        target.attention_conv_kernel_projection = bound_weight(
            draft, materialized, prefix + "attention_conv/kernel_projection", 4 * kGroups, kHidden,
            device);
        target.mlp_conv_kernel_projection = bound_weight(
            draft, materialized, prefix + "mlp_conv/kernel_projection", 4 * kGroups, kHidden,
            device);
    }
    return out;
}

DFlash2Drafter::DFlash2Drafter(DFlash2Operands operands, std::int32_t vocab, std::int32_t domain,
                               std::vector<CyclicKVCacheLayerView> ring,
                               std::uint32_t max_context, DFlash2Head head)
    : p_(std::move(operands)), vocab_(vocab), domain_(domain), ring_(std::move(ring)),
      max_context_(max_context), head_(std::move(head)) {
    require(static_cast<std::int32_t>(p_.layers.size()) == DFlash2Geometry::layers,
            "one operand set per drafter layer");
    require(static_cast<std::int32_t>(ring_.size()) == DFlash2Geometry::layers,
            "one context ring layer per drafter layer");
    for (const auto& layer : ring_) {
        require(layer.capacity == kWindow, "a drafter ring capacity");
    }
    require(vocab_ > 0 && domain_ >= kCandidates && domain_ <= vocab_ && head_ != nullptr,
            "the target's head, vocabulary and selector domain");
}

void DFlash2Drafter::materialize(const DFlash2ContextCall& call, WorkspaceArena& work,
                                 cudaStream_t stream) const {
    const std::int32_t w = call.features.ne[1];
    const std::int32_t b = call.features.ne[2];
    require(call.features.dtype == DType::BF16 &&
                call.features.ne[0] == DFlash2Geometry::feature_rows && w > 0 && b > 0 &&
                b <= 8 && call.max_count <= static_cast<std::uint32_t>(w) &&
                call.min_count <= call.max_count,
            "the materialize call's geometry");
    require(call.positions.dtype == DType::I32 && call.positions.ne[0] == w &&
                call.positions.ne[1] == b && call.counts.dtype == DType::I32 &&
                call.counts.ne[0] == b && call.lanes.dtype == DType::I32 &&
                call.lanes.ne[0] == b,
            "the materialize call's positions/counts/lanes");

    auto scope = work.scope();
    // A single row wider than the ring keeps only its last window of columns; every later block
    // still sees that window in full. Multi-row calls are chunked by the schedule instead.
    std::int32_t first = 0, width = w;
    Tensor counts = call.counts;
    ops::ContextKVMaterializeExecutionEnvelope envelope{call.min_count, call.max_count};
    if (b == 1 && call.max_count > kWindow) {
        require(call.min_count == call.max_count,
                "an oversized materialize commit must be one row of known count");
        first    = width - static_cast<std::int32_t>(kWindow);
        width    = static_cast<std::int32_t>(kWindow);
        counts   = work.alloc(DType::I32, {1});
        ops::set_i32_scalar(counts, width, stream);
        envelope = {kWindow, kWindow};
    }
    const std::int32_t columns = width * b;
    const Tensor features =
        width == w ? call.features : call.features.slice(1, first, width);
    const Tensor positions =
        width == w ? call.positions : call.positions.slice(0, first, width);

    Tensor projected = work.alloc(DType::BF16, {kHidden, columns});
    ops::linear(features.view({DFlash2Geometry::feature_rows, columns}),
                p_.feature_projection, projected, ops::LinearPolicy::A16Only, work, stream);
    Tensor context = work.alloc(DType::BF16, {kHidden, columns});
    ops::rmsnorm(projected, p_.context_norm, DFlash2Geometry::rms_epsilon, false, context, stream);

    std::array<ops::ContextKVMaterializeLayerView, DFlash2Geometry::layers> layers{};
    for (std::int32_t layer = 0; layer < DFlash2Geometry::layers; ++layer) {
        const auto& source = p_.layers[static_cast<std::size_t>(layer)];
        layers[static_cast<std::size_t>(layer)] = {
            .key_weight      = source.context_key,
            .value_weight    = source.context_value,
            .key_norm_weight = source.key_norm,
            .cache           = ring_[static_cast<std::size_t>(layer)],
        };
    }
    ops::context_kv_materialize(context, positions, counts, call.lanes, layers,
                                DFlash2Geometry::rope_theta, envelope, work, stream);
}

void DFlash2Drafter::propose(const DFlash2ProposalCall& call, WorkspaceArena& work,
                             cudaStream_t stream) const {
    const std::int32_t w          = call.width;
    const std::int32_t b          = call.batch;
    const std::int32_t k          = w - 1;
    const std::int32_t columns    = w * b;
    const std::int32_t masks      = k * b;
    require(w >= 2 && w <= 16 && b >= 1 && b <= 8 && call.configs != nullptr,
            "the proposal's geometry");
    require(call.ids.dtype == DType::I32 && call.positions.dtype == DType::I32 &&
                call.valid_columns.dtype == DType::I32 && call.lanes.dtype == DType::I32,
            "the proposal's control operands");
    require(masks <= 15 * 8, "the proposal's selector steps");

    auto scope       = work.scope();
    Tensor ids       = call.ids;
    Tensor positions = call.positions;
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns,
                              DFlash2Geometry::mask_token, ids, positions, stream);
    const Tensor flat_positions = positions.view({columns});

    Tensor residual = work.alloc(DType::BF16, {kHidden, w, b});
    {
        Tensor flat = residual.view({kHidden, columns});
        ops::embedding(ids.view({columns}), p_.embedding, flat, stream);
    }
    const ops::SwaContextExecutionEnvelope envelope{0, max_context_};
    const float scale = DFlash2Geometry::attention_scale;

    for (std::int32_t layer = 0; layer < DFlash2Geometry::layers; ++layer) {
        const auto& source = p_.layers[static_cast<std::size_t>(layer)];
        {
            auto branch                               = work.scope();
            Tensor prepared                           = work.alloc(DType::BF16, {kHidden, w, b});
            Tensor finish_delta = work.alloc(DType::BF16, {kGroups, 2, w, b});
            ops::rmsnorm_dynamic_grouped_conv_prepare(
                residual, source.input_norm, DFlash2Geometry::rms_epsilon, source.attention_base,
                source.attention_conv_kernel_projection, prepared, finish_delta, work, stream);

            Tensor q_raw  = work.alloc(DType::BF16, {kQueryWidth, columns});
            Tensor k_raw  = work.alloc(DType::BF16, {kKeyWidth, columns});
            Tensor value  = work.alloc(DType::BF16, {kKeyWidth, columns});
            ops::dflash2_attn_input_proj(prepared.view({kHidden, columns}),
                                         source.query_key_value, q_raw, k_raw, value, stream);

            Tensor query      = work.alloc(DType::BF16, {kHeadDim, kQueryHeads, w, b});
            Tensor key        = work.alloc(DType::BF16, {kHeadDim, kKvHeads, w, b});
            Tensor query_rows = query.view({kHeadDim, kQueryHeads * columns});
            Tensor key_rows   = key.view({kHeadDim, kKvHeads * columns});
            ops::rmsnorm(q_raw.view({kHeadDim, kQueryHeads * columns}), source.query_norm,
                         DFlash2Geometry::rms_epsilon, false, query_rows, stream);
            ops::rmsnorm(k_raw.view({kHeadDim, kKvHeads * columns}), source.key_norm,
                         DFlash2Geometry::rms_epsilon, false, key_rows, stream);
            Tensor query_heads = query.view({kHeadDim, kQueryHeads, columns});
            Tensor key_heads   = key.view({kHeadDim, kKvHeads, columns});
            ops::rope(flat_positions, kHeadDim, DFlash2Geometry::rope_theta, query_heads,
                      key_heads, stream);

            Tensor attended = work.alloc(DType::BF16, {kHeadDim, kQueryHeads, w, b});
            ops::swa(query, key, value.view({kHeadDim, kKvHeads, w, b}), positions,
                     call.valid_columns, call.lanes, scale, ring_[static_cast<std::size_t>(layer)],
                     envelope, work, attended, stream);
            ops::linear_dynamic_grouped_conv_add(
                attended.view({kQueryWidth, w, b}), source.attention_output, source.attention_base,
                finish_delta, residual, work, stream);
        }
        {
            auto branch          = work.scope();
            Tensor prepared      = work.alloc(DType::BF16, {kHidden, w, b});
            Tensor finish_delta  = work.alloc(DType::BF16, {kGroups, 2, w, b});
            ops::rmsnorm_dynamic_grouped_conv_prepare(
                residual, source.post_attention_norm, DFlash2Geometry::rms_epsilon,
                source.mlp_base, source.mlp_conv_kernel_projection, prepared, finish_delta, work,
                stream);
            Tensor product = work.alloc(DType::BF16, {kIntermediate, columns});
            ops::dflash2_linear_swiglu(prepared.view({kHidden, columns}), source.gate_up, product,
                                      work, stream);
            ops::linear_dynamic_grouped_conv_add(product.view({kIntermediate, w, b}), source.down,
                                                 source.mlp_base, finish_delta, residual, work,
                                                 stream);
        }
    }

    Tensor normed      = work.alloc(DType::BF16, {kHidden, w, b});
    Tensor normed_flat = normed.view({kHidden, columns});
    ops::rmsnorm(residual.view({kHidden, columns}), p_.final_norm,
                 DFlash2Geometry::rms_epsilon, false, normed_flat, stream);

    // The mask columns' final hidden, packed row after row.
    Tensor hidden            = work.alloc(DType::BF16, {kHidden, masks});
    const std::size_t row    = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(
        hidden.data, static_cast<std::size_t>(k) * row,
        static_cast<const std::byte*>(normed.data) + row, static_cast<std::size_t>(w) * row,
        static_cast<std::size_t>(k) * row, static_cast<std::size_t>(b), cudaMemcpyDeviceToDevice,
        stream));

    Tensor scores = work.alloc(DType::FP32, {kCandidates, k, b});
    {
        auto head_scope = work.scope();
        Tensor logits   = work.alloc(DType::BF16, {vocab_, masks});
        head_(hidden, logits, work, stream);
        Tensor candidate_ids    = call.candidates.view({kCandidates, masks});
        Tensor candidate_scores = scores.view({kCandidates, masks});
        ops::candidate_topk(logits, domain_, candidate_ids, candidate_scores, work, stream);
    }
    Tensor projected      = work.alloc(DType::BF16, {kRank, k, b});
    Tensor projected_flat = projected.view({kRank, masks});
    ops::linear(hidden, p_.selector_projection, projected_flat, ops::LinearPolicy::A16Only, work,
                stream);

    Tensor drafts      = call.drafts;
    Tensor proposal_q  = call.proposal_q;
    Tensor confidence  = call.confidence;
    Tensor* confidence_ptr = confidence.data != nullptr ? &confidence : nullptr;
    ops::candidate_selector_path(call.candidates, scores, projected, call.anchors,
                                 p_.predecessor_codebook, p_.successor_codebook, call.frontiers,
                                 call.configs, drafts, proposal_q, work, stream, confidence_ptr,
                                 call.edge_softcap);
}

std::size_t DFlash2Drafter::materialize_workspace_bytes(std::int32_t max_batch,
                                                        std::int32_t max_width) {
    // A single row may commit a whole prefill chunk: materialize keeps only that row's last
    // window of columns, so its workspace is bounded by the window even when the caller sizes
    // for a much wider chunk.
    if (max_batch < 1 || max_batch > 8 || max_width < 1 ||
        max_width > (1 << 20)) {
        throw std::invalid_argument("DFlash2 drafter workspace: invalid W/B interval");
    }
    const std::int32_t width =
        max_batch == 1 && max_width > static_cast<std::int32_t>(kWindow)
            ? static_cast<std::int32_t>(kWindow)
            : max_width;
    const std::uint64_t columns = static_cast<std::uint64_t>(width) *
                                  static_cast<std::uint64_t>(max_batch);
    const std::size_t op_workspace =
        ops::context_kv_materialize_workspace_capacity_bytes(max_batch, 1, width);
    return 2 * plane(static_cast<std::uint64_t>(kHidden), columns, 2) + op_workspace + kSlack;
}

std::size_t DFlash2Drafter::propose_workspace_bytes(std::int32_t max_batch,
                                                    std::int32_t max_width,
                                                    std::uint32_t max_context,
                                                    std::size_t head_bytes) {
    if (max_batch < 1 || max_batch > 8 || max_width < 2 || max_width > 16) {
        throw std::invalid_argument("DFlash2 drafter workspace: invalid W/B interval");
    }
    const std::uint64_t columns = static_cast<std::uint64_t>(max_width) *
                                  static_cast<std::uint64_t>(max_batch);
    const std::uint64_t masks = static_cast<std::uint64_t>(max_width - 1) *
                                static_cast<std::uint64_t>(max_batch);
    const std::size_t prepare_workspace =
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            kHidden, 2, max_width, 1, max_batch);
    const std::size_t attention_add_workspace =
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            kHidden, kQueryWidth, 2, max_width, 1, max_batch);
    const std::size_t mlp_add_workspace =
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            kHidden, kIntermediate, 2, max_width, 1, max_batch);
    const std::size_t swiglu_workspace =
        ops::dflash2_linear_swiglu_workspace_capacity_bytes(1, static_cast<std::int32_t>(columns));
    const std::size_t swa_workspace =
        ops::swa_workspace_capacity_bytes({0, max_context}, 1, max_width, max_batch);
    const std::size_t topk_workspace =
        ops::candidate_topk_workspace_capacity_bytes(static_cast<std::int32_t>(masks));
    const std::size_t selector_workspace =
        ops::candidate_selector_path_workspace_capacity_bytes(1, max_width - 1, 1, max_batch);

    const std::size_t prepared = plane(static_cast<std::uint64_t>(kHidden), columns, 2) +
                                 plane(static_cast<std::uint64_t>(kGroups), 2 * columns, 2);
    const std::size_t attention =
        prepared + prepare_workspace + plane(static_cast<std::uint64_t>(kQueryWidth), columns, 2) +
        2 * plane(static_cast<std::uint64_t>(kKeyWidth), columns, 2) +
        2 * plane(static_cast<std::uint64_t>(kQueryWidth), columns, 2) +
        plane(static_cast<std::uint64_t>(kKeyWidth), columns, 2) + attention_add_workspace +
        swa_workspace;
    const std::size_t mlp =
        prepared + prepare_workspace + plane(static_cast<std::uint64_t>(kIntermediate), columns, 2) +
        swiglu_workspace + mlp_add_workspace;
    const std::size_t tail = plane(static_cast<std::uint64_t>(kHidden), columns, 2) +
                             plane(static_cast<std::uint64_t>(kHidden), masks, 2) +
                             plane(static_cast<std::uint64_t>(DFlash2Geometry::vocab), masks, 2) +
                             topk_workspace +
                             plane(static_cast<std::uint64_t>(kRank), masks, 2) +
                             selector_workspace + head_bytes;
    return plane(static_cast<std::uint64_t>(kHidden), columns, 2) +
           std::max({attention, mlp, tail}) + kSlack;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
