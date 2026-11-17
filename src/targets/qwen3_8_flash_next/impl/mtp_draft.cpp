#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/rmsnorm.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/mtp_bindings.h"
#include "targets/qwen3_8_flash_next/impl/leaves.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#include "targets/qwen4_exp/impl/runtime/text_context_impl.h"
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_QWEN4EXP_VARIANT

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

namespace family = ninfer::targets::qwen4_exp::detail::flash_next_runtime;
using ninfer::targets::qwen4_exp::detail::kTextActivationDtype;
using mtp::MtpPlan;
using mtp::bind_mtp_artifact;

constexpr std::int32_t kHidden = TextConfig::hidden;
constexpr std::int32_t kStreams = TextConfig::hc_streams;
constexpr std::int32_t kResidual = TextConfig::residual_width;
constexpr std::int32_t kLowrank = TextConfig::hc_lowrank;
constexpr std::int32_t kExperts = TextConfig::experts;
constexpr std::int32_t kTopK = TextConfig::experts_per_token;
constexpr std::int32_t kIntermediate = TextConfig::expert_intermediate;
constexpr std::int32_t kShared = TextConfig::shared_intermediate;
constexpr std::int32_t kQueryWidth = TextConfig::query_width;
constexpr std::int32_t kKeyValueRows =
    2 * TextConfig::kv_width + TextConfig::index_query_width + TextConfig::index_key_width;
constexpr std::int32_t kQueryGateRows = 2 * kQueryWidth;
constexpr std::int32_t kVocab = TextConfig::vocab;
constexpr float kEpsilon = TextConfig::rms_epsilon;

inline std::uint16_t bf16_to_fp16(std::uint16_t word) {
    __nv_bfloat16 value;
    std::memcpy(&value, &word, sizeof(value));
    __half converted = __float2half_rn(__bfloat162float(value));
    std::uint16_t out;
    std::memcpy(&out, &converted, sizeof(out));
    return out;
}

inline float bf16_to_float(std::uint16_t word) {
    __nv_bfloat16 value;
    std::memcpy(&value, &word, sizeof(value));
    return __bfloat162float(value);
}

inline float e4m3_to_float(std::uint8_t code) {
    const float sign = (code & 0x80U) != 0 ? -1.0F : 1.0F;
    const std::uint8_t exponent = (code >> 3) & 0x0FU;
    const std::uint8_t fraction = code & 0x07U;
    if (exponent == 0) {
        if (fraction == 0) { return sign * 0.0F; }
        return sign * static_cast<float>(fraction) * (1.0F / 512.0F);
    }
    if (exponent == 0x0FU && fraction == 0x07U) { return sign * std::numeric_limits<float>::quiet_NaN(); }
    return sign * (1.0F + static_cast<float>(fraction) / 8.0F) *
           std::ldexp(1.0F, static_cast<int>(exponent) - 7);
}

inline std::size_t align256(std::size_t value) { return (value + 255U) / 256U * 256U; }

struct WeightLoader {
    artifact::Reader& reader;
    const MtpPlan& plan;
    std::vector<DeviceBuffer>& owned;

    std::uint64_t bytes(const std::string& name) const {
        const auto* object = reader.find(name);
        if (object == nullptr) { throw std::runtime_error("MTP artifact is missing " + name); }
        return artifact::object_bytes(*object);
    }

    std::vector<std::uint16_t> words(const BoundParameter& parameter, std::size_t rows,
                                     std::size_t columns) const {
        if (rows * columns == 0) { return {}; }
        std::vector<std::uint16_t> out(rows * columns, 0);
        std::size_t row_destination = 0;
        for (const BoundPart& part : parameter.parts) {
            const BoundObject& object = plan.objects.at(part.object);
            const auto span = reader.payload(object.name);
            const auto& view = part.view;
            const std::size_t object_columns =
                object.shape.size() >= 2 && object.shape[1] != 0 ? object.shape[1] : 1;
            const std::size_t part_rows = static_cast<std::size_t>(view.matrix_row_count);
            const std::size_t part_columns = static_cast<std::size_t>(view.column_count);
            if (part_columns != columns || row_destination + part_rows > rows) {
                throw std::runtime_error("MTP weight region does not match its declared rectangle");
            }
            const std::uint16_t* base = reinterpret_cast<const std::uint16_t*>(span.data.data()) +
                                        static_cast<std::size_t>(view.matrix_row_begin) * object_columns +
                                        static_cast<std::size_t>(view.column_begin);
            for (std::size_t row = 0; row < part_rows; ++row) {
                std::copy_n(base + row * object_columns, part_columns,
                            out.data() + (row_destination + row) * columns);
            }
            row_destination += part_rows;
        }
        if (row_destination != rows) {
            throw std::runtime_error("MTP weight region " + parameter.parameter +
                                     " has " + std::to_string(row_destination) + " rows, expected " +
                                     std::to_string(rows));
        }
        return out;
    }

    void* store(const void* data, std::size_t size) {
        owned.emplace_back(size);
        owned.back().copy_from_host(data, size);
        return owned.back().p;
    }

    Weight matrix(const BoundParameter& parameter, std::int32_t rows, std::int32_t columns) {
        auto source = words(parameter, static_cast<std::size_t>(rows), static_cast<std::size_t>(columns));
        std::vector<std::uint16_t> converted(source.size());
        for (std::size_t index = 0; index < source.size(); ++index) {
            converted[index] = bf16_to_fp16(source[index]);
        }
        Weight weight{};
        weight.payload = store(converted.data(), converted.size() * 2);
        weight.qdata = weight.payload;
        weight.payload_bytes = static_cast<std::uint64_t>(converted.size() * 2);
        weight.qtype = QType::FP16_CTRL;
        weight.layout = QuantLayout::Contiguous;
        weight.n = rows;
        weight.k = columns;
        weight.shape[0] = rows;
        weight.shape[1] = columns;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = columns;
        weight.ndim = 2;
        return weight;
    }

    Tensor vector(const BoundParameter& parameter, std::initializer_list<std::int32_t> shape) {
        std::int32_t elements = 1;
        for (const std::int32_t extent : shape) { elements *= extent; }
        auto source = words(parameter, static_cast<std::size_t>(elements), 1);
        return Tensor(store(source.data(), source.size() * 2), DType::BF16, shape);
    }

};

Weight fp16_weight_from_host(void* data, std::int32_t rows, std::int32_t columns) {
    Weight weight{};
    weight.payload = data;
    weight.qdata = data;
    weight.payload_bytes = static_cast<std::uint64_t>(rows) * columns * 2;
    weight.qtype = QType::FP16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = rows;
    weight.shape[1] = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.ndim = 2;
    return weight;
}

void decode_bank_experts(artifact::Reader& reader, const BoundParameter& bank,
                         DeviceBuffer& destination) {
    const ExpertLibrary& library = bank.library.value();
    const auto span = reader.payload(library.object);
    const std::size_t matrices = static_cast<std::size_t>(library.matrices);
    const std::size_t rows = static_cast<std::size_t>(library.rows);
    const std::size_t columns = static_cast<std::size_t>(library.columns);
    const std::size_t code_bytes = matrices * rows * columns;
    const std::size_t scale_offset = align256(code_bytes);
    const std::size_t row_blocks = rows / 128;
    const std::size_t column_blocks = columns / 128;
    const std::size_t blocks = row_blocks * column_blocks;
    if (span.data.size() < scale_offset + matrices * blocks * 2) {
        throw std::runtime_error("MTP expert bank payload is truncated");
    }
    const auto* codes = span.data.data();
    const auto* scales = reinterpret_cast<const std::uint16_t*>(span.data.data() + scale_offset);
    std::vector<std::uint16_t> decoded(matrices * rows * columns);
    for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const float code = e4m3_to_float(
                    static_cast<std::uint8_t>(codes[matrix * rows * columns + row * columns + column]));
                const std::uint16_t scale_word =
                    scales[matrix * blocks + (row / 128) * column_blocks + column / 128];
                const float value = code * bf16_to_float(scale_word);
                __half half = __float2half_rn(value);
                std::memcpy(&decoded[(matrix * rows + row) * columns + column], &half, sizeof(half));
            }
        }
    }
    destination = DeviceBuffer(decoded.size() * 2);
    destination.copy_from_host(decoded.data(), decoded.size() * 2);
}

void no_op_ple_fetch(void*, const ops::PleRowAddress*, std::size_t, std::byte*) {}

float g_ple_divisors[ops::kPleShards]{};

} // namespace

struct MtpDraft::Impl {
    static Variant::ModelView initial_model() {
        Variant::ModelView result{};
        result.loaded_layers = 1;
        result.ple.rows = qwen4_exp::PleRowSource{
            .context = nullptr, .fetch = no_op_ple_fetch, .divisors = g_ple_divisors};
        return result;
    }

    explicit Impl(const std::filesystem::path& artifact, int device)
        : reader(artifact), device_context(device),
          activation(512U * 1024U * 1024U), workspace(512U * 1024U * 1024U),
          model(initial_model()),
          state(device_context, model, 64, 1, 1, 0),
          trunk(model, state, activation, workspace, device_context.stream),
          condition_ones(4 * sizeof(float)) {
        artifact::Binder binder(reader);
        const BindingPlan text = bind_artifact(binder, WeightsProfile::Nvfp4, 1);
        const MtpPlan mtp = bind_mtp_artifact(binder, text);
        WeightLoader loader{reader, mtp, owned};

        const QsaPlan& qsa = std::get<QsaPlan>(mtp.layer.mixer);
        const std::size_t attention_rows =
            static_cast<std::size_t>(kQueryGateRows + kKeyValueRows);
        std::vector<std::uint16_t> assembled_attention(attention_rows * static_cast<std::size_t>(kHidden));

        auto assemble = [&](const BoundParameter& parameter, std::size_t row_begin,
                            std::size_t count) {
            auto region = loader.words(parameter, count, static_cast<std::size_t>(kHidden));
            std::copy(region.begin(), region.end(),
                      assembled_attention.begin() + row_begin * static_cast<std::size_t>(kHidden));
        };
        assemble(qsa.query, 0, static_cast<std::size_t>(kQueryWidth));
        assemble(qsa.output_gate, static_cast<std::size_t>(kQueryWidth),
                 static_cast<std::size_t>(kQueryWidth));
        assemble(qsa.key, static_cast<std::size_t>(kQueryGateRows), static_cast<std::size_t>(TextConfig::kv_width));
        assemble(qsa.value, static_cast<std::size_t>(kQueryGateRows) + TextConfig::kv_width,
                 static_cast<std::size_t>(TextConfig::kv_width));
        assemble(qsa.index_query,
                 static_cast<std::size_t>(kQueryGateRows) + 2U * TextConfig::kv_width,
                 static_cast<std::size_t>(TextConfig::index_query_width));
        assemble(qsa.index_key,
                 static_cast<std::size_t>(kQueryGateRows) + 2U * TextConfig::kv_width +
                     TextConfig::index_query_width,
                 static_cast<std::size_t>(TextConfig::index_key_width));
        std::vector<std::uint16_t> assembled_fp16(assembled_attention.size());
        for (std::size_t index = 0; index < assembled_attention.size(); ++index) {
            assembled_fp16[index] = bf16_to_fp16(assembled_attention[index]);
        }

        attention_projection.query_gate = fp16_weight_from_host(
            loader.store(assembled_fp16.data(), assembled_fp16.size() * 2), kQueryGateRows, kHidden);
        attention_projection.key_value = fp16_weight_from_host(
            const_cast<std::byte*>(static_cast<const std::byte*>(
                attention_projection.query_gate.payload)) +
                static_cast<std::size_t>(kQueryGateRows) * kHidden * 2,
            kKeyValueRows, kHidden);
        attention_projection.query_norm = loader.vector(qsa.query_norm, {TextConfig::head_dim});
        attention_projection.key_norm = loader.vector(qsa.key_norm, {TextConfig::head_dim});
        attention_projection.index_query_norm =
            loader.vector(qsa.index_query_norm, {TextConfig::index_head_dim});
        attention_projection.index_key_norm =
            loader.vector(qsa.index_key_norm, {TextConfig::index_head_dim});
        attention_projection.output = loader.matrix(qsa.output, kHidden, kQueryWidth);

        attention_hc = hc_weights(loader, mtp.layer.attention_hc);
        moe_hc = hc_weights(loader, mtp.layer.moe_hc);
        final_hc_weights = hc_final_weights(loader, mtp.final_hc);
        pre_embedding_norm = loader.vector(mtp.pre_fc_embedding_norm, {kHidden});
        pre_hidden_norm = loader.vector(mtp.pre_fc_hidden_norm, {kResidual});
        fc_embedding = loader.matrix(mtp.fc_embedding, kHidden, kHidden);
        fc_hidden = loader.matrix(mtp.fc_hidden, kHidden, kHidden);
        embedding_table = loader.matrix(mtp.token_embedding, kVocab, kHidden);
        output_head = loader.matrix(mtp.output_head, kVocab, kHidden);

        const MoePlan& moe = mtp.layer.moe;
        auto control_source = loader.words(moe.router, static_cast<std::size_t>(kExperts),
                                           static_cast<std::size_t>(kHidden));
        auto score_source = loader.words(moe.shared_score, 1, static_cast<std::size_t>(kHidden));
        std::vector<std::uint16_t> control_words(control_source.size() + score_source.size());
        std::copy(control_source.begin(), control_source.end(), control_words.begin());
        std::copy(score_source.begin(), score_source.end(), control_words.begin() + control_source.size());
        std::vector<std::uint16_t> control_fp16(control_words.size());
        for (std::size_t index = 0; index < control_words.size(); ++index) {
            control_fp16[index] = bf16_to_fp16(control_words[index]);
        }
        control = fp16_weight_from_host(loader.store(control_fp16.data(), control_fp16.size() * 2),
                                        kExperts + 1, kHidden);

        auto gate_source = loader.words(moe.shared_gate, static_cast<std::size_t>(kShared),
                                        static_cast<std::size_t>(kHidden));
        auto up_source = loader.words(moe.shared_up, static_cast<std::size_t>(kShared),
                                      static_cast<std::size_t>(kHidden));
        std::vector<std::uint16_t> shared_words(gate_source.size() + up_source.size());
        std::copy(gate_source.begin(), gate_source.end(), shared_words.begin());
        std::copy(up_source.begin(), up_source.end(), shared_words.begin() + gate_source.size());
        std::vector<std::uint16_t> shared_fp16(shared_words.size());
        for (std::size_t index = 0; index < shared_words.size(); ++index) {
            shared_fp16[index] = bf16_to_fp16(shared_words[index]);
        }
        shared_up = fp16_weight_from_host(loader.store(shared_fp16.data(), shared_fp16.size() * 2),
                                          2 * kShared, kHidden);
        shared_down = loader.matrix(moe.shared_down, kHidden, kShared);

        decode_bank_experts(reader, moe.routed_gate, gate_bank);
        decode_bank_experts(reader, moe.routed_down, down_bank);
        expert_bytes = gate_bank.bytes + down_bank.bytes;

        model.layers[0].attention_projection = attention_projection;
        model.ple.rows = qwen4_exp::PleRowSource{
            .context = nullptr, .fetch = no_op_ple_fetch, .divisors = ple_divisors};

        const float ones[4] = {1.0F, 1.0F, 1.0F, 1.0F};
        condition_ones.copy_from_host(ones, sizeof(ones));
    }

    static Weight hc_down_injection(WeightLoader& loader, const HcPlan& plan) {
        auto down = loader.words(plan.down, static_cast<std::size_t>(kLowrank),
                                 static_cast<std::size_t>(kResidual));
        auto inject = loader.words(plan.inject, static_cast<std::size_t>(kStreams),
                                   static_cast<std::size_t>(kResidual));
        std::vector<std::uint16_t> combined(down.size() + inject.size());
        std::copy(down.begin(), down.end(), combined.begin());
        std::copy(inject.begin(), inject.end(), combined.begin() + down.size());
        std::vector<std::uint16_t> converted(combined.size());
        for (std::size_t index = 0; index < combined.size(); ++index) {
            converted[index] = bf16_to_fp16(combined[index]);
        }
        return fp16_weight_from_host(loader.store(converted.data(), converted.size() * 2),
                                     kLowrank + kStreams, kResidual);
    }

    static qwen4_exp::HcWeights hc_weights(WeightLoader& loader, const HcPlan& plan) {
        return qwen4_exp::HcWeights{
            .down_injection = hc_down_injection(loader, plan),
            .up = loader.matrix(plan.up, kResidual, kLowrank),
            .norm = loader.vector(plan.norm, {kHidden, kStreams}),
        };
    }

    static qwen4_exp::HcWeights hc_final_weights(WeightLoader& loader, const HcPlan& plan) {
        return qwen4_exp::HcWeights{
            .down_injection = loader.matrix(plan.down, kLowrank, kResidual),
            .up = loader.matrix(plan.up, kResidual, kLowrank),
            .norm = loader.vector(plan.norm, {kHidden, kStreams}),
        };
    }

    Weight expert_weight(DeviceBuffer& bank, std::int32_t expert, std::int32_t row_offset,
                         std::int32_t rows, std::int32_t columns) {
        const std::size_t bank_rows = bank.bytes / (static_cast<std::size_t>(kExperts) * columns * 2);
        const std::byte* base = static_cast<const std::byte*>(bank.p) +
                                (static_cast<std::size_t>(expert) * bank_rows + row_offset) * columns * 2;
        return fp16_weight_from_host(const_cast<std::byte*>(base), rows, columns);
    }

    TokenId run(const Tensor& conditioning, TokenId next_token, std::int32_t position) {
        CUDA_CHECK(cudaSetDevice(device_context.device));
        auto activation_scope = activation.scope();
        auto workspace_scope = workspace.scope();
        cudaStream_t stream = device_context.stream;

        const std::int32_t tokens = 1;
        Tensor ids = activation.alloc(DType::I32, {tokens});
        const std::int32_t token_host = next_token;
        CUDA_CHECK(cudaMemcpyAsync(ids.data, &token_host, sizeof(token_host), cudaMemcpyHostToDevice,
                                   stream));
        Tensor embedding = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        ops::embedding(ids, embedding_table, embedding, stream);
        Tensor normalized_embedding = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        ops::rmsnorm(embedding, pre_embedding_norm, kEpsilon, true, normalized_embedding, stream);
        Tensor projected_embedding = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        ops::linear(normalized_embedding, fc_embedding, projected_embedding,
                    ops::LinearPolicy::A16Only, workspace, stream);

        Tensor conditioning_flat = conditioning.view({kResidual, tokens});
        Tensor normalized_hidden = activation.alloc(kTextActivationDtype, {kResidual, tokens});
        ops::rmsnorm(conditioning_flat, pre_hidden_norm, kEpsilon, true, normalized_hidden, stream);
        Tensor residual = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        ops::linear(normalized_hidden.view({kHidden, kStreams}), fc_hidden, residual,
                    ops::LinearPolicy::A16Only, workspace, stream);
        Tensor injection = activation.alloc(DType::FP32, {kStreams, tokens});
        CUDA_CHECK(cudaMemcpyAsync(injection.data, condition_ones.p, kStreams * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        Tensor normalized = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        ops::hc_combine_norm(residual, projected_embedding, injection,
                             attention_hc.norm, kEpsilon, residual, normalized, stream);

        Tensor block_input = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        Tensor mixer_injection = activation.alloc(DType::FP32, {kStreams, tokens});
        ops::hc_mix(normalized,
                    ops::HcMixWeights{attention_hc.down_injection, attention_hc.up}, workspace,
                    block_input, mixer_injection, stream);

        Tensor positions = activation.alloc(DType::I32, {tokens, 1, 1});
        Tensor valid_columns = activation.alloc(DType::I32, {1});
        Tensor table_rows = activation.alloc(DType::I32, {1});
        const std::int32_t position_host = position;
        const std::int32_t live_host = 1;
        const std::int32_t row_host = 0;
        CUDA_CHECK(cudaMemcpyAsync(positions.data, &position_host, sizeof(position_host),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(valid_columns.data, &live_host, sizeof(live_host),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows.data, &row_host, sizeof(row_host),
                                   cudaMemcpyHostToDevice, stream));

        Tensor mixer_out = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        Tensor staging = workspace.alloc(kTextActivationDtype, {kHidden, tokens});
        std::array<const AttentionProjectionWeights*, artifact::kMaximumDevices> weights{};
        weights[0] = &attention_projection;
        std::array<family::QsaLayerStateView, artifact::kMaximumDevices> qsa_state{};
        std::pair<family::TextState*, std::int32_t> publish_context{&state, 0};
        qsa_state[0] = family::QsaLayerStateView{
            .cache = state.cache(0),
            .before = state.before_tail(0),
            .const_tail = state.const_tail(0),
            .after = state.tail(0),
            .records = state.records(static_cast<std::size_t>(tokens)),
            .publish = nullptr,
            .publish_context = &publish_context,
        };
        std::array<Tensor, artifact::kMaximumDevices> block_inputs{};
        std::array<Tensor, artifact::kMaximumDevices> position_views{};
        std::array<Tensor, artifact::kMaximumDevices> valid_views{};
        std::array<Tensor, artifact::kMaximumDevices> table_views{};
        std::array<Tensor, artifact::kMaximumDevices> mixer_outputs{};
        std::array<Tensor, artifact::kMaximumDevices> staging_views{};
        block_inputs[0] = block_input;
        position_views[0] = positions;
        valid_views[0] = valid_columns;
        table_views[0] = table_rows;
        mixer_outputs[0] = mixer_out;
        staging_views[0] = staging;
        trunk.run_qsa_layer(tokens, weights, qsa_state, block_inputs, position_views, valid_views,
                            table_views, mixer_outputs, staging_views);

        Tensor stage = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        Tensor normalized_moe = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        ops::hc_combine_norm(residual, mixer_out, mixer_injection, moe_hc.norm, kEpsilon, stage,
                             normalized_moe, stream);
        Tensor moe_block_input = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        Tensor moe_injection = activation.alloc(DType::FP32, {kStreams, tokens});
        ops::hc_mix(normalized_moe,
                    ops::HcMixWeights{moe_hc.down_injection, moe_hc.up}, workspace,
                    moe_block_input, moe_injection, stream);

        Tensor moe_out = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        CUDA_CHECK(cudaMemsetAsync(moe_out.data, 0, moe_out.bytes(), stream));
        moe_forward(moe_block_input, moe_out, stream);

        Tensor residual_out = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        Tensor normalized_out = activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        ops::hc_combine_norm(stage, moe_out, moe_injection, final_hc_weights.norm, kEpsilon,
                             residual_out, normalized_out, stream);
        Tensor sample = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        ops::hc_final_mix(normalized_out,
                          ops::HcFinalMixWeights{final_hc_weights.down_injection,
                                                 final_hc_weights.up},
                          workspace, sample, stream);
        Tensor logits = activation.alloc(kTextActivationDtype, {kVocab, tokens});
        ops::linear(sample, output_head, logits, ops::LinearPolicy::A16Only, workspace, stream);
        Tensor winner = activation.alloc(DType::I32, {tokens});
        ops::argmax(logits, winner, kVocab, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int32_t result = 0;
        CUDA_CHECK(cudaMemcpy(&result, winner.data, sizeof(result), cudaMemcpyDeviceToHost));
        return static_cast<TokenId>(result);
    }

    void moe_forward(const Tensor& x, Tensor& destination, cudaStream_t stream) {
        Tensor control_out = activation.alloc(DType::FP16, {kExperts + 1, 1});
        ops::linear(x, control, control_out, ops::LinearPolicy::A16Only, workspace, stream);
        std::vector<__half> control_host(kExperts + 1);
        CUDA_CHECK(cudaMemcpyAsync(control_host.data(), control_out.data,
                                   control_host.size() * sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::pair<float, std::int32_t>> scores(kExperts);
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            scores[static_cast<std::size_t>(expert)] = {
                __half2float(control_host[static_cast<std::size_t>(expert)]), expert};
        }
        std::partial_sort(scores.begin(), scores.begin() + kTopK, scores.end(),
                          [](const auto& left, const auto& right) { return left.first > right.first; });
        std::array<std::int32_t, kTopK> selected{};
        std::array<float, kTopK> selected_scores{};
        float maximum = scores[0].first;
        for (std::int32_t index = 0; index < kTopK; ++index) {
            selected[static_cast<std::size_t>(index)] = scores[static_cast<std::size_t>(index)].second;
            selected_scores[static_cast<std::size_t>(index)] = scores[static_cast<std::size_t>(index)].first;
            maximum = std::max(maximum, selected_scores[static_cast<std::size_t>(index)]);
        }
        float denominator = 0.0F;
        for (std::int32_t index = 0; index < kTopK; ++index) {
            selected_scores[static_cast<std::size_t>(index)] =
                std::exp(selected_scores[static_cast<std::size_t>(index)] - maximum);
            denominator += selected_scores[static_cast<std::size_t>(index)];
        }
        for (std::int32_t index = 0; index < kTopK; ++index) {
            selected_scores[static_cast<std::size_t>(index)] /= denominator;
        }
        const float shared = 1.0F / (1.0F + std::exp(-__half2float(control_host[kExperts])));

        Tensor shared_projection = activation.alloc(kTextActivationDtype, {2 * kShared, 1});
        ops::linear(x, shared_up, shared_projection, ops::LinearPolicy::A16Only, workspace, stream);
        std::vector<__half> projection_host(2 * kShared);
        CUDA_CHECK(cudaMemcpyAsync(projection_host.data(), shared_projection.data,
                                   projection_host.size() * sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> activated(kShared);
        for (std::int32_t index = 0; index < kShared; ++index) {
            const float gate = __half2float(projection_host[static_cast<std::size_t>(index)]);
            const float up = __half2float(projection_host[static_cast<std::size_t>(kShared + index)]);
            const float value = (gate / (1.0F + std::exp(-gate))) * up;
            __half rounded = __float2half_rn(value);
            std::memcpy(&activated[static_cast<std::size_t>(index)], &rounded, sizeof(rounded));
        }
        Tensor shared_activation = activation.alloc(kTextActivationDtype, {kShared, 1});
        CUDA_CHECK(cudaMemcpyAsync(shared_activation.data, activated.data(),
                                   activated.size() * 2, cudaMemcpyHostToDevice, stream));
        Tensor shared_output = activation.alloc(kTextActivationDtype, {kHidden, 1});
        ops::linear(shared_activation, shared_down, shared_output, ops::LinearPolicy::A16Only,
                    workspace, stream);
        std::vector<__half> shared_host(kHidden);
        CUDA_CHECK(cudaMemcpyAsync(shared_host.data(), shared_output.data,
                                   shared_host.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> total(kHidden, 0.0F);
        for (std::int32_t index = 0; index < kHidden; ++index) {
            total[static_cast<std::size_t>(index)] =
                shared * __half2float(shared_host[static_cast<std::size_t>(index)]);
        }

        Tensor gate_up = activation.alloc(kTextActivationDtype, {2 * kIntermediate, 1});
        Tensor expert_activation = activation.alloc(kTextActivationDtype, {kIntermediate, 1});
        Tensor expert_output = activation.alloc(kTextActivationDtype, {kHidden, 1});
        std::vector<std::uint16_t> expert_activated(kIntermediate);
        std::vector<__half> expert_projection(2 * kIntermediate);
        std::vector<__half> expert_result(kHidden);
        for (std::int32_t index = 0; index < kTopK; ++index) {
            const std::int32_t expert = selected[static_cast<std::size_t>(index)];
            const Weight gate = expert_weight(gate_bank, expert, 0, kIntermediate, kHidden);
            const Weight up = expert_weight(gate_bank, expert, kIntermediate, kIntermediate, kHidden);
            Tensor gate_part = gate_up.slice(0, 0, kIntermediate);
            Tensor up_part = gate_up.slice(0, kIntermediate, kIntermediate);
            ops::linear(x, gate, gate_part, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(x, up, up_part, ops::LinearPolicy::A16Only, workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(expert_projection.data(), gate_up.data,
                                       expert_projection.size() * sizeof(__half),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (std::int32_t row = 0; row < kIntermediate; ++row) {
                const float gate_value = __half2float(expert_projection[static_cast<std::size_t>(row)]);
                const float up_value =
                    __half2float(expert_projection[static_cast<std::size_t>(kIntermediate + row)]);
                const float value = (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
                __half rounded = __float2half_rn(value);
                std::memcpy(&expert_activated[static_cast<std::size_t>(row)], &rounded, sizeof(rounded));
            }
            CUDA_CHECK(cudaMemcpyAsync(expert_activation.data, expert_activated.data(),
                                       expert_activated.size() * 2, cudaMemcpyHostToDevice, stream));
            const Weight down = expert_weight(down_bank, expert, 0, kHidden, kIntermediate);
            ops::linear(expert_activation, down, expert_output, ops::LinearPolicy::A16Only,
                        workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(expert_result.data(), expert_output.data,
                                       expert_result.size() * sizeof(__half),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const float coefficient = selected_scores[static_cast<std::size_t>(index)];
            for (std::int32_t row = 0; row < kHidden; ++row) {
                total[static_cast<std::size_t>(row)] +=
                    coefficient * __half2float(expert_result[static_cast<std::size_t>(row)]);
            }
        }
        std::vector<std::uint16_t> rounded(kHidden);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            __half value = __float2half_rn(total[static_cast<std::size_t>(row)]);
            std::memcpy(&rounded[static_cast<std::size_t>(row)], &value, sizeof(value));
        }
        CUDA_CHECK(cudaMemcpyAsync(destination.data, rounded.data(), rounded.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    artifact::Reader reader;
    DeviceContext device_context;
    std::vector<DeviceBuffer> owned;
    WorkspaceArena activation;
    WorkspaceArena workspace;
    Variant::ModelView model{};
    float ple_divisors[ops::kPleShards]{};
    family::TextState state;
    family::TextTrunk trunk;
    DeviceBuffer condition_ones;

    AttentionProjectionWeights attention_projection{};
    qwen4_exp::HcWeights attention_hc{};
    qwen4_exp::HcWeights moe_hc{};
    qwen4_exp::HcWeights final_hc_weights{};
    Tensor pre_embedding_norm{};
    Tensor pre_hidden_norm{};
    Weight fc_embedding{};
    Weight fc_hidden{};
    Weight embedding_table{};
    Weight output_head{};
    Weight control{};
    Weight shared_up{};
    Weight shared_down{};
    DeviceBuffer gate_bank{};
    DeviceBuffer down_bank{};
    std::uint64_t expert_bytes = 0;
};

MtpDraft::MtpDraft(const std::filesystem::path& artifact, int device)
    : impl_(std::make_unique<Impl>(artifact, device)) {}

MtpDraft::~MtpDraft() = default;

TokenId MtpDraft::run(const Tensor& conditioning, TokenId next_token, std::int32_t position) {
    return impl_->run(conditioning, next_token, position);
}

std::uint64_t MtpDraft::expert_bytes() const noexcept { return impl_->expert_bytes; }

} // namespace ninfer::targets::qwen3_8_flash_next::detail
