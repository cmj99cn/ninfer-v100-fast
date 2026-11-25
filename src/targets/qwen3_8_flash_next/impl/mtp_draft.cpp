#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
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
#include "targets/qwen3_8_flash_next/impl/diag_switches.h"
#include "targets/qwen3_8_flash_next/impl/leaves.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft_kernels.h"
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
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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

static_assert(kTextActivationDtype == DType::FP16,
              "MTP draft kernels assume FP16 text activations");

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

static_assert(DraftRouterChoice::kTopK == static_cast<std::size_t>(kTopK),
              "draft router choice capacity must match the draft top-k");

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

inline std::size_t align256(std::size_t value) { return (value + 255U) / 256U * 256U; }

struct BankLayout {
    std::size_t code_bytes   = 0;
    std::size_t scale_offset = 0;
    std::size_t scale_bytes  = 0;
    std::size_t blocks       = 0;
    std::size_t total_bytes  = 0;
};

inline BankLayout bank_layout(std::size_t matrices, std::size_t rows, std::size_t columns) {
    BankLayout layout;
    layout.code_bytes   = matrices * rows * columns;
    layout.scale_offset = align256(layout.code_bytes);
    layout.blocks       = (rows / 128U) * (columns / 128U);
    layout.scale_bytes  = matrices * layout.blocks * sizeof(std::uint16_t);
    layout.total_bytes  = layout.scale_offset + layout.scale_bytes;
    return layout;
}

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

    Weight q4_weight(const BoundParameter& parameter, std::int32_t rows, std::int32_t columns) {
        if (parameter.parts.size() != 1) {
            throw std::runtime_error("MTP Q4 proposal head must be one whole artifact object");
        }
        const BoundObject& object = plan.objects.at(parameter.parts.front().object);
        const auto span = reader.payload(object.name);
        const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(rows),
                                                 static_cast<std::uint64_t>(columns)};
        const auto geometry =
            artifact::row_split_geometry(artifact::NumericFormat::Q4G64_F16S, shape);
        if (span.data.size() < geometry.encoded_bytes) {
            throw std::runtime_error("MTP Q4 proposal head payload is truncated");
        }
        void* device = store(span.data.data(), static_cast<std::size_t>(geometry.encoded_bytes));
        return artifact::staged_weight_view(artifact::NumericFormat::Q4G64_F16S, device, rows,
                                            columns);
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

BankLayout store_bank(artifact::Reader& reader, const BoundParameter& bank,
                      DeviceBuffer& destination, int rank, int tp, std::int32_t experts) {
    const ExpertLibrary& library = bank.library.value();
    const auto span = reader.payload(library.object);
    const std::size_t ranks = static_cast<std::size_t>(tp);
    if (tp < 1 || static_cast<std::size_t>(experts) % ranks != 0 ||
        library.matrices != static_cast<std::uint64_t>(experts)) {
        throw std::runtime_error("MTP expert bank cannot be split across the rank group");
    }
    const std::size_t local_matrices = static_cast<std::size_t>(experts) / ranks;
    const BankLayout full = bank_layout(static_cast<std::size_t>(library.matrices),
                                        static_cast<std::size_t>(library.rows),
                                        static_cast<std::size_t>(library.columns));
    const BankLayout layout = bank_layout(local_matrices, static_cast<std::size_t>(library.rows),
                                          static_cast<std::size_t>(library.columns));
    if (span.data.size() < full.total_bytes) {
        throw std::runtime_error("MTP expert bank payload is truncated");
    }
    destination = DeviceBuffer(layout.total_bytes);
    const std::size_t rank_index = static_cast<std::size_t>(rank);
    destination.copy_from_host(span.data.data() + rank_index * layout.code_bytes,
                               layout.code_bytes);
    destination.copy_from_host(span.data.data() + full.scale_offset +
                                   rank_index * layout.scale_bytes,
                               layout.scale_bytes, layout.scale_offset);
    return layout;
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

    explicit Impl(const std::filesystem::path& artifact, int device, int rank_index, int tp)
        : reader(artifact), device_context(device), rank(rank_index), tp(tp),
          router_rank(rank_index == 0),
          expert_begin(rank_index * (kExperts / tp)), local_experts(kExperts / tp),
          activation(512U * 1024U * 1024U), workspace(512U * 1024U * 1024U),
          model(initial_model()),
          reduce_staging(static_cast<std::size_t>(kHidden) * sizeof(float)) {
        if (!router_rank) {
            routed_input = DeviceBuffer(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
        }
        artifact::Binder binder(reader);
        const BindingPlan text = bind_artifact(binder, WeightsProfile::Nvfp4, 1);
        const MtpPlan mtp = bind_mtp_artifact(binder, text);
        WeightLoader loader{reader, mtp, owned};
        const MoePlan& moe = mtp.layer.moe;

        if (router_rank) {
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
            embedding_words = loader.words(mtp.token_embedding, static_cast<std::size_t>(kVocab),
                                           static_cast<std::size_t>(kHidden));
            output_head = diag_fp16_draft_head()
                              ? loader.matrix(mtp.output_head, kVocab, kHidden)
                              : loader.q4_weight(mtp.proposal_head, kVocab, kHidden);

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
        } // router_rank

        gate_layout = store_bank(reader, moe.routed_gate, gate_bank, rank, tp, kExperts);
        down_layout = store_bank(reader, moe.routed_down, down_bank, rank, tp, kExperts);
        expert_bytes = gate_bank.bytes + down_bank.bytes;

        gate_fp16 = DeviceBuffer(static_cast<std::size_t>(2 * kIntermediate) * kHidden * 2);
        down_fp16 = DeviceBuffer(static_cast<std::size_t>(kHidden) * kIntermediate * 2);
        total = DeviceBuffer(static_cast<std::size_t>(kHidden) * sizeof(float));

        if (router_rank) {
            model.layers[0].attention_projection = attention_projection;
            model.ple.rows = qwen4_exp::PleRowSource{
                .context = nullptr, .fetch = no_op_ple_fetch, .divisors = ple_divisors};
            state = std::make_unique<family::TextState>(
                device_context, model,
                static_cast<std::uint32_t>(TextConfig::max_position_embeddings), 1, 1, 0);
            trunk = std::make_unique<family::TextTrunk>(model, *state, activation, workspace,
                                                        device_context.stream);
            chain_current = DeviceBuffer(MtpDraft::hidden_elements * sizeof(std::uint16_t));
            chain_next = DeviceBuffer(MtpDraft::hidden_elements * sizeof(std::uint16_t));
            condition_ones = DeviceBuffer(4 * sizeof(float));
            const float ones[4] = {1.0F, 1.0F, 1.0F, 1.0F};
            condition_ones.copy_from_host(ones, sizeof(ones));
        }
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

    struct Step {
        Tensor moe_input{};
        Tensor stage{};
        Tensor moe_injection{};
        Tensor moe_out{};
        Tensor residual_out{};
        Tensor normalized_out{};
        Tensor shared_output{};
        float shared = 0.0F;
    };

    Step begin(const Tensor& conditioning, TokenId next_token, std::int32_t position) {
        CUDA_CHECK(cudaSetDevice(device_context.device));
        cudaStream_t stream = device_context.stream;

        const std::int32_t tokens = 1;
        const std::int32_t token = static_cast<std::int32_t>(next_token);
        if (token < 0 || token >= kVocab) {
            throw std::runtime_error("MTP draft token is outside the vocabulary");
        }
        Tensor embedding = activation.alloc(kTextActivationDtype, {kHidden, tokens});
        std::vector<std::uint16_t> embedding_row(static_cast<std::size_t>(kHidden));
        const std::uint16_t* embedding_source =
            embedding_words.data() + static_cast<std::size_t>(token) * kHidden;
        for (std::int32_t index = 0; index < kHidden; ++index) {
            embedding_row[static_cast<std::size_t>(index)] = bf16_to_fp16(embedding_source[index]);
        }
        CUDA_CHECK(cudaMemcpyAsync(embedding.data, embedding_row.data(), embedding_row.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
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
        std::pair<family::TextState*, std::int32_t> publish_context{state.get(), 0};
        qsa_state[0] = family::QsaLayerStateView{
            .cache = state->cache(0),
            .before = state->before_tail(0),
            .const_tail = state->const_tail(0),
            .after = state->tail(0),
            .records = state->records(static_cast<std::size_t>(tokens)),
            .publish = family::publish_text_qsa_tail,
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
        trunk->run_qsa_layer(tokens, weights, qsa_state, block_inputs, position_views, valid_views,
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
        Step step{};
        step.moe_input     = moe_block_input;
        step.stage         = stage;
        step.moe_injection = moe_injection;
        step.moe_out       = moe_out;
        step.residual_out =
            activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        step.normalized_out =
            activation.alloc(kTextActivationDtype, {kHidden, kStreams, tokens});
        route_and_shared(moe_block_input, step, stream);
        run_experts(moe_block_input, selected, selected_scores, stream);
        return step;
    }

    void finish_state(Step& step, Tensor* chained_hidden) {
        CUDA_CHECK(cudaSetDevice(device_context.device));
        cudaStream_t stream = device_context.stream;
        mtp_scale_add_f16(step.shared_output.data, step.shared, static_cast<float*>(total.p),
                          kHidden, stream);
        mtp_round_f16(static_cast<const float*>(total.p), step.moe_out.data, kHidden, stream);
        ops::hc_combine_norm(step.stage, step.moe_out, step.moe_injection, final_hc_weights.norm,
                             kEpsilon, step.residual_out, step.normalized_out, stream);
        if (chained_hidden != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(chained_hidden->data, step.residual_out.data,
                                       step.residual_out.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
    }

    TokenId finish_logits(Step& step, float* confidence = nullptr) {
        CUDA_CHECK(cudaSetDevice(device_context.device));
        cudaStream_t stream = device_context.stream;
        Tensor sample = activation.alloc(kTextActivationDtype, {kHidden, 1});
        ops::hc_final_mix(step.normalized_out,
                          ops::HcFinalMixWeights{final_hc_weights.down_injection,
                                                 final_hc_weights.up},
                          workspace, sample, stream);
        Tensor logits = activation.alloc(kTextActivationDtype, {kVocab, 1});
        ops::linear(sample, output_head, logits, ops::LinearPolicy::A16Only, workspace, stream);
        Tensor winner = activation.alloc(DType::I32, {1});
        ops::argmax(logits, winner, kVocab, stream);
        Tensor confidence_device;
        if (confidence != nullptr) {
            confidence_device = activation.alloc(DType::FP32, {1});
            mtp_top1_prob_f16(logits.data, static_cast<float*>(confidence_device.data), kVocab,
                              stream);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int32_t result = 0;
        CUDA_CHECK(cudaMemcpy(&result, winner.data, sizeof(result), cudaMemcpyDeviceToHost));
        if (confidence != nullptr) {
            CUDA_CHECK(cudaMemcpy(confidence, confidence_device.data, sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        return static_cast<TokenId>(result);
    }

    // Rank 0 only: the full 512-expert router and the shared expert. Produces the global top-k and
    // coefficients (still over all experts) in `selected`/`selected_scores`, the shared scalar in
    // step.shared and the shared expert output in step.shared_output.
    void route_and_shared(const Tensor& x, Step& step, cudaStream_t stream) {
        Tensor control_out = activation.alloc(DType::FP16, {kExperts + 1, 1});
        ops::linear(x, control, control_out, ops::LinearPolicy::A16Only, workspace, stream);
        std::vector<__half> control_host(kExperts + 1);
        CUDA_CHECK(cudaMemcpyAsync(control_host.data(), control_out.data,
                                   control_host.size() * sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream));
        ++host_syncs;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::pair<float, std::int32_t>> scores(kExperts);
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            scores[static_cast<std::size_t>(expert)] = {
                __half2float(control_host[static_cast<std::size_t>(expert)]), expert};
        }
        std::partial_sort(scores.begin(), scores.begin() + kTopK, scores.end(),
                          [](const auto& left, const auto& right) { return left.first > right.first; });
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
        step.shared = 1.0F / (1.0F + std::exp(-__half2float(control_host[kExperts])));

        Tensor shared_projection = activation.alloc(kTextActivationDtype, {2 * kShared, 1});
        ops::linear(x, shared_up, shared_projection, ops::LinearPolicy::A16Only, workspace, stream);
        Tensor shared_activation = activation.alloc(kTextActivationDtype, {kShared, 1});
        mtp_silu_mul_f16(shared_projection.data, shared_activation.data, kShared, stream);
        Tensor shared_output = activation.alloc(kTextActivationDtype, {kHidden, 1});
        ops::linear(shared_activation, shared_down, shared_output, ops::LinearPolicy::A16Only,
                    workspace, stream);
        step.shared_output = shared_output;
    }

    // Every rank: FP32 accumulation of the routed experts that fall in this rank's contiguous
    // range, using the global top-k ids and coefficients produced by rank 0.
    void run_experts(const Tensor& x, const std::array<std::int32_t, kTopK>& expert_ids,
                     const std::array<float, kTopK>& weights, cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(total.p, 0, static_cast<std::size_t>(kHidden) * sizeof(float),
                                   stream));

        Tensor gate_up = activation.alloc(kTextActivationDtype, {2 * kIntermediate, 1});
        Tensor expert_activation = activation.alloc(kTextActivationDtype, {kIntermediate, 1});
        Tensor expert_output = activation.alloc(kTextActivationDtype, {kHidden, 1});
        for (std::int32_t index = 0; index < kTopK; ++index) {
            const std::int32_t expert = expert_ids[static_cast<std::size_t>(index)];
            if (expert < expert_begin || expert >= expert_begin + local_experts) { continue; }
            const std::int32_t local = expert - expert_begin;
            const std::size_t gate_code_offset =
                static_cast<std::size_t>(local) * (2 * kIntermediate) * kHidden;
            const std::size_t gate_scale_offset =
                static_cast<std::size_t>(local) * gate_layout.blocks;
            mtp_decode_fp8_block128(
                static_cast<const std::uint8_t*>(gate_bank.p) + gate_code_offset,
                reinterpret_cast<const std::uint16_t*>(
                    static_cast<const std::uint8_t*>(gate_bank.p) + gate_layout.scale_offset) +
                    gate_scale_offset,
                2 * kIntermediate, kHidden, gate_fp16.p, stream);
            const Weight gate = fp16_weight_from_host(gate_fp16.p, kIntermediate, kHidden);
            const Weight up = fp16_weight_from_host(
                static_cast<std::byte*>(gate_fp16.p) +
                    static_cast<std::size_t>(kIntermediate) * kHidden * 2,
                kIntermediate, kHidden);
            Tensor gate_part = gate_up.slice(0, 0, kIntermediate);
            Tensor up_part = gate_up.slice(0, kIntermediate, kIntermediate);
            ops::linear(x, gate, gate_part, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(x, up, up_part, ops::LinearPolicy::A16Only, workspace, stream);
            mtp_silu_mul_f16(gate_up.data, expert_activation.data, kIntermediate, stream);

            const std::size_t down_code_offset =
                static_cast<std::size_t>(local) * kHidden * kIntermediate;
            const std::size_t down_scale_offset =
                static_cast<std::size_t>(local) * down_layout.blocks;
            mtp_decode_fp8_block128(
                static_cast<const std::uint8_t*>(down_bank.p) + down_code_offset,
                reinterpret_cast<const std::uint16_t*>(
                    static_cast<const std::uint8_t*>(down_bank.p) + down_layout.scale_offset) +
                    down_scale_offset,
                kHidden, kIntermediate, down_fp16.p, stream);
            const Weight down = fp16_weight_from_host(down_fp16.p, kHidden, kIntermediate);
            ops::linear(expert_activation, down, expert_output, ops::LinearPolicy::A16Only,
                        workspace, stream);
            mtp_scale_add_f16(expert_output.data, weights[static_cast<std::size_t>(index)],
                              static_cast<float*>(total.p), kHidden, stream);
        }
    }

    artifact::Reader reader;
    DeviceContext device_context;
    int rank = 0;
    int tp = 1;
    bool router_rank = false;
    std::int32_t expert_begin = 0;
    std::int32_t local_experts = kExperts;
    std::vector<DeviceBuffer> owned;
    WorkspaceArena activation;
    WorkspaceArena workspace;
    Variant::ModelView model{};
    float ple_divisors[ops::kPleShards]{};
    std::unique_ptr<family::TextState> state;
    std::unique_ptr<family::TextTrunk> trunk;
    std::array<std::int32_t, kTopK> selected{};
    std::array<float, kTopK> selected_scores{};
    DeviceBuffer condition_ones;

    AttentionProjectionWeights attention_projection{};
    qwen4_exp::HcWeights attention_hc{};
    qwen4_exp::HcWeights moe_hc{};
    qwen4_exp::HcWeights final_hc_weights{};
    Tensor pre_embedding_norm{};
    Tensor pre_hidden_norm{};
    Weight fc_embedding{};
    Weight fc_hidden{};
    std::vector<std::uint16_t> embedding_words;
    Weight output_head{};
    Weight control{};
    Weight shared_up{};
    Weight shared_down{};
    DeviceBuffer gate_bank{};
    DeviceBuffer down_bank{};
    BankLayout gate_layout{};
    BankLayout down_layout{};
    DeviceBuffer gate_fp16{};
    DeviceBuffer down_fp16{};
    DeviceBuffer total{};
    DeviceBuffer reduce_staging{};
    DeviceBuffer routed_input;
    DeviceBuffer chain_current;
    DeviceBuffer chain_next;
    std::uint64_t expert_bytes = 0;
    std::uint64_t host_syncs = 0;
};

MtpDraft::MtpDraft(const std::filesystem::path& artifact, const ExecutionContext& execution,
                   const ops::CollectiveComms& comms)
    : execution_(&execution), comms_(&comms) {
    if (execution.tp < 1 || execution.tp > static_cast<int>(kMaximumDevices)) {
        throw std::runtime_error("MTP draft execution width is invalid");
    }
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    for (int rank = 0; rank < execution.tp; ++rank) {
        if (!execution.dev[static_cast<std::size_t>(rank)].has_value()) {
            throw std::runtime_error("MTP draft rank has no device context");
        }
        const auto slot = static_cast<std::size_t>(rank);
        ranks_.push_back(std::make_unique<Impl>(artifact, execution.dev[slot]->device, rank,
                                                execution.tp));
        CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
        CUDA_CHECK(cudaEventCreateWithFlags(&x_ready_[slot], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&x_after_[slot], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&reduce_before_[slot], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&reduce_after_[slot], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaSetDevice(previous));
}

MtpDraft::~MtpDraft() {
    for (int rank = 0; rank < execution_->tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (x_ready_[slot] != nullptr) { CUDA_CHECK(cudaEventDestroy(x_ready_[slot])); }
        if (x_after_[slot] != nullptr) { CUDA_CHECK(cudaEventDestroy(x_after_[slot])); }
        if (reduce_before_[slot] != nullptr) { CUDA_CHECK(cudaEventDestroy(reduce_before_[slot])); }
        if (reduce_after_[slot] != nullptr) { CUDA_CHECK(cudaEventDestroy(reduce_after_[slot])); }
    }
}

TokenId MtpDraft::run(const RankTensors& conditioning, TokenId next_token, std::int32_t position,
                      RankTensors* chained_hidden, float* confidence) {
    const int tp = execution_->tp;
    Impl& router = *ranks_[0];
    std::array<std::optional<WorkspaceArena::Scope>, kMaximumDevices> activation_scopes{};
    std::array<std::optional<WorkspaceArena::Scope>, kMaximumDevices> workspace_scopes{};
    activation_scopes[0].emplace(router.activation.scope());
    workspace_scopes[0].emplace(router.workspace.scope());

    // Rank 0 alone runs the non-expert path, the full router, the shared expert and its own expert
    // range. The global top-k decision and shared scalar are published through MtpDraft host
    // members, which the expert-only ranks read directly (one process, one host address space).
    Impl::Step step = router.begin(conditioning[0], next_token, position);
    router_.experts = router.selected;
    router_.scores = router.selected_scores;
    router_.shared = step.shared;

    RankTensors totals{};
    RankTensors staging{};
    totals[0]  = Tensor(router.total.p, DType::FP32, {kHidden, 1});
    staging[0] = Tensor(router.reduce_staging.p, DType::FP32, {kHidden, 1});

    if (tp > 1) {
        // Publish rank 0's MoE input (2560 FP16) to every peer on the device in FP16, with no host
        // staging. The draft stream produces x while collectives run on each engine stream, so the
        // handoff is ordered with pre-created events only; there is no host wait here.
        CUDA_CHECK(cudaSetDevice(execution_->dev[0]->device));
        CUDA_CHECK(cudaEventRecord(x_ready_[0], router.device_context.stream));
        CUDA_CHECK(cudaStreamWaitEvent(execution_->dev[0]->stream, x_ready_[0], 0));
        std::array<Tensor, kMaximumDevices> received{};
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            received[slot] =
                Tensor(ranks_[slot]->routed_input.p, kTextActivationDtype, {kHidden, 1});
        }
        ops::broadcast_rank0(step.moe_input, received, *execution_, *comms_);

        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            Impl& expert = *ranks_[slot];
            CUDA_CHECK(cudaSetDevice(execution_->dev[slot]->device));
            CUDA_CHECK(cudaEventRecord(x_after_[slot], execution_->dev[slot]->stream));
            CUDA_CHECK(cudaStreamWaitEvent(expert.device_context.stream, x_after_[slot], 0));
            activation_scopes[slot].emplace(expert.activation.scope());
            workspace_scopes[slot].emplace(expert.workspace.scope());
            expert.run_experts(received[slot], router_.experts, router_.scores,
                               expert.device_context.stream);
            totals[slot]  = Tensor(expert.total.p, DType::FP32, {kHidden, 1});
            staging[slot] = Tensor(expert.reduce_staging.p, DType::FP32, {kHidden, 1});
        }
    }

    // One FP32 all-reduce over the per-rank expert accumulators; the shared term is added exactly
    // once afterwards by rank 0, which alone rounds to FP16 and runs the output head and argmax.
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(execution_->dev[slot]->device));
        CUDA_CHECK(cudaEventRecord(reduce_before_[slot], ranks_[slot]->device_context.stream));
        CUDA_CHECK(cudaStreamWaitEvent(execution_->dev[slot]->stream, reduce_before_[slot], 0));
    }
    ops::allreduce_sum(totals, staging, *execution_, *comms_);
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        CUDA_CHECK(cudaSetDevice(execution_->dev[slot]->device));
        CUDA_CHECK(cudaEventRecord(reduce_after_[slot], execution_->dev[slot]->stream));
        CUDA_CHECK(cudaStreamWaitEvent(ranks_[slot]->device_context.stream, reduce_after_[slot], 0));
    }

    Tensor* chained = chained_hidden == nullptr ? nullptr : &(*chained_hidden)[0];
    router.finish_state(step, chained);
    return router.finish_logits(step, confidence);
}

std::vector<TokenId> MtpDraft::run_chain(const RankTensors& conditioning, TokenId first_token,
                                         std::uint32_t steps, std::int32_t start_position,
                                         float min_p) {
    std::vector<TokenId> tokens;
    tokens.reserve(steps);
    if (steps == 0) { return tokens; }

    const std::size_t bytes = hidden_elements * sizeof(std::uint16_t);
    Impl& router = *ranks_[0];
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(execution_->dev[0]->device));
    CUDA_CHECK(cudaMemcpy(router.chain_current.p, conditioning[0].data, bytes,
                          cudaMemcpyDeviceToDevice));
    void* current = router.chain_current.p;
    void* next = router.chain_next.p;
    const bool gated = min_p > 0.0F;
    for (std::uint32_t step = 0; step < steps; ++step) {
        RankTensors current_views{};
        RankTensors next_views{};
        current_views[0] = Tensor(current, DType::FP16, {2560, 4});
        next_views[0]    = Tensor(next, DType::FP16, {2560, 4});
        float confidence = 0.0F;
        first_token = run(current_views, first_token,
                          start_position + static_cast<std::int32_t>(step), &next_views,
                          gated ? &confidence : nullptr);
        tokens.push_back(first_token);
        std::swap(current, next);
        if (gated && confidence < min_p) { break; }
    }
    CUDA_CHECK(cudaSetDevice(previous));
    return tokens;
}

int MtpDraft::device() const noexcept { return ranks_[0]->device_context.device; }

std::uint64_t MtpDraft::expert_bytes() const noexcept { return ranks_[0]->expert_bytes; }

std::uint64_t MtpDraft::host_syncs() const noexcept { return ranks_[0]->host_syncs; }

} // namespace ninfer::targets::qwen3_8_flash_next::detail
