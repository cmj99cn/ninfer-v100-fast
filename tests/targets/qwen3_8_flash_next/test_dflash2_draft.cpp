// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

// DFlash2 drafter integration test. It loads the real companion, materializes its device objects,
// and compares `materialize` (target features -> per-layer context ring K/V) and `propose`
// (masked block -> target head top-16 -> conditional selector) against one independent naive FP64
// oracle at small real-shape workloads. The oracle evaluates the represented BF16 values and the
// exact decoded W8 codes/scales; it does not copy any production reduction tree or staging cast.
//
// To keep the propose oracle tractable the test zeroes the code planes of drafter layers 1..4
// after materialize. A layer whose q/k/v, output, gate/up and down projections are exactly zero is
// an exact identity on the residual (zeros flow through the norms, attention and convolutions), so
// only layer 0 contributes to the final hidden.

#include "targets/qwen3_8_flash_next/impl/dflash2_draft.h"

#include "artifact/materializer.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace detail = ninfer::targets::qwen3_8_flash_next::detail;

namespace {

constexpr int kHidden       = detail::DFlash2Geometry::hidden;
constexpr int kIntermediate = detail::DFlash2Geometry::intermediate;
constexpr int kHeadDim      = detail::DFlash2Geometry::head_dim;
constexpr int kQueryHeads   = detail::DFlash2Geometry::query_heads;
constexpr int kKvHeads      = detail::DFlash2Geometry::kv_heads;
constexpr int kQueryWidth   = detail::DFlash2Geometry::query_width;
constexpr int kKeyWidth     = detail::DFlash2Geometry::key_width;
constexpr int kGroups       = detail::DFlash2Geometry::conv_groups;
constexpr int kRank         = detail::DFlash2Geometry::selector_rank;
constexpr int kCandidates   = detail::DFlash2Geometry::candidates;
constexpr int kFeatureRows  = detail::DFlash2Geometry::feature_rows;
constexpr int kMaskToken    = detail::DFlash2Geometry::mask_token;
constexpr int kWindow       = static_cast<int>(detail::DFlash2Geometry::window);
constexpr int kLayers       = static_cast<int>(detail::DFlash2Geometry::layers);
constexpr float kEps        = detail::DFlash2Geometry::rms_epsilon;
constexpr float kTheta      = detail::DFlash2Geometry::rope_theta;
constexpr float kScale      = detail::DFlash2Geometry::attention_scale;
constexpr int kDomain       = 32;
constexpr int kAnchorToken  = 1000;

constexpr ReductionCriterion kCriterion{
    .relative_l2                     = 1.2e-2,
    .gross_absolute                  = 2.0e-2,
    .gross_relative_to_max_reference = 3.0e-2,
};

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

float bf(float value) { return bf16_to_f32(f32_to_bf16(value)); }

void round_bf(std::vector<float>& values) {
    for (auto& value : values) { value = bf(value); }
}

std::vector<double> to_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

std::vector<float> copy_bf16(const void* device, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    if (count != 0) {
        cuda_check(cudaMemcpy(bits.data(), device, count * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "copy_bf16");
    }
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = bf16_to_f32(bits[i]); }
    return out;
}

std::vector<float> decode_w8(const Weight& weight) {
    const int n = weight.n, k = weight.k;
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(n) * (k / 32));
    cuda_check(cudaMemcpy(codes.data(), weight.qdata, codes.size(), cudaMemcpyDeviceToHost),
               "decode_w8 codes");
    cuda_check(cudaMemcpy(scales.data(), weight.scales, scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "decode_w8 scales");
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < k / 32; ++group) {
            const float scale =
                f16_to_f32(scales[static_cast<std::size_t>(row) * (k / 32) + group]);
            for (int i = 0; i < 32; ++i) {
                const int column = group * 32 + i;
                const auto code =
                    static_cast<std::int8_t>(codes[static_cast<std::size_t>(row) * k + column]);
                out[static_cast<std::size_t>(row) * k + column] =
                    static_cast<float>(code) * scale;
            }
        }
    }
    return out;
}

std::vector<float> rows(const std::vector<float>& matrix, int begin, int count, int columns) {
    return std::vector<float>(matrix.begin() + static_cast<std::ptrdiff_t>(begin) * columns,
                              matrix.begin() +
                                  static_cast<std::ptrdiff_t>(begin + count) * columns);
}

// w [n, rows] row-major; x [rows, columns] with rows contiguous in each column.
std::vector<float> mm(const std::vector<float>& w, int n, int rows, const std::vector<float>& x,
                      int columns, bool bf16_out) {
    std::vector<float> out(static_cast<std::size_t>(n) * columns);
    for (int c = 0; c < columns; ++c) {
        for (int row = 0; row < n; ++row) {
            double sum = 0.0;
            for (int i = 0; i < rows; ++i) {
                sum += static_cast<double>(w[static_cast<std::size_t>(row) * rows + i]) *
                       x[static_cast<std::size_t>(c) * rows + i];
            }
            out[static_cast<std::size_t>(c) * n + row] =
                bf16_out ? bf(static_cast<float>(sum)) : static_cast<float>(sum);
        }
    }
    return out;
}

// x [d, columns]; weight [d].
std::vector<float> rmsnorm(const std::vector<float>& x, int d, int columns,
                           const std::vector<float>& weight, float eps) {
    std::vector<float> out(x.size());
    for (int c = 0; c < columns; ++c) {
        double squares = 0.0;
        for (int i = 0; i < d; ++i) {
            const double value = x[static_cast<std::size_t>(c) * d + i];
            squares += value * value;
        }
        const double inv = 1.0 / std::sqrt(squares / d + eps);
        for (int i = 0; i < d; ++i) {
            out[static_cast<std::size_t>(c) * d + i] =
                bf(static_cast<float>(x[static_cast<std::size_t>(c) * d + i] * inv * weight[i]));
        }
    }
    return out;
}

// x [heads, d] per column, flat column -> head -> d. Mirrors applying the d-wide norm to the
// logical [d, heads * columns] view.
std::vector<float> rmsnorm_heads(const std::vector<float>& x, int heads, int d, int columns,
                                 const std::vector<float>& weight, float eps) {
    std::vector<float> out(x.size());
    for (int c = 0; c < columns; ++c) {
        for (int head = 0; head < heads; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(c) * heads + head) * d;
            double squares = 0.0;
            for (int i = 0; i < d; ++i) {
                const double value = x[base + i];
                squares += value * value;
            }
            const double inv = 1.0 / std::sqrt(squares / d + eps);
            for (int i = 0; i < d; ++i) {
                out[base + i] = bf(static_cast<float>(x[base + i] * inv * weight[i]));
            }
        }
    }
    return out;
}

void rope(std::vector<float>& x, int heads, int columns, const std::vector<int>& positions) {
    const int d = kHeadDim;
    for (int c = 0; c < columns; ++c) {
        for (int h = 0; h < heads; ++h) {
            const std::size_t base = (static_cast<std::size_t>(c) * heads + h) * d;
            for (int j = 0; j < d / 2; ++j) {
                const double frequency =
                    std::pow(static_cast<double>(kTheta), -static_cast<double>(j) / (d / 2));
                const double angle  = positions[static_cast<std::size_t>(c)] * frequency;
                const double cosine = std::cos(angle), sine = std::sin(angle);
                const double x0 = x[base + j], x1 = x[base + j + d / 2];
                x[base + j]         = bf(static_cast<float>(x0 * cosine - x1 * sine));
                x[base + j + d / 2] = bf(static_cast<float>(x1 * cosine + x0 * sine));
            }
        }
    }
}

// x [H, W, B]; base [H,2,2] flat [side][tap][channel].
void conv_prepare(const std::vector<float>& x, int width, int batch,
                  const std::vector<float>& coeff, const std::vector<float>& base,
                  std::vector<float>& prepared,
                  std::vector<std::vector<float>>& finish_tap) {
    const int columns = width * batch;
    prepared.assign(static_cast<std::size_t>(kHidden) * columns, 0.0F);
    finish_tap.assign(2, std::vector<float>(static_cast<std::size_t>(kGroups) * columns, 0.0F));
    for (int c = 0; c < columns; ++c) {
        const int i = c % width;
        for (int h = 0; h < kHidden; ++h) {
            const int g = h / 16;
            const double base00 = base[static_cast<std::size_t>(0 * kHidden) + h];
            const double base01 = base[static_cast<std::size_t>(1 * kHidden) + h];
            double y = (base00 + coeff[static_cast<std::size_t>(c) * 4 * kGroups + g]) *
                       x[static_cast<std::size_t>(c) * kHidden + h];
            if (i > 0) {
                y += (base01 + coeff[static_cast<std::size_t>(c) * 4 * kGroups + kGroups + g]) *
                     x[static_cast<std::size_t>(c - 1) * kHidden + h];
            }
            prepared[static_cast<std::size_t>(c) * kHidden + h] = bf(static_cast<float>(y));
            finish_tap[0][static_cast<std::size_t>(c) * kGroups + g] =
                bf(coeff[static_cast<std::size_t>(c) * 4 * kGroups + 2 * kGroups + g]);
            finish_tap[1][static_cast<std::size_t>(c) * kGroups + g] =
                bf(coeff[static_cast<std::size_t>(c) * 4 * kGroups + 3 * kGroups + g]);
        }
    }
}

void conv_add(std::vector<float>& residual, const std::vector<float>& z, int width, int batch,
              const std::vector<float>& base, const std::vector<std::vector<float>>& finish_tap) {
    const int columns = width * batch;
    for (int c = 0; c < columns; ++c) {
        const int i = c % width;
        for (int h = 0; h < kHidden; ++h) {
            const int g = h / 16;
            const double base10 = base[static_cast<std::size_t>((1 * 2 + 0) * kHidden) + h];
            const double base11 = base[static_cast<std::size_t>((1 * 2 + 1) * kHidden) + h];
            double y = residual[static_cast<std::size_t>(c) * kHidden + h] +
                       (base10 + finish_tap[0][static_cast<std::size_t>(c) * kGroups + g]) *
                           z[static_cast<std::size_t>(c) * kHidden + h];
            if (i > 0) {
                y += (base11 + finish_tap[1][static_cast<std::size_t>(c) * kGroups + g]) *
                     z[static_cast<std::size_t>(c - 1) * kHidden + h];
            }
            residual[static_cast<std::size_t>(c) * kHidden + h] = bf(static_cast<float>(y));
        }
    }
}

// Non-causal attention over the whole live block (the context ring is empty; frontier 0).
std::vector<float> swa_block(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, int width, int batch) {
    const int columns = width * batch;
    std::vector<float> out(static_cast<std::size_t>(kHeadDim) * kQueryHeads * columns, 0.0F);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const int c = b * width + i;
            for (int hq = 0; hq < kQueryHeads; ++hq) {
                const int hk = hq / (kQueryHeads / kKvHeads);
                std::vector<double> scores(width);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int j = 0; j < width; ++j) {
                    const int cj = b * width + j;
                    double dot   = 0.0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        dot += static_cast<double>(
                                   q[(static_cast<std::size_t>(c) * kQueryHeads + hq) * kHeadDim +
                                     d]) *
                               k[(static_cast<std::size_t>(cj) * kKvHeads + hk) * kHeadDim + d];
                    }
                    scores[j] = dot * kScale;
                    maximum   = std::max(maximum, scores[j]);
                }
                double total = 0.0;
                for (int j = 0; j < width; ++j) {
                    scores[j] = std::exp(scores[j] - maximum);
                    total += scores[j];
                }
                for (int d = 0; d < kHeadDim; ++d) {
                    double sum = 0.0;
                    for (int j = 0; j < width; ++j) {
                        const int cj = b * width + j;
                        sum += scores[j] *
                               static_cast<double>(
                                   v[(static_cast<std::size_t>(cj) * kKvHeads + hk) * kHeadDim + d]);
                    }
                    out[(static_cast<std::size_t>(c) * kQueryHeads + hq) * kHeadDim + d] =
                        bf(static_cast<float>(sum / total));
                }
            }
        }
    }
    return out;
}

std::vector<float> silu_mul(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<float> out(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        out[i] = bf(static_cast<float>((g / (1.0 + std::exp(-g))) * up[i]));
    }
    return out;
}

struct Candidate {
    float value;
    int id;
};

std::vector<Candidate> top_k(const std::vector<float>& logits, int domain) {
    std::vector<Candidate> values(static_cast<std::size_t>(domain));
    for (int v = 0; v < domain; ++v) { values[static_cast<std::size_t>(v)] = {logits[v], v}; }
    std::partial_sort(values.begin(), values.begin() + kCandidates, values.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.value > b.value || (a.value == b.value && a.id < b.id);
                      });
    values.resize(kCandidates);
    return values;
}

void zero_codes(const Weight& weight) {
    cuda_check(cudaMemset(const_cast<void*>(weight.qdata), 0,
                          static_cast<std::size_t>(weight.n) * weight.k),
               "zero W8 codes");
}

std::vector<float> copy_codebook_row(const Tensor& codebook, int row) {
    const std::size_t bytes = static_cast<std::size_t>(kRank) * sizeof(std::uint16_t);
    std::vector<std::uint16_t> bits(kRank);
    cuda_check(cudaMemcpy(bits.data(),
                          static_cast<const std::uint8_t*>(codebook.data) +
                              static_cast<std::size_t>(row) * bytes,
                          bytes, cudaMemcpyDeviceToHost),
               "codebook row");
    std::vector<float> out(kRank);
    for (int i = 0; i < kRank; ++i) { out[static_cast<std::size_t>(i)] = bf16_to_f32(bits[i]); }
    return out;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "dflash2_draft: SKIP (CUDA unavailable)\n";
        return 77;
    }
    const char* value = std::getenv("NINFER_DFLASH2_ARTIFACT");
    if (value == nullptr || value[0] == '\0') {
        std::cerr << "NINFER_DFLASH2_ARTIFACT is not set\n";
        return 77;
    }

    try {
        detail::DFlash2Draft companion(value, detail::DFlash2Placement::Device);
        ExecutionContext execution({0});
        auto& device = execution.primary();
        auto materialized = artifact::materialize(companion.reader(), companion.plan(), execution);
        device.synchronize();

        auto operands = detail::bind_dflash2_operands(companion, materialized, 0);

        // Synthetic target token rows and head. The monotone head (row v = v + 1) gives a wide,
        // numerically stable top-16 so the selector's discrete outcome is not an accuracy lottery.
        std::vector<float> anchor_row(kHidden), mask_row(kHidden);
        fill_uniform(anchor_row, 9101u, -0.5F, 0.5F);
        fill_uniform(mask_row, 9103u, -0.5F, 0.5F);
        round_bf(anchor_row);
        round_bf(mask_row);
        std::vector<std::uint16_t> anchor_bits(kHidden), mask_bits(kHidden);
        for (int h = 0; h < kHidden; ++h) {
            anchor_bits[static_cast<std::size_t>(h)] = f32_to_bf16(anchor_row[static_cast<std::size_t>(h)]);
            mask_bits[static_cast<std::size_t>(h)]   = f32_to_bf16(mask_row[static_cast<std::size_t>(h)]);
        }
        DeviceBuffer embedding_table(static_cast<std::size_t>(companion.vocab()) * kHidden * 2);
        cuda_check(cudaMemset(embedding_table.p, 0, embedding_table.bytes), "zero embedding");
        cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(embedding_table.p) +
                                  static_cast<std::size_t>(kAnchorToken) * kHidden * 2,
                              anchor_bits.data(), kHidden * 2, cudaMemcpyHostToDevice),
                   "embedding anchor row");
        cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(embedding_table.p) +
                                  static_cast<std::size_t>(kMaskToken) * kHidden * 2,
                              mask_bits.data(), kHidden * 2, cudaMemcpyHostToDevice),
                   "embedding mask row");
        Weight embedding{};
        embedding.payload         = embedding_table.p;
        embedding.payload_bytes   = embedding_table.bytes;
        embedding.qtype           = QType::BF16_CTRL;
        embedding.layout          = QuantLayout::Contiguous;
        embedding.ndim            = 2;
        embedding.n               = companion.vocab();
        embedding.k               = kHidden;
        embedding.shape[0]        = companion.vocab();
        embedding.shape[1]        = kHidden;
        embedding.padded_shape[0] = companion.vocab();
        embedding.padded_shape[1] = kHidden;
        embedding.qdata           = embedding_table.p;
        operands.embedding        = embedding;

        std::vector<float> head_host(static_cast<std::size_t>(kDomain) * kHidden);
        for (int v = 0; v < kDomain; ++v) {
            for (int h = 0; h < kHidden; ++h) {
                head_host[static_cast<std::size_t>(v) * kHidden + h] = static_cast<float>(v + 1);
            }
        }
        std::vector<std::uint16_t> head_bits(head_host.size());
        for (std::size_t i = 0; i < head_bits.size(); ++i) {
            head_bits[i] = f32_to_bf16(head_host[i]);
        }
        DeviceBuffer head_storage(head_bits.size() * sizeof(std::uint16_t));
        head_storage.copy_from_host(head_bits.data(), head_storage.bytes);
        Weight head_weight{};
        head_weight.payload        = head_storage.p;
        head_weight.payload_bytes  = head_storage.bytes;
        head_weight.qtype          = QType::BF16_CTRL;
        head_weight.layout         = QuantLayout::Contiguous;
        head_weight.ndim           = 2;
        head_weight.n              = kDomain;
        head_weight.k              = kHidden;
        head_weight.shape[0]       = kDomain;
        head_weight.shape[1]       = kHidden;
        head_weight.padded_shape[0] = kDomain;
        head_weight.padded_shape[1] = kHidden;
        head_weight.qdata          = head_storage.p;

        const std::size_t ring_elements =
            static_cast<std::size_t>(kHeadDim) * kWindow * kKvHeads;
        auto make_ring = [&]() {
            struct Ring {
                std::vector<DeviceBuffer> k, v;
                std::vector<CyclicKVCacheLayerView> views;
            } result;
            result.k.resize(kLayers);
            result.v.resize(kLayers);
            for (int layer = 0; layer < kLayers; ++layer) {
                result.k[layer] = DeviceBuffer(ring_elements * sizeof(std::uint16_t));
                result.v[layer] = DeviceBuffer(ring_elements * sizeof(std::uint16_t));
                cuda_check(cudaMemset(result.k[layer].p, 0, result.k[layer].bytes), "zero ring k");
                cuda_check(cudaMemset(result.v[layer].p, 0, result.v[layer].bytes), "zero ring v");
                result.views.push_back({
                    .k = Tensor(result.k[layer].p, DType::BF16, {kHeadDim, kWindow, kKvHeads, 1}),
                    .v = Tensor(result.v[layer].p, DType::BF16, {kHeadDim, kWindow, kKvHeads, 1}),
                    .capacity = kWindow,
                    .padded_capacity = kWindow,
                    .num_kv_heads = kKvHeads,
                    .head_dim = kHeadDim,
                    .lane_capacity = 1,
                });
            }
            return result;
        };

        auto head_callback = [&](const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                                 cudaStream_t stream) {
            Tensor small = work.alloc(DType::BF16, {kDomain, hidden.ne[1]});
            ops::linear(hidden, head_weight, small, stream);
            CUDA_CHECK(cudaMemcpy2DAsync(
                logits.data, static_cast<std::size_t>(companion.vocab()) * sizeof(std::uint16_t),
                small.data, static_cast<std::size_t>(kDomain) * sizeof(std::uint16_t),
                static_cast<std::size_t>(kDomain) * sizeof(std::uint16_t),
                static_cast<std::size_t>(hidden.ne[1]), cudaMemcpyDeviceToDevice, stream));
        };

        // ------------------------------------------------------------------ materialize oracle
        {
            auto ring = make_ring();
            detail::DFlash2Drafter drafter(operands, companion.vocab(), kDomain, ring.views, 4096,
                                           head_callback);
            constexpr int kWidth = 3, kBatch = 1, kColumns = kWidth * kBatch;
            std::vector<float> features(static_cast<std::size_t>(kFeatureRows) * kColumns);
            fill_uniform(features, 9203u, -0.5F, 0.5F);
            round_bf(features);
            const std::vector<int> positions{0, 1, 2};
            const std::vector<int> counts{3}, lanes{0};
            DeviceBuffer d_features = to_device_bf16(features);
            DeviceBuffer d_positions = to_device(positions);
            DeviceBuffer d_counts = to_device(counts);
            DeviceBuffer d_lanes = to_device(lanes);
            WorkspaceArena work(
                detail::DFlash2Drafter::materialize_workspace_bytes(kBatch, kWidth));
            detail::DFlash2ContextCall call{
                .features = Tensor(d_features.p, DType::BF16, {kFeatureRows, kWidth, kBatch}),
                .positions = Tensor(d_positions.p, DType::I32, {kWidth, kBatch}),
                .counts = Tensor(d_counts.p, DType::I32, {kBatch}),
                .lanes = Tensor(d_lanes.p, DType::I32, {kBatch}),
                .min_count = 3,
                .max_count = 3,
            };
            drafter.materialize(call, work, device.stream);
            device.synchronize();

            const auto feature_w = decode_w8(operands.feature_projection);
            const auto projected = mm(feature_w, kHidden, kFeatureRows, features, kColumns, true);
            const auto context_norm = copy_bf16(operands.context_norm.data, kHidden);
            const auto context = rmsnorm(projected, kHidden, kColumns, context_norm, kEps);

            for (int layer = 0; layer < kLayers; ++layer) {
                const auto& layer_operands = operands.layers[static_cast<std::size_t>(layer)];
                const auto key_w   = decode_w8(layer_operands.context_key);
                const auto value_w = decode_w8(layer_operands.context_value);
                const auto key_norm = copy_bf16(layer_operands.key_norm.data, kHeadDim);
                auto key = mm(key_w, kKeyWidth, kHidden, context, kColumns, true);
                const auto value = mm(value_w, kKeyWidth, kHidden, context, kColumns, true);
                key = rmsnorm_heads(key, kKvHeads, kHeadDim, kColumns, key_norm, kEps);
                rope(key, kKvHeads, kColumns, positions);

                const auto device_k =
                    from_device<std::uint16_t>(ring.k[layer].p, ring_elements);
                const auto device_v =
                    from_device<std::uint16_t>(ring.v[layer].p, ring_elements);
                std::vector<float> live_k, live_v, expect_k, expect_v;
                for (int c = 0; c < kColumns; ++c) {
                    const int slot = positions[static_cast<std::size_t>(c)] & (kWindow - 1);
                    for (int head = 0; head < kKvHeads; ++head) {
                        const std::size_t base =
                            static_cast<std::size_t>(kHeadDim) *
                            (slot + static_cast<std::size_t>(kWindow) * head);
                        for (int d = 0; d < kHeadDim; ++d) {
                            expect_k.push_back(
                                key[(static_cast<std::size_t>(c) * kKvHeads + head) * kHeadDim + d]);
                            expect_v.push_back(
                                value[(static_cast<std::size_t>(c) * kKvHeads + head) * kHeadDim +
                                      d]);
                            live_k.push_back(bf16_to_f32(device_k[base + d]));
                            live_v.push_back(bf16_to_f32(device_v[base + d]));
                        }
                    }
                }
                failures += verify_reduction("materialize K layer " + std::to_string(layer),
                                             to_double(live_k), to_double(expect_k), kCriterion);
                failures += verify_reduction("materialize V layer " + std::to_string(layer),
                                             to_double(live_v), to_double(expect_v), kCriterion);
            }
        }

        // Layers 1..4 become exact identities: their q/k/v, output, gate/up and down code planes
        // are zero, so every projection is exactly zero and the residual is unchanged.
        for (int layer = 1; layer < kLayers; ++layer) {
            const auto& source = operands.layers[static_cast<std::size_t>(layer)];
            zero_codes(source.query_key_value);
            zero_codes(source.attention_output);
            zero_codes(source.gate_up);
            zero_codes(source.down);
        }
        device.synchronize();

        // ---------------------------------------------------------------------- propose oracle
        auto ring = make_ring();
        constexpr int kWidth = 3, kBatch = 1, kSteps = kWidth - 1, kColumns = kWidth * kBatch;
        constexpr int kMasks = kSteps * kBatch;
        DeviceBuffer hidden_capture(static_cast<std::size_t>(kHidden) * kMasks * 2);
        detail::DFlash2Drafter drafter(operands, companion.vocab(), kDomain, ring.views, 4096,
                                       [&](const Tensor& hidden, Tensor& logits,
                                           WorkspaceArena& work, cudaStream_t stream) {
                                           CUDA_CHECK(cudaMemcpyAsync(
                                               hidden_capture.p, hidden.data, hidden.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
                                           head_callback(hidden, logits, work, stream);
                                       });
        std::vector<int> anchors{kAnchorToken}, frontiers{0}, valid_columns{kWidth}, lanes{0};
        ops::SamplingConfig config{};
        config.temperature = 0.0F;
        config.seed        = 1234567u;
        DeviceBuffer d_anchors = to_device(anchors);
        DeviceBuffer d_frontiers = to_device(frontiers);
        DeviceBuffer d_valid = to_device(valid_columns);
        DeviceBuffer d_lanes = to_device(lanes);
        DeviceBuffer d_config(sizeof(config));
        d_config.copy_from_host(&config, sizeof(config));
        DeviceBuffer d_ids(static_cast<std::size_t>(kWidth) * kBatch * sizeof(int));
        DeviceBuffer d_positions(static_cast<std::size_t>(kWidth) * kBatch * sizeof(int));
        DeviceBuffer d_drafts(static_cast<std::size_t>(kSteps) * kBatch * sizeof(int));
        DeviceBuffer d_candidates(static_cast<std::size_t>(kCandidates) * kMasks * sizeof(int));
        DeviceBuffer d_proposal_q(static_cast<std::size_t>(kCandidates) * kMasks * sizeof(float));
        DeviceBuffer d_confidence(static_cast<std::size_t>(kSteps) * kBatch * sizeof(float));
        WorkspaceArena work(
            detail::DFlash2Drafter::propose_workspace_bytes(kBatch, kWidth, 4096, 0));
        detail::DFlash2ProposalCall call{
            .anchors = Tensor(d_anchors.p, DType::I32, {kBatch}),
            .frontiers = Tensor(d_frontiers.p, DType::I32, {kBatch}),
            .valid_columns = Tensor(d_valid.p, DType::I32, {kBatch}),
            .lanes = Tensor(d_lanes.p, DType::I32, {kBatch}),
            .configs = static_cast<const ops::SamplingConfig*>(d_config.p),
            .batch = kBatch,
            .width = kWidth,
            .ids = Tensor(d_ids.p, DType::I32, {kWidth, kBatch}),
            .positions = Tensor(d_positions.p, DType::I32, {kWidth, kBatch}),
            .drafts = Tensor(d_drafts.p, DType::I32, {kSteps, kBatch}),
            .candidates = Tensor(d_candidates.p, DType::I32, {kCandidates, kSteps, kBatch}),
            .proposal_q = Tensor(d_proposal_q.p, DType::FP32, {kCandidates, kSteps, kBatch}),
            .confidence = Tensor(d_confidence.p, DType::FP32, {kSteps, kBatch}),
            .edge_softcap = 0.0F,
        };
        drafter.propose(call, work, device.stream);
        device.synchronize();

        const auto captured = copy_bf16(hidden_capture.p, static_cast<std::size_t>(kHidden) * kMasks);

        // Independent layer-0 CPU oracle. Layers 1..4 are exact identities.
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * kColumns);
        for (int c = 0; c < kColumns; ++c) {
            const auto& row = c == 0 ? anchor_row : mask_row;
            for (int h = 0; h < kHidden; ++h) {
                residual[static_cast<std::size_t>(c) * kHidden + h] = row[static_cast<std::size_t>(h)];
            }
        }
        const auto& layer = operands.layers[0];
        const auto input_norm = copy_bf16(layer.input_norm.data, kHidden);
        const auto post_norm = copy_bf16(layer.post_attention_norm.data, kHidden);
        const auto query_norm = copy_bf16(layer.query_norm.data, kHeadDim);
        const auto key_norm = copy_bf16(layer.key_norm.data, kHeadDim);
        const auto final_norm = copy_bf16(operands.final_norm.data, kHidden);
        const auto attention_base = copy_bf16(layer.attention_base.data, kHidden * 4);
        const auto mlp_base = copy_bf16(layer.mlp_base.data, kHidden * 4);
        const auto attention_conv =
            copy_bf16(layer.attention_conv_kernel_projection.qdata, 4 * kGroups * kHidden);
        const auto mlp_conv = copy_bf16(layer.mlp_conv_kernel_projection.qdata, 4 * kGroups * kHidden);
        const auto qkv_w = decode_w8(layer.query_key_value);
        const auto attention_output_w = decode_w8(layer.attention_output);
        const auto gate_up_w = decode_w8(layer.gate_up);
        const auto down_w = decode_w8(layer.down);
        std::vector<int> model_positions{0, 1, 2};

        // Attention branch.
        {
            const auto normed = rmsnorm(residual, kHidden, kColumns, input_norm, kEps);
            const auto coeff = mm(attention_conv, 4 * kGroups, kHidden, normed, kColumns, false);
            std::vector<float> prepared;
            std::vector<std::vector<float>> finish;
            conv_prepare(normed, kWidth, kBatch, coeff, attention_base, prepared, finish);
            auto q = mm(rows(qkv_w, 0, kQueryWidth, kHidden), kQueryWidth, kHidden, prepared,
                        kColumns, true);
            auto k = mm(rows(qkv_w, kQueryWidth, kKeyWidth, kHidden), kKeyWidth, kHidden, prepared,
                        kColumns, true);
            const auto v = mm(rows(qkv_w, kQueryWidth + kKeyWidth, kKeyWidth, kHidden), kKeyWidth,
                              kHidden, prepared, kColumns, true);
            q = rmsnorm_heads(q, kQueryHeads, kHeadDim, kColumns, query_norm, kEps);
            k = rmsnorm_heads(k, kKvHeads, kHeadDim, kColumns, key_norm, kEps);
            rope(q, kQueryHeads, kColumns, model_positions);
            rope(k, kKvHeads, kColumns, model_positions);
            const auto attended = swa_block(q, k, v, kWidth, kBatch);
            const auto z =
                mm(attention_output_w, kHidden, kQueryWidth, attended, kColumns, true);
            conv_add(residual, z, kWidth, kBatch, attention_base, finish);
        }
        // MLP branch.
        {
            const auto normed = rmsnorm(residual, kHidden, kColumns, post_norm, kEps);
            const auto coeff = mm(mlp_conv, 4 * kGroups, kHidden, normed, kColumns, false);
            std::vector<float> prepared;
            std::vector<std::vector<float>> finish;
            conv_prepare(normed, kWidth, kBatch, coeff, mlp_base, prepared, finish);
            const auto gate = mm(rows(gate_up_w, 0, kIntermediate, kHidden), kIntermediate, kHidden,
                                 prepared, kColumns, true);
            const auto up = mm(rows(gate_up_w, kIntermediate, kIntermediate, kHidden),
                               kIntermediate, kHidden, prepared, kColumns, true);
            const auto product = silu_mul(gate, up);
            const auto z = mm(down_w, kHidden, kIntermediate, product, kColumns, true);
            conv_add(residual, z, kWidth, kBatch, mlp_base, finish);
        }
        const auto final_hidden = rmsnorm(residual, kHidden, kColumns, final_norm, kEps);
        std::vector<float> expected_hidden(static_cast<std::size_t>(kHidden) * kMasks);
        for (int m = 0; m < kMasks; ++m) {
            for (int h = 0; h < kHidden; ++h) {
                expected_hidden[static_cast<std::size_t>(m) * kHidden + h] =
                    final_hidden[static_cast<std::size_t>(m + 1) * kHidden + h];
            }
        }
        failures += verify_reduction("propose final hidden", to_double(captured),
                                     to_double(expected_hidden), kCriterion);

        // Head top-16 and the conditional selector chain, evaluated from the represented BF16
        // hidden (the actual head input) so the discrete path is checked at its own boundary.
        std::vector<std::vector<Candidate>> oracle_candidates(
            kMasks);
        for (int m = 0; m < kMasks; ++m) {
            std::vector<float> logits(kDomain);
            for (int v = 0; v < kDomain; ++v) {
                double sum = 0.0;
                for (int h = 0; h < kHidden; ++h) {
                    sum += static_cast<double>(head_host[static_cast<std::size_t>(v) * kHidden + h]) *
                           captured[static_cast<std::size_t>(m) * kHidden + h];
                }
                logits[static_cast<std::size_t>(v)] = bf(static_cast<float>(sum));
            }
            oracle_candidates[static_cast<std::size_t>(m)] = top_k(logits, kDomain);
        }

        const auto device_candidates =
            from_device<std::int32_t>(d_candidates.p, static_cast<std::size_t>(kCandidates) * kMasks);
        bool candidates_ok = true;
        for (int m = 0; m < kMasks; ++m) {
            for (int c = 0; c < kCandidates; ++c) {
                candidates_ok =
                    candidates_ok &&
                    device_candidates[static_cast<std::size_t>(m) * kCandidates + c] ==
                        oracle_candidates[static_cast<std::size_t>(m)]
                                         [static_cast<std::size_t>(c)]
                                             .id;
            }
        }
        check(candidates_ok, "propose candidates: top-16 differs from the oracle");

        // selector_projection (BF16 [rank,H]) over the represented hidden.
        const auto selector_bits = copy_bf16(operands.selector_projection.qdata,
                                             static_cast<std::size_t>(kRank) * kHidden);
        const auto projected = mm(selector_bits, kRank, kHidden, captured, kMasks, true);
        const auto predecessor = copy_codebook_row(operands.predecessor_codebook, kAnchorToken);
        std::vector<std::vector<float>> successors(kDomain);
        for (int v = 0; v < kDomain; ++v) {
            successors[static_cast<std::size_t>(v)] =
                copy_codebook_row(operands.successor_codebook, v);
        }

        std::vector<int> oracle_drafts(kSteps);
        std::vector<float> oracle_q(static_cast<std::size_t>(kCandidates) * kSteps);
        std::vector<float> oracle_confidence(kSteps);
        int parent = kAnchorToken;
        for (int step = 0; step < kSteps; ++step) {
            const auto& column_candidates =
                oracle_candidates[static_cast<std::size_t>(step)];
            std::vector<double> edges(kCandidates);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int c = 0; c < kCandidates; ++c) {
                const int id = column_candidates[static_cast<std::size_t>(c)].id;
                const auto& successor = successors[static_cast<std::size_t>(id)];
                double sum = 0.0;
                for (int r = 0; r < kRank; ++r) {
                    sum += static_cast<double>(predecessor[static_cast<std::size_t>(r)]) *
                           projected[static_cast<std::size_t>(step) * kRank + r] *
                           successor[static_cast<std::size_t>(r)];
                }
                const double edge = sum + column_candidates[static_cast<std::size_t>(c)].value;
                edges[static_cast<std::size_t>(c)] = edge;
                maximum = std::max(maximum, edge);
            }
            int selected = 0;
            for (int c = 1; c < kCandidates; ++c) {
                if (edges[static_cast<std::size_t>(c)] > edges[static_cast<std::size_t>(selected)]) {
                    selected = c;
                }
            }
            oracle_drafts[static_cast<std::size_t>(step)] =
                column_candidates[static_cast<std::size_t>(selected)].id;
            parent      = oracle_drafts[static_cast<std::size_t>(step)];
            double total = 0.0;
            for (int c = 0; c < kCandidates; ++c) {
                const double weight = std::exp(edges[static_cast<std::size_t>(c)] - maximum);
                oracle_q[static_cast<std::size_t>(step) * kCandidates + c] =
                    static_cast<float>(weight);
                total += weight;
            }
            for (int c = 0; c < kCandidates; ++c) {
                oracle_q[static_cast<std::size_t>(step) * kCandidates + c] =
                    c == selected
                        ? 1.0F
                        : 0.0F;
            }
            oracle_confidence[static_cast<std::size_t>(step)] =
                static_cast<float>(std::exp(edges[static_cast<std::size_t>(selected)] - maximum) /
                                   total);
        }

        const auto device_drafts =
            from_device<std::int32_t>(d_drafts.p, static_cast<std::size_t>(kSteps));
        const auto device_q =
            from_device<float>(d_proposal_q.p, static_cast<std::size_t>(kCandidates) * kSteps);
        const auto device_confidence =
            from_device<float>(d_confidence.p, static_cast<std::size_t>(kSteps));
        bool drafts_ok = true;
        for (int step = 0; step < kSteps; ++step) {
            drafts_ok =
                drafts_ok &&
                device_drafts[static_cast<std::size_t>(step)] ==
                    oracle_drafts[static_cast<std::size_t>(step)];
        }
        check(drafts_ok, "propose drafts: selector chain differs from the oracle");
        failures += verify_reduction(
            "propose proposal_q", to_double(device_q), to_double(oracle_q),
            ReductionCriterion{.relative_l2 = 1.0e-6, .gross_absolute = 1.0e-6,
                               .gross_relative_to_max_reference = 1.0e-6});
        failures += verify_reduction(
            "propose confidence", to_double(device_confidence), to_double(oracle_confidence),
            ReductionCriterion{.relative_l2 = 1.0e-3, .gross_absolute = 1.0e-3,
                               .gross_relative_to_max_reference = 1.0e-3});

        // ------------------------------------------------------------- determinism, real companion
        {
            auto ring2 = make_ring();
            detail::DFlash2Drafter again(operands, companion.vocab(), kDomain, ring2.views, 4096,
                                         head_callback);
            DeviceBuffer drafts2(static_cast<std::size_t>(kSteps) * kBatch * sizeof(int));
            DeviceBuffer q2(static_cast<std::size_t>(kCandidates) * kMasks * sizeof(float));
            DeviceBuffer confidence2(static_cast<std::size_t>(kSteps) * kBatch * sizeof(float));
            detail::DFlash2ProposalCall call2 = call;
            call2.drafts = Tensor(drafts2.p, DType::I32, {kSteps, kBatch});
            call2.proposal_q = Tensor(q2.p, DType::FP32, {kCandidates, kSteps, kBatch});
            call2.confidence = Tensor(confidence2.p, DType::FP32, {kSteps, kBatch});
            WorkspaceArena work2(
                detail::DFlash2Drafter::propose_workspace_bytes(kBatch, kWidth, 4096, 0));
            again.propose(call2, work2, device.stream);
            device.synchronize();
            const auto drafts_b =
                from_device<std::int32_t>(drafts2.p, static_cast<std::size_t>(kSteps));
            const auto q_b = from_device<float>(q2.p, static_cast<std::size_t>(kCandidates) * kSteps);
            const auto confidence_b =
                from_device<float>(confidence2.p, static_cast<std::size_t>(kSteps));
            bool identical = std::equal(device_drafts.begin(), device_drafts.end(), drafts_b.begin());
            identical = identical && std::equal(device_q.begin(), device_q.end(), q_b.begin());
            identical = identical &&
                        std::equal(device_confidence.begin(), device_confidence.end(),
                                   confidence_b.begin());
            check(identical, "propose determinism: repeated real-companion run differs");
        }
    } catch (const std::exception& error) {
        std::cerr << "dflash2_draft: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0) {
        std::cerr << "dflash2_draft failures=" << failures << '\n';
        return 1;
    }
    std::cout << "dflash2_draft: PASS\n";
    return 0;
}
