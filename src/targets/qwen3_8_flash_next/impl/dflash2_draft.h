#pragma once

// Ported and adapted from iamwavecut/ninfer-all, commit 9ba1276
// (https://github.com/iamwavecut/ninfer-all), Apache License, Version 2.0.
// Modified for the NInfer qwen4_exp family runtime.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::targets::qwen3_8_flash_next::detail {

// The main-artifact identity the companion's target_config must describe. The main artifact is
// already admitted against these values by the flash-next text binding plan.
struct DFlash2TargetConfig {
    std::string architecture;
    std::uint32_t hidden = 0;
    std::uint32_t vocab  = 0;
};

// The companion's fixed trained geometry. The loader validates every value against the artifact
// before a single object is bound, so the drafter can size its planes from these constants.
struct DFlash2Geometry {
    static constexpr std::int32_t layers         = 5;
    static constexpr std::int32_t hidden         = 2560;
    static constexpr std::int32_t intermediate   = 7680;
    static constexpr std::int32_t query_heads    = 32;
    static constexpr std::int32_t kv_heads       = 8;
    static constexpr std::int32_t head_dim       = 128;
    static constexpr std::int32_t query_width    = query_heads * head_dim;
    static constexpr std::int32_t key_width      = kv_heads * head_dim;
    static constexpr std::int32_t conv_groups    = hidden / 16;
    static constexpr std::int32_t feature_rows   = 5 * hidden;
    static constexpr std::int32_t vocab          = 248320;
    static constexpr std::int32_t selector_rank  = 256;
    static constexpr std::int32_t candidates     = 16;
    static constexpr std::int32_t mask_token     = 248077;
    static constexpr std::uint32_t window        = 2048;
    // The trained block proposes at most seven drafts; the public option accepts [1,7].
    static constexpr std::uint32_t maximum_draft_window = 7;
    // Ring lanes the drafter serves. One Engine owns one sequence at this target.
    static constexpr std::int32_t lane_capacity = 1;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 10000.0F;
    static constexpr float attention_scale       = 0.08838834764831845F; // 1 / sqrt(128)
};

// How the companion is placed when it is bound. The load-only probe (the default) validates the
// whole projection without touching device memory; the drafter's device load places every object.
enum class DFlash2Placement : std::uint8_t {
    ValidateOnly,
    Device,
};

struct BoundDFlash2Tensor {
    std::string name;
    std::string object;
    artifact::ObjectHandle handle;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    std::vector<std::uint64_t> shape;
};

void validate_dflash2_target_config(const std::map<std::string, std::string>& config);

// Owns the standalone DFlash2 companion: its own Reader, its complete read-only binder plan and
// the resolved drafter configuration. With ValidateOnly (the default) every object is validated
// and the plan allocates nothing; with Device every object is placed for a later
// artifact::materialize. The weights stay at artifact-native precision either way.
class DFlash2Draft {
public:
    explicit DFlash2Draft(const std::filesystem::path& path,
                          DFlash2Placement placement = DFlash2Placement::ValidateOnly);

    [[nodiscard]] const artifact::Reader& reader() const noexcept { return reader_; }
    [[nodiscard]] const std::vector<BoundDFlash2Tensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const artifact::MaterializationPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const DFlash2TargetConfig& target() const noexcept { return target_; }

    [[nodiscard]] std::string_view target_arch() const noexcept { return target_.architecture; }
    [[nodiscard]] std::uint32_t hidden() const noexcept { return target_.hidden; }
    [[nodiscard]] std::uint32_t vocab() const noexcept { return target_.vocab; }
    // The number of drafter transformer layers in this companion, not the target's.
    [[nodiscard]] std::uint32_t layers() const noexcept { return drafter_layers_; }
    [[nodiscard]] const std::vector<std::uint32_t>& target_layer_ids() const noexcept {
        return target_layer_ids_;
    }

private:
    artifact::Reader reader_;
    std::vector<BoundDFlash2Tensor> tensors_;
    artifact::MaterializationPlan plan_;
    DFlash2TargetConfig target_;
    std::uint32_t drafter_layers_ = 0;
    std::vector<std::uint32_t> target_layer_ids_;
};

// One drafter layer's operands. The query/key/value stream is the companion's packed
// [6144,2560] W8 parent; context K/V alias its key/value rows (the single-matrix package this
// companion ships). The convolution's two-tap coefficients come from the BF16 kernel projection
// and the BF16 [H,2,2] base kernel.
struct DFlash2LayerOperands {
    Tensor input_norm;          // BF16 [H]
    Tensor post_attention_norm; // BF16 [H]
    Tensor query_norm;          // BF16 [D]
    Tensor key_norm;            // BF16 [D]
    Tensor attention_base;      // BF16 [H,2,2]
    Tensor mlp_base;            // BF16 [H,2,2]
    Weight query_key_value;     // W8 [6144, H]
    Weight context_key;         // W8 [1024, H], aliases the parent's key rows
    Weight context_value;       // W8 [1024, H], aliases the parent's value rows
    Weight attention_output;    // W8 [H, 4096]
    Weight gate_up;             // W8 [15360, H]
    Weight down;                // W8 [H, 7680]
    Weight attention_conv_kernel_projection; // BF16 [4H/16, H]
    Weight mlp_conv_kernel_projection;       // BF16 [4H/16, H]
};

struct DFlash2Operands {
    Tensor context_norm;   // BF16 [H]
    Tensor final_norm;     // BF16 [H]
    Weight feature_projection;         // W8 [H, 5H]
    Weight selector_projection;        // BF16 [256, H]
    Tensor predecessor_codebook;       // BF16 [vocab, 256]
    Tensor successor_codebook;         // BF16 [vocab, 256]
    Weight embedding;                  // the target text model's token rows [vocab, H]
    std::vector<DFlash2LayerOperands> layers;
};

// Wraps the companion's materialized device objects in the semantic operand views. `materialized`
// must have been produced from this companion's Device plan.
[[nodiscard]] DFlash2Operands bind_dflash2_operands(const DFlash2Draft& draft,
                                                    const artifact::MaterializedArtifact& materialized,
                                                    int device = 0);

// The target's head over the block's final hidden (BF16 [H, columns]) into `logits`
// (BF16 [vocab, columns]): the target head, exactly as the step-3 program supplies it.
using DFlash2Head =
    std::function<void(const Tensor& hidden, Tensor& logits, WorkspaceArena&, cudaStream_t)>;

// Committed target features into the ring: row b's first counts[b] columns (positions[i,b]) are
// stored in lane lanes[b]; its other columns are inert.
struct DFlash2ContextCall {
    Tensor features;  // BF16 [5H, W, B]
    Tensor positions; // I32 [W, B]
    Tensor counts;    // I32 [B]
    Tensor lanes;     // I32 [B]
    std::uint32_t min_count = 0, max_count = 0;
};

// One proposal per row: the block [anchor, mask x K] at positions frontier .. frontier + K against
// the ring's committed context [frontier - window, frontier).
struct DFlash2ProposalCall {
    Tensor anchors;       // I32 [B]
    Tensor frontiers;     // I32 [B]
    Tensor valid_columns; // I32 [B]
    Tensor lanes;         // I32 [B]
    const ops::SamplingConfig* configs = nullptr; // device [B]
    std::int32_t batch = 0, width = 0;            // W = K + 1 in [2, 16]
    Tensor ids, positions;                        // I32 [W, B] scratch
    Tensor drafts;                                // I32 [K, B]
    Tensor candidates;                            // I32 [16, K, B]
    Tensor proposal_q;                            // FP32 [16, K, B]
    Tensor confidence;                            // FP32 [K, B]; empty data disables it
    float edge_softcap = 0.0F;
};

// The DFlash2 drafter. The ring holds one layer view per drafter layer (capacity the window).
// Every call enqueues on the given stream and allocates only from the given workspace, so a CUDA
// graph may capture it.
class DFlash2Drafter {
public:
    DFlash2Drafter(DFlash2Operands operands, std::int32_t vocab, std::int32_t domain,
                   std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context,
                   DFlash2Head head);

    void materialize(const DFlash2ContextCall& call, WorkspaceArena& work,
                     cudaStream_t stream) const;
    void propose(const DFlash2ProposalCall& call, WorkspaceArena& work, cudaStream_t stream) const;

    // Workspace of materialize at up to `max_width` columns per row in rows of up to max_batch.
    [[nodiscard]] static std::size_t materialize_workspace_bytes(std::int32_t max_batch,
                                                                 std::int32_t max_width);
    // Workspace of propose at widths [2, max_width] over rows of up to max_batch beside
    // `head_bytes` (the head's own workspace at up to (max_width - 1) * max_batch columns).
    [[nodiscard]] static std::size_t propose_workspace_bytes(std::int32_t max_batch,
                                                             std::int32_t max_width,
                                                             std::uint32_t max_context,
                                                             std::size_t head_bytes);

    [[nodiscard]] std::int32_t taps() const noexcept {
        return static_cast<std::int32_t>(p_.layers.size());
    }

private:
    DFlash2Operands p_;
    std::int32_t hidden_      = DFlash2Geometry::hidden;
    std::int32_t vocab_       = 0;
    std::int32_t domain_      = 0;
    std::vector<CyclicKVCacheLayerView> ring_;
    std::uint32_t max_context_ = 0;
    DFlash2Head head_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
