#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// The QSA projection payloads: the two row-split parents consumed as row ranges, and the four
// per-head RMSNorm gains.
struct AttentionProjectionWeights {
    Weight query_gate;
    Weight key_value;
    Tensor query_norm;
    Tensor key_norm;
    Tensor index_query_norm;
    Tensor index_key_norm;
    Weight output;
};

// The GDN projection and control payloads. `qkvz` is the four-segment row-split parent and
// `convolution` is the tap-major artifact object seen as the channel-major kernel view the Op
// consumes.
struct GdnProjectionWeights {
    Weight qkvz;
    Weight control;
    Tensor a_log;
    Tensor dt_bias;
    Tensor convolution;
    Tensor norm;
    Weight output;
};

// The post-mixer payload is the MoE Op's own weight view: the router plus shared-score parent rows,
// the shared down projection, and the two batched banks consumed in place with the gate and up row
// ranges inside each matrix.
using MoeWeights = ops::MoeNvfp4Weights;

struct AttentionProjectionLeaf {
    static void project(const Tensor& hidden, const AttentionProjectionWeights& weights,
                        Tensor& query, Tensor& output_gate, Tensor& key, Tensor& value,
                        Tensor& index_query, Tensor& index_key, WorkspaceArena& workspace,
                        cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

struct GdnProjectionLeaf {
    static void project(const Tensor& hidden, const GdnProjectionWeights& weights, Tensor& query,
                        Tensor& key, Tensor& value, Tensor& z, Tensor& g, Tensor& beta,
                        WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

struct PostMixerLeaf {
    static void mix(const Tensor& hidden, const MoeWeights& weights, Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
