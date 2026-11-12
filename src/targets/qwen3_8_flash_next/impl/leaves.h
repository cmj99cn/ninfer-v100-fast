#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// The three closed execution leaves of this package.
//
// The leaves own the projections of the two mixers and the mixture of experts; the four-stream
// residual around them belongs to the family runtime. Every entry point below reports that it
// has no implementation yet: the leaves are carried by the M2 text leaf milestone, and a
// consumer must not mistake the declared shape for a working one.

// The QSA attention projection of one layer.
//
// The six projections are separate weights: the loader slices the artifact's packed parents
// into them, so the leaf never carries a row offset inside a parent. The four norms are the
// per-head-dimension gains of the offset RMSNorm.
struct AttentionProjectionWeights {
    Weight query;
    Weight output_gate;
    Weight key;
    Weight value;
    Weight index_query;
    Weight index_key;
    Tensor query_norm;
    Tensor key_norm;
    Tensor index_query_norm;
    Tensor index_key_norm;
};

// The GDN projection and control of one layer.
//
// The four input projections are separate weights, cut out of the artifact's packed parent.
// The control projections produce the two decay parameters, and the two vectors are the
// per-head-dimension gains of the gated delta net's state.
struct GdnProjectionWeights {
    Weight query;
    Weight key;
    Weight value;
    Weight z;
    Weight a_projection;
    Weight b_projection;
    Tensor a_log;
    Tensor dt_bias;
    Tensor convolution;
    Tensor norm;
    Weight output;
};

// The mixture of experts of one layer.
//
// The routing control holds the routed logits, the shared logit, the shared expert's gate and
// up projections and the shared expert's down projection. The routed bank holds the two
// batched NVFP4 libraries; the gate and the up projection are the two halves of one matrix of
// every expert, and the split is the expert's intermediate width.
struct MoeWeights {
    Weight router;
    Weight shared_score;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_down;
    Weight routed_gate_up;
    Weight routed_down;
};

// The indexed-attention projection leaf: the six projections of one QSA layer and the
// per-head normalization and rotation of their results.
//
// The entry point projects the hidden state of the layer and writes the prepared query,
// output gate, key, value, index query and index key. `positions` holds the three rotary
// coordinates of every column.
struct AttentionProjectionLeaf {
    static void project(const Tensor& hidden, const AttentionProjectionWeights& weights,
                        const Tensor& positions, Tensor& query, Tensor& output_gate, Tensor& key,
                        Tensor& value, Tensor& index_query, Tensor& index_key,
                        WorkspaceArena& workspace, cudaStream_t stream);

    // The workspace the leaf needs for `tokens` columns.
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

// The GDN projection and control leaf: the four input projections of one linear-attention
// layer and the two control projections beside them.
struct GdnProjectionLeaf {
    static void project(const Tensor& hidden, const GdnProjectionWeights& weights, Tensor& query,
                        Tensor& key, Tensor& value, Tensor& z, Tensor& a, Tensor& b,
                        WorkspaceArena& workspace, cudaStream_t stream);

    // The workspace the leaf needs for `tokens` columns.
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

// The post-mixer leaf: the mixture of experts of one layer, the shared expert and the routed
// experts.
struct PostMixerLeaf {
    static void mix(const Tensor& hidden, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream);

    // The workspace the leaf needs for `tokens` columns.
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
