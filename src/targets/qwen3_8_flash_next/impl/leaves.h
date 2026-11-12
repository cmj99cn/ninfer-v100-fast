#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

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

struct MoeWeights {
    Weight router;
    Weight shared_score;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_down;
    ExpertLibrary routed_gate;
    ExpertLibrary routed_up;
    ExpertLibrary routed_down;
};

struct AttentionProjectionLeaf {
    static void project(const Tensor& hidden, const AttentionProjectionWeights& weights,
                        const Tensor& positions, Tensor& query, Tensor& output_gate, Tensor& key,
                        Tensor& value, Tensor& index_query, Tensor& index_key,
                        WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

struct GdnProjectionLeaf {
    static void project(const Tensor& hidden, const GdnProjectionWeights& weights, Tensor& query,
                        Tensor& key, Tensor& value, Tensor& z, Tensor& a, Tensor& b,
                        WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

struct PostMixerLeaf {
    static void mix(const Tensor& hidden, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
