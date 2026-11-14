#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_nvfp4.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct AttentionProjectionWeights {
    Weight query_gate;
    Weight key_value;
    Tensor query_norm;
    Tensor key_norm;
    Tensor index_query_norm;
    Tensor index_key_norm;
    Weight output;
};

struct GdnProjectionWeights {
    Weight qkvz;
    Weight control;
    Tensor a_log;
    Tensor dt_bias;
    Tensor convolution;
    Tensor norm;
    Weight output;
};

class ExpertRuntime;

struct MoeWeights : ops::MoeNvfp4Weights {
    ExpertRuntime* cache = nullptr;
    std::uint32_t layer  = 0;
};

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
    static void routed(const Tensor& hidden, const MoeWeights& weights,
                       const ops::MoeNvfp4RouteView& route, Tensor& destination,
                       cudaStream_t stream);
    static void shared(const Tensor& hidden, const MoeWeights& weights,
                       const ops::MoeNvfp4RouteView& route, Tensor& destination,
                       cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::int32_t tokens);
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
