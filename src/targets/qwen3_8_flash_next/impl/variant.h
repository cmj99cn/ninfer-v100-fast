#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/leaves.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct Variant {
    using WeightsProfile        = detail::WeightsProfile;
    using TextConfig            = detail::TextConfig;
    using ModelView             = qwen4_exp::TextModelView<AttentionProjectionWeights,
                                                            GdnProjectionWeights, MoeWeights,
                                                            TextConfig::layers>;
    using GraphExecutionProfile = qwen4_exp::GraphExecutionProfile;

    static constexpr float attention_scale             = 1.0F;
    static constexpr float gdn_scale                   = 0.08838834764831845F;
    static constexpr std::uint32_t prefill_chunk_alignment = 64;
    static constexpr std::uint32_t maximum_context     = TextConfig::max_position_embeddings;

    static constexpr std::uint32_t kv_page_tokens = 64;
    static constexpr std::size_t bytes_per_main_page_group =
        static_cast<std::size_t>(kv_page_tokens) * TextConfig::kv_width * 2 +
        static_cast<std::size_t>(TextConfig::index_head_dim) * 16 *
            TextConfig::index_kv_heads * 2;
    static constexpr std::size_t minimum_device_reservation_bytes =
        bytes_per_main_page_group;
    static constexpr std::size_t bytes_per_additional_main_page_group =
        bytes_per_main_page_group;

    static void attention_projection(const Tensor& hidden, const AttentionProjectionWeights& weights,
                                     Tensor& query, Tensor& output_gate, Tensor& key, Tensor& value,
                                     Tensor& index_query, Tensor& index_key,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
        AttentionProjectionLeaf::project(hidden, weights, query, output_gate, key, value, index_query,
                                         index_key, workspace, stream);
    }

    static void gdn_projection_control(const Tensor& hidden, const GdnProjectionWeights& weights,
                                       Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                       Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                       cudaStream_t stream) {
        GdnProjectionLeaf::project(hidden, weights, query, key, value, z, g, beta, workspace,
                                   stream);
    }

    static void post_mixer_routed(const Tensor& hidden, const MoeWeights& weights,
                                  const ops::MoeNvfp4RouteView& route, Tensor& destination,
                                  cudaStream_t stream) {
        PostMixerLeaf::routed(hidden, weights, route, destination, stream);
    }

    static void post_mixer_shared(const Tensor& hidden, const MoeWeights& weights,
                                  const ops::MoeNvfp4RouteView& route, Tensor& destination,
                                  cudaStream_t stream) {
        PostMixerLeaf::shared(hidden, weights, route, destination, stream);
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
