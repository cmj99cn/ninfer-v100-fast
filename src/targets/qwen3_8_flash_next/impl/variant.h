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
    static constexpr float gdn_scale                   = 1.0F;
    static constexpr std::uint32_t prefill_chunk_alignment = 64;
    static constexpr std::uint32_t maximum_context     = TextConfig::max_position_embeddings;

    static constexpr std::uint32_t kv_page_tokens = 64;
    static constexpr std::uint32_t kv_group_size = 64;
    static constexpr std::size_t bytes_per_main_page_group =
        kv_page_tokens * (static_cast<std::size_t>(TextConfig::kv_width) * 2 +
                          static_cast<std::size_t>(TextConfig::kv_width) / kv_group_size * 2);
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
                                       Tensor& qkv, Tensor& z, Tensor& g, Tensor& beta,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
        GdnProjectionLeaf::project(hidden, weights, qkv, z, g, beta, workspace, stream);
    }

    static void post_mixer(const Tensor& hidden, const MoeWeights& weights, Tensor& destination,
                           WorkspaceArena& workspace, cudaStream_t stream) {
        PostMixerLeaf::mix(hidden, weights, destination, workspace, stream);
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
