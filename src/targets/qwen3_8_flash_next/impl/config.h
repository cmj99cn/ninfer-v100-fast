#pragma once

#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct TextConfig {
    static constexpr int hidden                 = 2560;
    static constexpr int layers                 = 48;
    static constexpr int vocab                  = 248320;
    static constexpr int max_position_embeddings = 262144;

    static constexpr int hc_streams    = 4;
    static constexpr int hc_lowrank    = 320;
    static constexpr int residual_width = hidden * hc_streams;

    static constexpr int query_heads = 24;
    static constexpr int kv_heads    = 2;
    static constexpr int head_dim    = 256;
    static constexpr int query_width = query_heads * head_dim;
    static constexpr int kv_width    = kv_heads * head_dim;

    static constexpr int index_heads       = 4;
    static constexpr int index_kv_heads    = 1;
    static constexpr int index_head_dim    = 128;
    static constexpr int index_query_width = index_heads * index_head_dim;
    static constexpr int index_key_width   = index_kv_heads * index_head_dim;
    static constexpr int index_compress    = 4;
    static constexpr int index_budget      = 2048;
    static constexpr int index_block_budget = index_budget / index_compress;

    static constexpr int gdn_key_heads    = 16;
    static constexpr int gdn_value_heads  = 48;
    static constexpr int gdn_head_dim     = 128;
    static constexpr int gdn_key_width    = gdn_key_heads * gdn_head_dim;
    static constexpr int gdn_value_width  = gdn_value_heads * gdn_head_dim;
    static constexpr int gdn_conv_kernel  = 4;
    static constexpr int gdn_conv_channels = 2 * gdn_key_width + gdn_value_width;

    static constexpr int experts           = 512;
    static constexpr int experts_per_token = 10;
    static constexpr int expert_intermediate = 640;
    static constexpr int shared_intermediate = 640;

    static constexpr int ple_layer            = 1;
    static constexpr int ple_embed_dim      = hidden;
    static constexpr int ple_ngram          = 3;
    static constexpr int ple_heads_per_ngram = 8;
    static constexpr int ple_split_parts    = 128;
    static constexpr int ple_heads          = (ple_ngram - 1) * ple_heads_per_ngram;
    static constexpr int ple_head_width     = ple_embed_dim / ple_heads;
    static constexpr int ple_conv_kernel    = 4;
    static constexpr int ple_conv_dilation  = 3;
    static constexpr int ple_shard_rows     = 2500012;
    static constexpr int ple_shards         = ple_split_parts;

    static constexpr double rope_theta    = 1.0e7;
    static constexpr float partial_rotary = 0.25F;
    static constexpr std::array<int, 3> rope_sections{11, 11, 10};
    static constexpr int rotary_dim = 64;

    static constexpr float rms_epsilon = 1.0e-6F;

    static constexpr int eos_token_id          = 248044;
    static constexpr int image_token_id        = 248056;
    static constexpr int video_token_id        = 248057;
    static constexpr int vision_start_token_id = 248053;
    static constexpr int vision_end_token_id   = 248054;

    [[nodiscard]] static constexpr bool is_qsa_layer(int layer) { return layer % 4 == 3; }
    [[nodiscard]] static constexpr bool is_gdn_layer(int layer) { return !is_qsa_layer(layer); }

    [[nodiscard]] static constexpr int qsa_layers() {
        int count = 0;
        for (int layer = 0; layer < layers; ++layer) {
            if (is_qsa_layer(layer)) { ++count; }
        }
        return count;
    }

    [[nodiscard]] static constexpr int gdn_layers() { return layers - qsa_layers(); }
};

static_assert(TextConfig::experts == 512);
static_assert(TextConfig::residual_width == 10240);
static_assert(TextConfig::query_width == 6144);
static_assert(TextConfig::kv_width == 512);
static_assert(TextConfig::index_query_width == 512);
static_assert(TextConfig::index_key_width == 128);
static_assert(TextConfig::index_block_budget == 512);
static_assert(TextConfig::gdn_key_width == 2048);
static_assert(TextConfig::gdn_value_width == 6144);
static_assert(TextConfig::gdn_conv_channels == 10240);
static_assert(TextConfig::ple_head_width == 160);
static_assert(TextConfig::qsa_layers() == 12);
static_assert(TextConfig::gdn_layers() == 36);
static_assert(TextConfig::head_dim * TextConfig::partial_rotary ==
              static_cast<float>(TextConfig::rotary_dim));
static_assert(TextConfig::rope_sections[0] + TextConfig::rope_sections[1] +
                  TextConfig::rope_sections[2] ==
              TextConfig::rotary_dim / 2);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
