#pragma once

#include "core/tensor.h"

#include <array>
#include <cstddef>

namespace ninfer::targets::qwen4_exp {

// The four-stream gated residual weights of one HC site: the low-rank down projection with its
// S injection rows, the up projection, and the offset RMSNorm gain over the flattened streams.
struct HcWeights {
    Weight down_injection;
    Weight up;
    Tensor norm;
};

// The PLE weights: the key and value projections over the n-gram embedding, the three offset
// RMSNorm gains, and the tap-major dilated convolution kernel.
struct PleWeights {
    Weight key;
    Weight value;
    Tensor norm_key;
    Tensor norm_query;
    Tensor norm_conv;
    Tensor kernel;
};

// One text layer: the two HC sites, the mixer payload for whichever mixer the layer owns, and
// the post-mixer payload.
template <class AttentionPayload, class GdnPayload, class MoePayload>
struct TextLayerWeights {
    HcWeights attention_hc;
    HcWeights moe_hc;
    AttentionPayload attention_projection;
    GdnPayload gdn_projection;
    MoePayload post_mixer;
};

// The text model view. Every entry is a geometric view over load-time device residency; nothing
// here is decoded or repacked.
template <class AttentionPayload, class GdnPayload, class MoePayload, std::size_t Layers>
struct TextModelView {
    using Layer = TextLayerWeights<AttentionPayload, GdnPayload, MoePayload>;

    Weight embedding;
    Weight output_head;
    HcWeights final_hc;
    PleWeights ple;
    std::array<Layer, Layers> layers;
    std::size_t loaded_layers = 0;
    int tp = 1;

    [[nodiscard]] const TextModelView& view(int rank) const { return rank == 0 ? *this : *this; }
};

} // namespace ninfer::targets::qwen4_exp
