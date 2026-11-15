#pragma once

#include "core/tensor.h"
#include "ninfer/ops/ple.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen4_exp {

// The four-stream gated residual weights of one HC site: the low-rank down projection with its
// S injection rows, the up projection, and the offset RMSNorm gain over the flattened streams.
struct HcWeights {
    Weight down_injection;
    Weight up;
    Tensor norm;
};

struct PleRowSource {
    void* context = nullptr;
    void (*fetch)(void* context, const ops::PleRowAddress* addresses, std::size_t count,
                  std::byte* destination) = nullptr;
    const float* divisors = nullptr;
};

struct PleWeights {
    Weight key;
    Weight value;
    Tensor norm_key;
    Tensor norm_query;
    Tensor norm_conv;
    Tensor kernel;
    ops::PleHashControls controls{};
    PleRowSource rows;
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

// The live residency accounts of one rank. The package owns the storage behind this pointer; the
// pointer itself is part of the model view so the family's memory summary can report the device
// weights, the expert cache, and the host staging the package actually allocated.
struct TextResidentStats {
    std::uint64_t weights_capacity_bytes = 0;
    std::uint64_t weights_used_bytes     = 0;
    std::uint64_t expert_cache_bytes     = 0;
    std::uint64_t expert_scratch_bytes   = 0;
    std::uint64_t expert_cache_pairs     = 0;
    std::uint64_t expert_cache_hits      = 0;
    std::uint64_t expert_cache_misses    = 0;
    std::uint64_t expert_cache_miss_bytes = 0;
    std::uint64_t expert_host_bytes      = 0;
};

// The text model view. Every entry is a geometric view over load-time device residency; nothing
// here is decoded or repacked.
template <class AttentionPayload, class GdnPayload, class MoePayload, std::size_t Layers>
struct TextModelView {
    using Layer = TextLayerWeights<AttentionPayload, GdnPayload, MoePayload>;

    Weight embedding;
    Weight output_head;
    HcWeights final_hc;
    // The norm the last loaded layer's combine uses: the next layer's attention HC norm for a
    // truncated prefix, `final_hc.norm` for the complete stack.
    Tensor tail_norm;
    PleWeights ple;
    std::array<Layer, Layers> layers;
    TextResidentStats* resident = nullptr;
    std::size_t loaded_layers = 0;
    int tp = 1;

    [[nodiscard]] const TextModelView& view(int rank) const { return rank == 0 ? *this : *this; }
};

} // namespace ninfer::targets::qwen4_exp
