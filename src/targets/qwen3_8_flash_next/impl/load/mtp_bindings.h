#pragma once

#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail::mtp {

struct MtpPlan {
    std::vector<BoundObject> objects;
    BoundParameter token_embedding;
    BoundParameter output_head;
    BoundParameter proposal_head;
    BoundParameter fc_embedding;
    BoundParameter fc_hidden;
    BoundParameter pre_fc_embedding_norm;
    BoundParameter pre_fc_hidden_norm;
    HcPlan final_hc;
    LayerPlan layer;
    std::size_t binding_count = 0;

    [[nodiscard]] std::vector<const BoundParameter*> parameters() const;
};

[[nodiscard]] MtpPlan bind_mtp_artifact(artifact::Binder& binder, const BindingPlan& text);

} // namespace ninfer::targets::qwen3_8_flash_next::detail::mtp
