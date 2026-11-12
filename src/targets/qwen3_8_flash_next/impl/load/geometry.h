#pragma once

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// The storage geometry one artifact parameter occupies.
//
// A parameter of this artifact is either a whole object or a row range of a packed parent. The
// geometry names the parent's own format, layout and shape, the parameter's rectangle inside
// that parent -- the output-feature rows it starts at, how many it spans and the input-feature
// columns it covers -- and the number of parts the parameter occupies.
//
// Every layer of a role family shares its geometry: the artifact gives each layer its own
// object, but the objects of one family have the same format, layout, shape and element range.
struct RoleGeometry {
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    // The rank of the parent object's shape; the unused dimensions of `shape` are zero.
    std::uint32_t rank = 2;
    std::array<std::uint64_t, 3> shape{};
    std::uint64_t row_begin    = 0;
    std::uint64_t row_count    = 0;
    std::uint64_t column_begin = 0;
    std::uint64_t column_count = 0;
    // The number of matrices of the parent object. One for every parent but the routed expert
    // banks, which stack one matrix per expert.
    std::uint64_t matrix_count = 1;
    // The number of parts the parameter occupies: one for a parameter of a single parent, one
    // per expert for a routed expert bank and one per shard for the n-gram table.
    std::uint64_t part_count = 1;

    [[nodiscard]] std::span<const std::uint64_t> object_shape() const {
        return {shape.data(), rank};
    }
};

// The geometry of one role.
[[nodiscard]] RoleGeometry role_geometry(roles::Role role) noexcept;

} // namespace ninfer::targets::qwen3_8_flash_next::detail
