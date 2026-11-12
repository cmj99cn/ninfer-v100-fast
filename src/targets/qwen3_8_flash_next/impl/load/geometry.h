#pragma once

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct RoleGeometry {
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    std::uint32_t rank = 2;
    std::array<std::uint64_t, 3> shape{};
    std::uint64_t row_begin = 0;
    std::uint64_t row_count = 0;
    std::uint64_t column_begin = 0;
    std::uint64_t column_count = 0;
    std::uint64_t matrix_count = 1;
    std::uint64_t part_count = 1;

    [[nodiscard]] std::span<const std::uint64_t> object_shape() const {
        return {shape.data(), rank};
    }
};

[[nodiscard]] RoleGeometry role_geometry(roles::Role role) noexcept;

} // namespace ninfer::targets::qwen3_8_flash_next::detail
