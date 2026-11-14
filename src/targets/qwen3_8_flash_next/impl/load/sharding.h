#pragma once

#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <cstdint>
#include <vector>

// The shard map of the text roles: the axis a role is split along and the range every rank owns.
//
// `ShardAxis::Head` splits a role by its head width, `Intermediate` by the MoE intermediate width
// and `Vocab` by the vocabulary. `Replicated` names a role whose whole object lives on every rank,
// and `Host` names a role that is not a device shard at all and is read from the pinned host row
// source. An empty `ranks` is the binder convention for a full copy on device 0.
//
// A `Shard` is a half-open range on the stored axis `SplitAxis` selects: the output-feature axis of
// a column-parallel role, the input-feature axis of a row-parallel one. Ranges are positions in the
// parent object, so a role's own span is `begin` to `begin + extent` and the ranks partition it.
namespace ninfer::targets::qwen3_8_flash_next::detail::sharding {

enum class ShardAxis { Replicated, Host, Vocab, Head, Intermediate };

enum class SplitAxis { Output, Input };

struct Shard {
    std::int64_t begin;
    std::int64_t end;
};

struct ShardMapping {
    ShardAxis axis;
    SplitAxis split;
    std::vector<Shard> ranks;
};

// The map of one role at one tensor-parallel width. A span the axis cannot split evenly, or a
// boundary off the role's head, group or row-band boundary, is rejected with the role's own name.
[[nodiscard]] ShardMapping shard_mapping(roles::Role role, int tp);

} // namespace ninfer::targets::qwen3_8_flash_next::detail::sharding
