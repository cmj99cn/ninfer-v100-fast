// Host-only: no device. The per-rank shard byte plan of the text roles at tp 2.
//
// Every expected byte mapping is computed here from the storage-layout formulas of
// docs/maintainer/storage-layouts.md (sections 3.7, 4 and 11) over the artifact's own stored
// bytes, read through the reader. The package's own slice arithmetic is only the thing under
// test, so a geometry mistake shows up as a byte mismatch instead of cancelling out.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen4_exp/frontend.h>
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/geometry.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include "targets/qwen3_8_flash_next/impl/load/sharding.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace artifact = ninfer::artifact;

namespace {

using ninfer::artifact::Binder;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::PlaneCopy;
using ninfer::artifact::Reader;
using ninfer::artifact::SliceRange;
using ninfer::artifact::StorageLayout;

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;
namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
namespace shard = ninfer::targets::qwen3_8_flash_next::detail::sharding;

using ninfer::targets::qwen3_8_flash_next::detail::RoleGeometry;
using ninfer::targets::qwen3_8_flash_next::detail::role_geometry;

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

constexpr std::uint64_t kAlign      = 256;
constexpr std::uint64_t kRowTile    = 128;
constexpr std::uint64_t kTileColumn = 64;
constexpr std::uint64_t kTileBytes  = 512;
constexpr std::uint64_t kGroup      = 32;   // W8G32 group width
constexpr std::uint64_t kKAlign     = 128;  // row-split K alignment

std::uint64_t align_up(std::uint64_t value, std::uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

// storage-layouts.md section 4: one scale word's position inside one matrix's swizzled plane.
std::uint64_t swizzle_word(std::uint64_t row, std::uint64_t group, std::uint64_t columns) {
    const std::uint64_t tiles = columns / kTileColumn;
    return (row / kRowTile * tiles + group / 4) * kTileBytes + (row % 32) * 16 +
           ((row % kRowTile) / 32) * 4 + group % 4;
}

using Copies = std::vector<PlaneCopy>;

Copies ordered(Copies copies) {
    std::sort(copies.begin(), copies.end(), [](const PlaneCopy& a, const PlaneCopy& b) {
        return std::tie(a.source_offset, a.dest_offset, a.bytes) <
               std::tie(b.source_offset, b.dest_offset, b.bytes);
    });
    return copies;
}

bool same(const Copies& left, const Copies& right) {
    if (left.size() != right.size()) { return false; }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].source_offset != right[index].source_offset ||
            left[index].dest_offset != right[index].dest_offset ||
            left[index].bytes != right[index].bytes) {
            return false;
        }
    }
    return true;
}

Copies coalesced(Copies copies) {
    copies = ordered(copies);
    Copies out;
    for (const auto& copy : copies) {
        if (!out.empty()) {
            const auto& last = out.back();
            if (last.source_offset + last.bytes == copy.source_offset &&
                last.dest_offset + last.bytes == copy.dest_offset) {
                out.back().bytes += copy.bytes;
                continue;
            }
        }
        out.push_back(copy);
    }
    return out;
}

std::uint64_t total(const std::vector<SliceRange>& ranges) {
    return std::accumulate(ranges.begin(), ranges.end(), std::uint64_t{0},
                           [](std::uint64_t value, const SliceRange& range) {
                               return value + range.count;
                           });
}

std::uint64_t word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
        return 4;
    default:
        throw std::runtime_error("the contiguous oracle covers BF16 and FP32 only");
    }
}

std::vector<std::uint64_t> narrowed_shape(const RoleGeometry& geometry, shard::SplitAxis split,
                                          std::uint64_t selected) {
    std::vector<std::uint64_t> shape(geometry.object_shape().begin(), geometry.object_shape().end());
    if (split == shard::SplitAxis::Input) { shape[1] = selected; } else { shape[0] = selected; }
    return shape;
}

// The byte mapping the layout formulas give for one object's shard on one rank.
Copies expected_copies(const RoleGeometry& geometry, shard::SplitAxis split,
                       const std::vector<SliceRange>& ranges) {
    const auto shape = std::vector<std::uint64_t>(geometry.object_shape().begin(),
                                                  geometry.object_shape().end());
    Copies out;
    if (ranges.empty()) {
        out.push_back({0, 0, artifact::tensor_encoded_size(geometry.layout, geometry.format, shape)});
        return out;
    }
    const std::uint64_t selected = total(ranges);

    // A shard that takes the whole axis is the whole object: a full copy has no internal structure
    // to re-derive, whatever the layout.
    const auto extent = split == shard::SplitAxis::Input ? shape[1] : shape[0];
    if (ranges.size() == 1 && ranges.front().begin == 0 && ranges.front().count == extent) {
        out.push_back({0, 0, artifact::tensor_encoded_size(geometry.layout, geometry.format, shape)});
        return out;
    }

    if (geometry.layout == StorageLayout::ContiguousLeV1) {
        const std::uint64_t word = word_bytes(geometry.format);
        if (split == shard::SplitAxis::Input) {
            for (std::uint64_t row = 0; row < shape[0]; ++row) {
                std::uint64_t column = 0;
                for (const auto& range : ranges) {
                    out.push_back({(row * shape[1] + range.begin) * word,
                                   (row * selected + column) * word, range.count * word});
                    column += range.count;
                }
            }
            return out;
        }
        std::uint64_t dest_row = 0;
        for (const auto& range : ranges) {
            out.push_back({range.begin * shape[1] * word, dest_row * shape[1] * word,
                           range.count * shape[1] * word});
            dest_row += range.count;
        }
        return out;
    }

    if (geometry.layout == StorageLayout::RowSplitK128V1) {
        const auto groups_per_row = align_up(shape[1], kKAlign) / kGroup;
        const auto parent_scale   = align_up(shape[0] * groups_per_row * kGroup, kAlign);
        if (split == shard::SplitAxis::Input) {
            // storage-layouts.md 3.8: a K-divided shard is a standalone tensor whose own K_pad and
            // groups_per_row are recomputed from the selected columns.
            const auto shard_groups = align_up(selected, kKAlign) / kGroup;
            const auto shard_scale  = align_up(shape[0] * shard_groups * kGroup, kAlign);
            const auto groups       = ranges.front().count / kGroup;
            for (std::uint64_t row = 0; row < shape[0]; ++row) {
                out.push_back({row * groups_per_row * kGroup + ranges.front().begin,
                               row * shard_groups * kGroup, ranges.front().count});
                out.push_back({parent_scale + row * groups_per_row * 2 + ranges.front().begin / kGroup * 2,
                               shard_scale + row * shard_groups * 2, groups * 2});
            }
            return out;
        }
        const auto shard_scale = align_up(selected * groups_per_row * kGroup, kAlign);
        std::uint64_t dest_row = 0;
        for (const auto& range : ranges) {
            for (std::uint64_t index = 0; index < range.count; ++index) {
                out.push_back({(range.begin + index) * groups_per_row * kGroup,
                               (dest_row + index) * groups_per_row * kGroup, groups_per_row * kGroup});
                out.push_back({parent_scale + (range.begin + index) * groups_per_row * 2,
                               shard_scale + (dest_row + index) * groups_per_row * 2, groups_per_row * 2});
            }
            dest_row += range.count;
        }
        return out;
    }

    if (geometry.layout == StorageLayout::BatchedBlockScaleK16M128x4V1) {
        const auto matrix_code  = shape[1] * shape[2] / 2;
        const auto matrix_scale = shape[1] * shape[2] / 16;
        const auto parent_scale   = align_up(shape[0] * matrix_code, kAlign);
        const auto parent_divisor = parent_scale + shape[0] * matrix_scale;
        const auto shard_scale    = align_up(selected * matrix_code, kAlign);
        const auto shard_divisor  = shard_scale + selected * matrix_scale;
        std::uint64_t index = 0;
        for (const auto& range : ranges) {
            for (std::uint64_t matrix = range.begin; matrix < range.begin + range.count; ++matrix) {
                out.push_back({matrix * matrix_code, index * matrix_code, matrix_code});
                out.push_back({parent_scale + matrix * matrix_scale, shard_scale + index * matrix_scale,
                               matrix_scale});
                out.push_back({parent_divisor + matrix * 4, shard_divisor + index * 4, 4});
                ++index;
            }
        }
        return out;
    }

    throw std::runtime_error("the oracle does not cover this layout");
}

std::vector<std::byte> read_object(const Reader& reader, const std::string& name) {
    const auto* descriptor = reader.find(name);
    if (descriptor == nullptr) { throw std::runtime_error("the artifact has no object " + name); }
    if (std::holds_alternative<artifact::ResourceDescriptor>(*descriptor)) {
        throw std::runtime_error("the object " + name + " is a resource, not a tensor");
    }
    std::vector<std::byte> out(artifact::object_bytes(*descriptor));
    const auto offset = reader.payload_offset() + artifact::object_offset(*descriptor);
    if (reader.read_direct(offset, out) != out.size()) {
        throw std::runtime_error("short artifact read for " + name);
    }
    return out;
}

std::vector<std::byte> assemble(const std::vector<std::byte>& parent, const Copies& copies) {
    std::uint64_t size = 0;
    for (const auto& copy : copies) {
        size = std::max(size, copy.dest_offset + copy.bytes);
    }
    std::vector<std::byte> out(size, std::byte{0});
    for (const auto& copy : copies) {
        if (copy.source_offset + copy.bytes > parent.size()) {
            throw std::runtime_error("a copy reads past the parent object: " + std::to_string(copy.source_offset) +
                                     "+" + std::to_string(copy.bytes) + " > " + std::to_string(parent.size()));
        }
        for (std::uint64_t index = 0; index < copy.bytes; ++index) {
            out[static_cast<std::size_t>(copy.dest_offset + index)] =
                parent[static_cast<std::size_t>(copy.source_offset + index)];
        }
    }
    return out;
}

const shard::ShardMapping& mapping_of(const flash::ShardPlanByName& shards, std::string_view role) {
    const auto found = shards.find(role);
    if (found == shards.end()) { throw std::runtime_error("the shard map has no role " + std::string(role)); }
    return found->second;
}

struct BoundObject {
    std::string name;
    RoleGeometry geometry;
    std::vector<roles::Role> owners;
    std::vector<shard::SplitAxis> axes;
};

std::map<std::string, BoundObject> bound_objects(const flash::BindingPlan& plan,
                                                  const flash::ShardPlanByName& shards) {
    std::map<std::string, BoundObject> by_object;
    for (const auto* parameter : plan.parameters()) {
        for (const auto& part : parameter->parts) {
            const auto& object = plan.objects.at(part.object);
            auto& entry = by_object[object.name];
            if (entry.owners.empty()) { entry.geometry = role_geometry(parameter->role); }
            if (std::find(entry.owners.begin(), entry.owners.end(), parameter->role) == entry.owners.end()) {
                entry.owners.push_back(parameter->role);
                entry.axes.push_back(mapping_of(shards, roles::role_name(parameter->role)).split);
            }
            entry.name = object.name;
        }
    }
    return by_object;
}

// What one rank owns of one object: every role bound to that object contributes its own range on
// the object's split axis, or its whole rectangle when the role is replicated.
std::vector<SliceRange> rank_ranges(const flash::ShardPlanByName& shards, const BoundObject& object,
                                    int rank) {
    std::vector<SliceRange> ranges;
    for (const auto role : object.owners) {
        const auto& mapping = mapping_of(shards, roles::role_name(role));
        const auto geometry = role_geometry(role);
        if (mapping.ranks.empty()) {
            const auto begin = mapping.split == shard::SplitAxis::Input ? geometry.column_begin
                                                                        : geometry.row_begin;
            const auto count = mapping.split == shard::SplitAxis::Input ? geometry.column_count
                                                                        : geometry.row_count;
            ranges.push_back({begin, count});
            continue;
        }
        const auto& part = mapping.ranks[static_cast<std::size_t>(rank)];
        ranges.push_back({static_cast<std::uint64_t>(part.begin),
                          static_cast<std::uint64_t>(part.end - part.begin)});
    }
    std::sort(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin < b.begin;
    });
    // Roles that share one batched stack name the same matrix range: the object owns it once.
    ranges.erase(std::unique(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin == b.begin && a.count == b.count;
    }), ranges.end());
    return ranges;
}

void test_partition(const flash::ShardPlanByName& shards) {
    for (const auto role : roles::all_roles()) {
        const auto name = roles::role_name(role);
        const auto mapping = mapping_of(shards, name);
        if (mapping.ranks.empty()) { continue; }
        const auto geometry = role_geometry(role);
        const auto begin = mapping.split == shard::SplitAxis::Input ? geometry.column_begin
                     : (mapping.split == shard::SplitAxis::Batch ? 0 : geometry.row_begin);
        const auto extent = mapping.split == shard::SplitAxis::Input ? geometry.column_count
                     : (mapping.split == shard::SplitAxis::Batch ? geometry.object_shape()[0]
                                                                 : geometry.row_count);
        std::uint64_t cursor = begin;
        for (const auto& part : mapping.ranks) {
            require(part.begin == static_cast<std::int64_t>(cursor),
                   std::string(name) + " rank ranges do not partition the role rectangle in order");
            require(part.begin % 128 == 0, std::string(name) + " rank starts inside a 128-row band");
            require((part.end - part.begin) % 128 == 0,
                   std::string(name) + " rank covers part of a 128-row band");
            cursor = static_cast<std::uint64_t>(part.end);
        }
        require(cursor == begin + extent, std::string(name) + " ranks end at " +
                                      std::to_string(cursor) + "; the role rectangle ends at " +
                                      std::to_string(begin + extent));
    }
}

void test_expert_subsets(const flash::ShardPlanByName& shards) {
    std::size_t seen = 0;
    for (const auto role : roles::all_roles()) {
        const auto name = roles::role_name(role);
        const auto mapping = mapping_of(shards, name);
        if (mapping.axis != shard::ShardAxis::Expert) { continue; }
        ++seen;
        require(mapping.split == shard::SplitAxis::Batch, std::string(name) + " is not split on the batch axis");
        require(mapping.ranks.size() == 2, std::string(name) + " does not name two ranks");
        for (std::size_t rank = 0; rank < mapping.ranks.size(); ++rank) {
            const auto& part = mapping.ranks[rank];
            require(part.end - part.begin == 256, std::string(name) + " rank " + std::to_string(rank) +
                                                      " owns " + std::to_string(part.end - part.begin) +
                                                      " matrices; the expert set gives 256");
            require(part.begin == static_cast<std::int64_t>(rank) * 256,
                   std::string(name) + " rank " + std::to_string(rank) + " owns matrices starting at " +
                       std::to_string(part.begin) + "; the expert set gives " + std::to_string(rank * 256));
        }
    }
    require(seen == 3, "the shard map names " + std::to_string(seen) + " expert roles; the MoE has three");
}

void test_replicated(const flash::ShardPlanByName& shards,
                     const std::map<std::string, BoundObject>& by_object,
                     const std::vector<flash::ObjectShardPlan>& byte_plan) {
    std::size_t checked = 0;
    for (const auto& entry : byte_plan) {
        const auto found = by_object.find(entry.object);
        if (found == by_object.end()) { continue; }
        const auto& object = found->second;
        for (const auto role : object.owners) {
            const auto name = roles::role_name(role);
            const auto mapping = mapping_of(shards, name);
            if (mapping.axis != shard::ShardAxis::Replicated) { continue; }
            ++checked;
            const auto geometry = role_geometry(role);
            const auto begin = mapping.split == shard::SplitAxis::Input ? geometry.column_begin
                                                                       : geometry.row_begin;
            const auto count = mapping.split == shard::SplitAxis::Input ? geometry.column_count
                                                                        : geometry.row_count;
            for (int rank = 0; rank < 2; ++rank) {
                const auto& ranges = entry.ranges[static_cast<std::size_t>(rank)];
                const auto owns = std::find_if(ranges.begin(), ranges.end(),
                                               [&](const SliceRange& range) {
                                                   return range.begin == begin && range.count == count;
                                               });
                require(owns != ranges.end(), std::string(name) + " rank " + std::to_string(rank) +
                                                 " does not own its whole role rectangle in " +
                                                 entry.object);
                if (object.axes.size() == 1) {
                    require(same(entry.copies[static_cast<std::size_t>(rank)], entry.copies[0]),
                            std::string(name) + " rank " + std::to_string(rank) +
                                " maps to a different object offset");
                }
            }
        }
    }
    require(checked > 0, "no replicated role was checked");
}

void test_byte_plan(const flash::BindingPlan& plan, const flash::ShardPlanByName& shards,
                    const std::map<std::string, BoundObject>& by_object,
                    const std::vector<flash::ObjectShardPlan>& byte_plan) {
    require(byte_plan.size() == plan.objects.size(),
            "the byte plan covers " + std::to_string(byte_plan.size()) + " objects; the plan binds " +
                std::to_string(plan.objects.size()));

    std::size_t matched = 0;
    for (const auto& entry : byte_plan) {
        const auto found = by_object.find(entry.object);
        if (found == by_object.end()) { continue; }
        const auto& object = found->second;
        if (object.owners.empty()) {
            require(entry.copies[0].size() == 1 && entry.copies[0].front().source_offset == 0 &&
                    entry.copies[0].front().dest_offset == 0,
                    "the unbound object " + entry.object + " is not a whole-object copy");
            continue;
        }
        const auto split = object.axes.front();
        for (int rank = 0; rank < 2; ++rank) {
            const auto ranges   = rank_ranges(shards, object, rank);
            const auto expected = coalesced(expected_copies(object.geometry, split, ranges));
            const Copies actual = coalesced(entry.copies[static_cast<std::size_t>(rank)]);
            require(same(expected, actual),
                    "object " + entry.object + " rank " + std::to_string(rank) +
                        " does not match the layout formulas for split axis " +
                        std::to_string(static_cast<int>(split)));
            require(same(expected, actual), "the byte plan for " + entry.object + " rank " +
                                            std::to_string(rank) + " holds " +
                                            std::to_string(actual.size()) + " ranges; the layout formulas give " +
                                            std::to_string(expected.size()));
            const auto shape = narrowed_shape(object.geometry, split, total(ranges));
            require(entry.encoded_bytes[static_cast<std::size_t>(rank)] ==
                    artifact::tensor_encoded_size(object.geometry.layout, object.geometry.format, shape),
                    "the shard of " + entry.object + " rank " + std::to_string(rank) + " is " +
                        std::to_string(entry.encoded_bytes[static_cast<std::size_t>(rank)]) +
                        " bytes; the narrowed shape gives " +
                        std::to_string(artifact::tensor_encoded_size(object.geometry.layout,
                                                                    object.geometry.format, shape)));
            ++matched;
        }
    }
    std::cout << "flash-next shard byte plan: " << matched << " object ranks match the layout formulas\n";
}

const std::vector<std::pair<roles::Role, std::uint32_t>> kSampled = {
    {roles::Role::MoeRoutedGate, 0},
    {roles::Role::MoeRoutedUp, 0},
    {roles::Role::MoeRoutedDown, 0},
    {roles::Role::GdnQuery, 0},
    {roles::Role::QsaKey, 3},
    {roles::Role::TokenEmbedding, 0},
};

void test_real_bytes(const flash::BindingPlan& plan, const flash::ShardPlanByName& shards,
                     const Reader& reader,
                     const std::vector<flash::ObjectShardPlan>& byte_plan) {
    const auto by_object = bound_objects(plan, shards);
    const auto parameters = plan.parameters();

    for (const auto [role, layer] : kSampled) {
        const auto name = std::string(roles::parameter_name(role, layer));
        const auto found = std::find_if(parameters.begin(), parameters.end(),
                                        [&](const flash::BoundParameter* candidate) {
                                            return candidate->parameter == name;
                                        });
        if (found == parameters.end()) { throw std::runtime_error("no parameter " + name); }
        const auto& parameter = *found;
        const auto& object = plan.objects.at(parameter->parts.front().object);
        const auto entry_found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                              [&](const flash::ObjectShardPlan& candidate) {
                                                  return candidate.object == object.name;
                                              });
        if (entry_found == byte_plan.end()) { throw std::runtime_error("no byte plan for " + object.name); }
        const auto& entry = *entry_found;
        const auto info_found = by_object.find(object.name);
        if (info_found == by_object.end()) { throw std::runtime_error("no bound entry for object " + object.name); }
        const auto& info   = info_found->second;
        const auto split   = mapping_of(shards, roles::role_name(parameter->role)).split;
        const auto parent  = read_object(reader, object.name);
        for (int rank = 0; rank < 2; ++rank) {
            const auto ranges = rank_ranges(shards, info, rank);
            const auto expected = ordered(expected_copies(info.geometry, split, ranges));
            const Copies actual = ordered(entry.copies[static_cast<std::size_t>(rank)]);
            require(assemble(parent, actual) == assemble(parent, expected),
                    "the shard of " + object.name + " rank " + std::to_string(rank) +
                        " does not hold the bytes the layout formulas give");
        }
    }
}

void test_scale_plane(const flash::ShardPlanByName& shards, const Reader& reader,
                      const std::vector<flash::ObjectShardPlan>& byte_plan) {
    const auto found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                    [&](const flash::ObjectShardPlan& candidate) {
                                        return candidate.object == "weight/000000";
                                    });
    if (found == byte_plan.end()) { throw std::runtime_error("no byte plan for weight/000000"); }
    const auto& entry = *found;

    const std::vector<std::uint64_t> shape{512, 1280, 2560};
    const auto matrix_code  = shape[1] * shape[2] / 2;
    const auto matrix_scale = shape[1] * shape[2] / 16;
    const auto parent_scale = align_up(shape[0] * matrix_code, kAlign);
    const auto parent       = read_object(reader, "weight/000000");
    const auto& mapping     = mapping_of(shards, roles::role_name(roles::Role::MoeRoutedGate));

    for (int rank = 0; rank < 2; ++rank) {
        const auto shard_scale = align_up(256 * matrix_code, kAlign);
        require(entry.scale_plane_offset[static_cast<std::size_t>(rank)] == shard_scale,
                "the scale plane of weight/000000 rank " + std::to_string(rank) + " starts at " +
                    std::to_string(entry.scale_plane_offset[static_cast<std::size_t>(rank)]) +
                    "; the batched formula gives " + std::to_string(shard_scale));

        for (std::uint64_t index = 0; index < 256; index += 31) {
            const std::uint64_t matrix = static_cast<std::uint64_t>(mapping.ranks[rank].begin) + index;
            for (const std::uint64_t row : {0ULL, 128ULL, 639ULL}) {
                for (const std::uint64_t group : {0ULL, 3ULL, 159ULL}) {
                    const auto word = swizzle_word(row, group, shape[2]);
                    const auto source = parent_scale + matrix * matrix_scale + word;
                    const auto dest   = shard_scale + index * matrix_scale + word;
                    const auto carrier = std::find_if(entry.copies[static_cast<std::size_t>(rank)].begin(),
                                                      entry.copies[static_cast<std::size_t>(rank)].end(),
                                                      [&](const PlaneCopy& copy) {
                                                          return copy.source_offset <= source &&
                                                                 source < copy.source_offset + copy.bytes;
                                                      });
                    require(carrier != entry.copies[static_cast<std::size_t>(rank)].end(),
                            "the shard of weight/000000 rank " + std::to_string(rank) +
                                " has no copy carrying the parent scale word for matrix " +
                                std::to_string(matrix));
                    if (carrier == entry.copies[static_cast<std::size_t>(rank)].end()) { continue; }
                    require(carrier->dest_offset + (source - carrier->source_offset) == dest,
                            "the shard scale word for rank " + std::to_string(rank) + " index " +
                                std::to_string(index) + " row " + std::to_string(row) + " group " +
                                std::to_string(group) + " lands at " +
                                std::to_string(carrier->dest_offset + (source - carrier->source_offset)) +
                                ", not at the shard position " + std::to_string(dest) +
                                " for parent matrix " + std::to_string(matrix));
                }
            }
        }
    }
}

const void* entry_payload(const flash::Variant::ModelView& view, roles::Role role, std::size_t layer) {
    switch (role) {
    case roles::Role::MoeRoutedGate:
    case roles::Role::MoeRoutedUp:
        return view.layers[layer].post_mixer.routed_gate_up.payload;
    case roles::Role::MoeRoutedDown:
        return view.layers[layer].post_mixer.routed_down.payload;
    case roles::Role::GdnQuery:
        return view.layers[layer].gdn_projection.qkvz.payload;
    case roles::Role::TokenEmbedding:
        return view.embedding.payload;
    default:
        return nullptr;
    }
}

// The materialized per-rank views. Needs two devices; skipped when they cannot be taken.
int test_materialized_views(const char* path) {
    ninfer::ExecutionContext execution({0, 1});
    if (execution.tp != 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }

    Reader reader(path);
    Binder binder(reader, 2);
    const auto plan       = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 2);
    const auto shards     = flash::shard_plan_by_name(2);
    const auto byte_plan  = flash::shard_byte_plan(plan, shards, 2);
    (void)ninfer::targets::qwen4_exp::bind_frontend_resources(binder);
    const auto by_object  = bound_objects(plan, shards);
    const auto placements = flash::shard_placements(plan, shards, 2);
    const auto parameters = plan.parameters();

    for (const auto& object : plan.objects) {
        require(placements.count(object.name) != 0,
                "the tensor object " + object.name + " has no shard placement");
        binder.materialize_on_device(object.handle);
    }
    binder.validate_unconsumed_matching("text/");
    binder.validate_unconsumed_matching("resource/text/");
    for (const auto& object : plan.unclaimed.object_names) {
        binder.validate_only(object);
    }

    artifact::MaterializedArtifact materialized;
    try {
        materialized = artifact::materialize(reader, binder.finish(), execution);
    } catch (const std::exception& error) {
        std::cout << "skip: the two devices cannot be taken: " << error.what() << "\n";
        return 77;
    }
    const auto views = flash::shard_views(plan, materialized, shards, 2);

    for (const auto [role, layer] : kSampled) {
        const auto name = std::string(roles::parameter_name(role, layer));
        const auto found = std::find_if(parameters.begin(), parameters.end(),
                                        [&](const flash::BoundParameter* candidate) {
                                            return candidate->parameter == name;
                                        });
        if (found == parameters.end()) { throw std::runtime_error("no parameter " + name); }
        const auto& parameter = *found;
        const auto& object = plan.objects.at(parameter->parts.front().object);
        const auto entry_found = std::find_if(byte_plan.begin(), byte_plan.end(),
                                              [&](const flash::ObjectShardPlan& candidate) {
                                                  return candidate.object == object.name;
                                              });
        if (entry_found == byte_plan.end()) { throw std::runtime_error("no byte plan for " + object.name); }
        const auto& entry = *entry_found;
        const auto info_found = by_object.find(object.name);
        if (info_found == by_object.end()) { throw std::runtime_error("no bound entry for object " + object.name); }
        const auto& info   = info_found->second;
        const auto split   = mapping_of(shards, roles::role_name(parameter->role)).split;
        const auto parent  = read_object(reader, object.name);
        for (int rank = 0; rank < 2; ++rank) {
            const auto& view = views.views[static_cast<std::size_t>(rank)];
            const auto ranges = rank_ranges(shards, info, rank);
            const auto expected = assemble(parent, ordered(expected_copies(info.geometry, split, ranges)));
            const Copies actual = ordered(entry.copies[static_cast<std::size_t>(rank)]);
            const auto base = entry_payload(view, parameter->role, parameter->layer);
            require(base != nullptr, "rank " + std::to_string(rank) + " has no view for " + name);
            if (parameter->library.has_value()) {
                const auto& bank = parameter->role == roles::Role::MoeRoutedDown
                                       ? view.layers[parameter->layer].post_mixer.routed_down
                                       : view.layers[parameter->layer].post_mixer.routed_gate_up;
                require(bank.matrices == 256, name + " rank " + std::to_string(rank) + " view holds " +
                                                  std::to_string(bank.matrices) + " matrices");
                require(bank.rows == static_cast<std::int32_t>(info.geometry.object_shape()[1]),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(bank.rows) + " rows; the fused role rectangles together span " +
                            std::to_string(info.geometry.object_shape()[1]));
                require(bank.columns == static_cast<std::int32_t>(info.geometry.object_shape()[2]),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(bank.columns) + " columns; the role rectangle spans " +
                            std::to_string(info.geometry.object_shape()[2]));
                require(bank.codes == base, name + " rank " + std::to_string(rank) +
                                                " code plane is not the shard base");
                require(bank.scales == static_cast<const std::uint8_t*>(base) +
                        entry.scale_plane_offset[static_cast<std::size_t>(rank)],
                        name + " rank " + std::to_string(rank) + " scale plane is not the shard's plane");
                require(reinterpret_cast<const std::uint8_t*>(bank.divisors) ==
                        static_cast<const std::uint8_t*>(base) +
                        entry.divisor_plane_offset[static_cast<std::size_t>(rank)],
                        name + " rank " + std::to_string(rank) + " divisor plane is not the shard's plane");
            } else {
                const auto rows = static_cast<std::int32_t>(total(ranges));
                const auto weight = parameter->role == roles::Role::GdnQuery
                                        ? view.layers[parameter->layer].gdn_projection.qkvz
                                        : view.layers[parameter->layer].attention_projection.key_value;
                require(weight.n == rows, name + " rank " + std::to_string(rank) + " view spans " +
                                              std::to_string(weight.n) + " rows; the shard gives " +
                                              std::to_string(rows));
                require(weight.k == static_cast<std::int32_t>(info.geometry.column_count),
                        name + " rank " + std::to_string(rank) + " view spans " +
                            std::to_string(weight.k) + " columns");
            }
            for (std::size_t index = 0; index < expected.size(); ++index) {
                require(static_cast<const std::byte*>(base)[index] == expected[index],
                        "the shard view of " + object.name + " rank " + std::to_string(rank) + " byte " +
                            std::to_string(index) + " is not the byte the shard map places there");
            }
        }
    }

    for (int rank = 0; rank < 2; ++rank) {
        require(views.views[static_cast<std::size_t>(rank)].tp == 2,
                "rank " + std::to_string(rank) + " view does not carry the shard width");
    }
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    const bool views_mode    = argc > 1 && std::string_view(argv[1]) == "views";

    try {
        const auto shards = flash::shard_plan_by_name(2);
        test_partition(shards);
        test_expert_subsets(shards);
        if (!artifact_mode && !views_mode) { return g_failures == 0 ? 0 : 1; }

        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }

        Reader reader(artifact);
        Binder binder(reader, 2);
        const auto plan = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 1);
        const auto byte_plan = flash::shard_byte_plan(plan, shards, 2);
        const auto by_object = bound_objects(plan, shards);
        test_replicated(shards, by_object, byte_plan);
        test_byte_plan(plan, shards, by_object, byte_plan);
        test_real_bytes(plan, shards, reader, byte_plan);
        test_scale_plane(shards, reader, byte_plan);

        if (views_mode) { return test_materialized_views(artifact); }
        if (g_failures != 0) { return 1; }
        std::cout << "flash-next shard byte plan: " << plan.objects.size() << " objects x 2 ranks, "
                  << shards.size() << " roles\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
