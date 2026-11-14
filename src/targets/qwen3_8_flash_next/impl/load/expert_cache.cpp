#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

constexpr std::uint64_t kAlign      = 256;
constexpr std::uint64_t kRowTile    = 128;
constexpr std::uint64_t kTileColumn = 64;
constexpr std::uint64_t kTileBytes  = 512;

std::uint64_t align_up(std::uint64_t value, std::uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

void copy_range(const artifact::Reader& reader, std::uint64_t object_offset, std::uint64_t bytes,
                std::span<std::byte> destination, std::size_t& cursor) {
    const auto absolute = reader.payload_offset() + object_offset;
    std::uint64_t copied = 0;
    for (const auto& view : reader.payload_views(absolute, bytes)) {
        std::copy(view.data.begin(), view.data.end(),
                  destination.begin() + static_cast<std::ptrdiff_t>(cursor + copied));
        copied += view.data.size();
    }
    if (copied != bytes) {
        throw artifact::ArtifactError("the payload range is shorter than the pair requests");
    }
    cursor += static_cast<std::size_t>(bytes);
}

const std::array<CoveragePoint, 5> kCurve = {{
    {0, 0.0},
    {5247, 0.990},
    {5861, 0.995},
    {6353, 0.999},
    {kRankPairs, 1.0},
}};

} // namespace

double curve_coverage(std::uint64_t pairs) noexcept {
    double mass = 0.0;
    for (const auto& point : kCurve) {
        if (point.pairs > pairs) { break; }
        mass = point.mass;
    }
    return mass;
}

std::uint64_t pairs_for_coverage(double mass) noexcept {
    for (const auto& point : kCurve) {
        if (point.mass >= mass) { return point.pairs; }
    }
    return kRankPairs;
}

ExpertCachePlan ExpertCachePlan::auto_sized(std::uint64_t available_bytes, std::uint64_t pair_bytes) {
    ExpertCachePlan plan;
    plan.budget_pairs   = available_bytes / pair_bytes;
    plan.target_pairs   = pairs_for_coverage(kCoverageTarget);
    plan.pairs          = std::min(plan.target_pairs, plan.budget_pairs);
    plan.bytes          = plan.pairs * pair_bytes;
    plan.coverage_mass  = curve_coverage(plan.pairs);
    plan.budget_exhausted      = plan.pairs == plan.budget_pairs;
    plan.coverage_target_met   = plan.coverage_mass >= kCoverageTarget;
    return plan;
}

HostPairSource::HostPairSource(const std::filesystem::path& path, std::uint64_t pair_bytes)
    : reader_(path), pair_bytes_(pair_bytes) {
    static const std::array<roles::Role, 3> kRoles = {
        roles::Role::MoeRoutedGate, roles::Role::MoeRoutedUp, roles::Role::MoeRoutedDown,
    };

    for (std::uint32_t layer = 0; layer < kTextLayers; ++layer) {
        std::array<PairGeometry, 3> row{};
        for (std::size_t index = 0; index < kRoles.size(); ++index) {
            const auto name = roles::parameter_name(kRoles[index], layer);
            const auto& bindings = reader_.v3_directory().part_bindings;
            const auto found = bindings.find(name);
            if (found == bindings.end() || found->second.empty()) {
                throw std::runtime_error("the artifact has no expert binding " + name);
            }
            const auto& part = found->second.front();
            const auto* object = reader_.find(part.object);
            if (object == nullptr) {
                throw std::runtime_error("the artifact has no object " + part.object);
            }
            const auto& tensor = std::get<artifact::TensorDescriptor>(*object);
            const auto view = artifact::tensor_element_interval_view(tensor.layout, tensor.format,
                                                                     tensor.shape, part.begin,
                                                                     part.end - part.begin);
            const auto geometry = artifact::batched_block_scale_geometry(tensor.format, tensor.shape);
            row[index] = PairGeometry{
                .base          = artifact::object_offset(*object),
                .matrix_code   = geometry.code_plane_bytes,
                .matrix_scale  = geometry.scale_plane_bytes,
                .scale_plane   = geometry.scale_plane_offset,
                .row_begin     = view.matrix_row_begin,
                .row_count     = view.matrix_row_count,
                .row_bytes     = view.column_count / 2,
                .tile_bytes    = (view.column_count / kTileColumn) * kTileBytes,
                .divisor       = geometry.weight_divisor_offset,
            };
        }
        geometry_.push_back(row);
    }
}

void HostPairSource::read(std::uint32_t layer, std::uint32_t expert,
                          std::span<std::byte> destination) const {
    if (destination.size() != static_cast<std::size_t>(pair_bytes_)) {
        throw std::invalid_argument("the host pair source reads " + std::to_string(pair_bytes_) +
                                    " bytes per pair; the destination holds " +
                                    std::to_string(destination.size()));
    }

    std::size_t cursor = 0;
    for (const auto& geometry : geometry_[layer]) {
        const auto code_start = expert * geometry.matrix_code + geometry.row_begin * geometry.row_bytes;
        const auto code_bytes = geometry.row_count * geometry.row_bytes;
        copy_range(reader_, geometry.base + code_start, code_bytes, destination, cursor);

        const auto scale_start = geometry.scale_plane + expert * geometry.matrix_scale +
                                 (geometry.row_begin / kRowTile) * geometry.tile_bytes;
        const auto scale_bytes = (geometry.row_count / kRowTile) * geometry.tile_bytes;
        copy_range(reader_, geometry.base + scale_start, scale_bytes, destination, cursor);
    }
}

void HostPairSource::read_divisors(std::uint32_t layer, std::uint32_t expert,
                                   float (&out)[3]) const {
    for (std::size_t index = 0; index < geometry_[layer].size(); ++index) {
        const auto& geometry = geometry_[layer][index];
        std::array<std::byte, 4> word{};
        std::size_t cursor = 0;
        copy_range(reader_, geometry.base + geometry.divisor + expert * 4, 4, word, cursor);
        std::memcpy(&out[index], word.data(), sizeof(float));
    }
}

ExpertCache::ExpertCache(const ExpertCacheOptions& options, std::uint64_t pair_bytes,
                         std::uint64_t available_bytes, HostPairSource& source)
    : pair_bytes_(pair_bytes), source_(source) {
    const std::uint64_t budget = available_bytes - options.vram_reserve_bytes;

    if (options.capacity_bytes.has_value()) {
        if (*options.capacity_bytes > budget) {
            throw std::invalid_argument("expert cache capacity " +
                                        std::to_string(*options.capacity_bytes) +
                                        " bytes exceeds the available device memory " +
                                        std::to_string(available_bytes) + " bytes, with " +
                                        std::to_string(options.vram_reserve_bytes) +
                                        " bytes reserved");
        }
        plan_.pairs        = *options.capacity_bytes / pair_bytes;
        plan_.budget_pairs = budget / pair_bytes;
        plan_.target_pairs = pairs_for_coverage(kCoverageTarget);
    } else {
        plan_ = ExpertCachePlan::auto_sized(budget, pair_bytes);
    }

    plan_.bytes              = plan_.pairs * pair_bytes;
    plan_.coverage_mass      = curve_coverage(plan_.pairs);
    plan_.budget_exhausted   = plan_.pairs == plan_.budget_pairs;
    plan_.coverage_target_met = plan_.coverage_mass >= kCoverageTarget;

    if (plan_.pairs == 0) {
        throw std::invalid_argument(
            "the available device memory holds no resident expert pair");
    }

    slot_of_.assign(static_cast<std::size_t>(kRankPairs), -1);
    resident_.assign(static_cast<std::size_t>(plan_.pairs), {0, 0});
    reference_.assign(static_cast<std::size_t>(plan_.pairs), 0);
    spare_mark_.assign(static_cast<std::size_t>(plan_.pairs), 0);

    if (options.profile.has_value()) { seed(*options.profile); }
}

std::uint64_t ExpertCache::key(std::uint32_t layer, std::uint32_t expert) const noexcept {
    return static_cast<std::uint64_t>(layer) * kRankExperts + expert;
}

void ExpertCache::assign(std::uint32_t slot, std::uint32_t layer, std::uint32_t expert) {
    slot_of_[key(layer, expert)] = static_cast<std::int32_t>(slot);
    resident_[slot]              = {layer, expert};
    reference_[slot]             = 1;
}

std::uint32_t ExpertCache::place(std::uint32_t layer, std::uint32_t expert) {
    const auto slots = static_cast<std::uint32_t>(plan_.pairs);

    if (!spare_.empty()) {
        const auto slot = spare_.back();
        spare_.pop_back();
        spare_mark_[slot] = 0;
        assign(slot, layer, expert);
        ++stats_.insertions;
        ++stats_.residency;
        return slot;
    }

    if (stats_.residency < slots) {
        const auto slot = next_free_++;
        assign(slot, layer, expert);
        ++stats_.insertions;
        ++stats_.residency;
        hand_ = (slot + 1) % slots;
        return slot;
    }

    while (reference_[hand_]) {
        reference_[hand_] = 0;
        hand_ = (hand_ + 1) % slots;
    }

    const auto slot = hand_;
    const auto evicted = resident_[slot];
    slot_of_[key(evicted.first, evicted.second)] = -1;
    ++stats_.evictions;
    assign(slot, layer, expert);
    ++stats_.insertions;
    hand_ = (slot + 1) % slots;
    return slot;
}

void ExpertCache::seed(const std::filesystem::path& profile) {
    std::ifstream input(profile);
    if (!input) {
        throw std::runtime_error("cannot read the expert cache profile " + profile.string());
    }

    struct Entry {
        std::uint32_t layer;
        std::uint32_t expert;
        std::uint64_t count;
    };

    std::vector<Entry> entries;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        Entry entry{};
        if (!(fields >> entry.layer >> entry.expert >> entry.count)) { continue; }
        entries.push_back(entry);
    }

    std::stable_sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        return left.count > right.count;
    });

    for (const auto& entry : entries) {
        if (stats_.residency == static_cast<std::uint32_t>(plan_.pairs)) { break; }
        place(entry.layer, entry.expert);
    }
}

std::int32_t ExpertCache::resident_slot(std::uint32_t layer, std::uint32_t expert) const noexcept {
    return slot_of_[key(layer, expert)];
}

void ExpertCache::reserve(
    std::uint32_t count,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& protected_pairs) {
    const auto slots = static_cast<std::uint32_t>(plan_.pairs);
    const std::uint32_t never_used = slots - next_free_;
    if (spare_.size() + never_used >= count) { return; }

    std::vector<bool> protected_slot(static_cast<std::size_t>(slots), false);
    for (const auto& [layer, expert] : protected_pairs) {
        const auto slot = slot_of_[key(layer, expert)];
        if (slot >= 0) { protected_slot[static_cast<std::size_t>(slot)] = true; }
    }

    std::uint32_t need = count - (spare_.size() + never_used);
    while (need != 0) {
        bool found = false;
        for (std::uint32_t step = 0; step < 2 * slots; ++step) {
            const std::uint32_t slot = hand_;
            hand_ = (hand_ + 1) % slots;
            if (slot >= next_free_) { continue; }
            if (spare_mark_[slot] != 0) { continue; }
            if (protected_slot[slot]) { continue; }
            if (reference_[slot]) {
                reference_[slot] = 0;
                continue;
            }
            const auto evicted = resident_[slot];
            slot_of_[key(evicted.first, evicted.second)] = -1;
            reference_[slot] = 0;
            resident_[slot] = {0, 0};
            spare_mark_[slot] = 1;
            spare_.push_back(slot);
            --stats_.residency;
            --need;
            found = true;
            break;
        }
        if (!found) {
            throw std::logic_error("the expert cache cannot reserve a slot outside the protected set");
        }
    }
}

std::optional<std::uint32_t> ExpertCache::find(std::uint32_t layer, std::uint32_t expert) noexcept {
    ++stats_.lookups;
    const auto slot = slot_of_[key(layer, expert)];
    if (slot < 0) {
        ++stats_.miss_count;
        return std::nullopt;
    }
    ++stats_.hits;
    return static_cast<std::uint32_t>(slot);
}

std::uint32_t ExpertCache::load(std::uint32_t layer, std::uint32_t expert) {
    ++stats_.lookups;
    const auto slot = slot_of_[key(layer, expert)];
    if (slot >= 0) {
        ++stats_.hits;
        reference_[slot] = 1;
        return static_cast<std::uint32_t>(slot);
    }
    ++stats_.miss_count;
    return place(layer, expert);
}

std::uint64_t ExpertCache::slot_offset(std::uint32_t slot) const noexcept {
    return static_cast<std::uint64_t>(slot) * pair_bytes_;
}

std::uint32_t ExpertCache::slots() const noexcept {
    return static_cast<std::uint32_t>(plan_.pairs);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
