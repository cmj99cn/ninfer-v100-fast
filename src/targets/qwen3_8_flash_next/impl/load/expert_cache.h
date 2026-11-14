#pragma once

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
inline constexpr std::uint64_t kPairCode  = static_cast<std::uint64_t>(TextConfig::expert_intermediate) *
                                            static_cast<std::uint64_t>(TextConfig::hidden) / 2;
inline constexpr std::uint64_t kPairScale = static_cast<std::uint64_t>(TextConfig::expert_intermediate) *
                                            static_cast<std::uint64_t>(TextConfig::hidden) / 16;
inline constexpr std::uint64_t kPairBytes = 3 * (kPairCode + kPairScale);
static_assert(kPairBytes == 2764800);

inline constexpr std::uint32_t kTextLayers  = static_cast<std::uint32_t>(TextConfig::layers);
inline constexpr std::uint32_t kTp          = 2;
inline constexpr std::uint32_t kRankExperts = static_cast<std::uint32_t>(TextConfig::experts) / kTp;
inline constexpr std::uint64_t kRankPairs   = static_cast<std::uint64_t>(kTextLayers) * kRankExperts;
struct CoveragePoint {
    std::uint64_t pairs;
    double mass;
};

[[nodiscard]] double curve_coverage(std::uint64_t pairs) noexcept;
[[nodiscard]] std::uint64_t pairs_for_coverage(double mass) noexcept;

inline constexpr double kCoverageTarget = 0.995;

struct ExpertCachePlan {
    std::uint64_t pairs      = 0; // resident pairs on this rank
    std::uint64_t bytes      = 0; // pairs * pair_bytes
    std::uint64_t budget_pairs = 0;
    std::uint64_t target_pairs = 0;
    double coverage_mass       = 0.0;
    bool budget_exhausted      = false;
    bool coverage_target_met   = false;
    [[nodiscard]] static ExpertCachePlan auto_sized(std::uint64_t available_bytes,
                                                     std::uint64_t pair_bytes);
};

struct ExpertCacheOptions {
    std::optional<std::uint64_t> capacity_bytes;   // nullopt selects the auto plan
    std::uint64_t vram_reserve_bytes               = 0;
    std::optional<std::filesystem::path> profile;
};

struct CacheStats {
    std::uint64_t lookups    = 0;
    std::uint64_t hits       = 0;
    std::uint64_t miss_count = 0;
    std::uint64_t insertions = 0;
    std::uint64_t evictions  = 0;
    std::uint32_t residency  = 0;
};
class HostPairSource {
public:
    HostPairSource(const std::filesystem::path& path, std::uint64_t pair_bytes);

    void read(std::uint32_t layer, std::uint32_t expert, std::span<std::byte> destination) const;

    [[nodiscard]] std::uint64_t pair_bytes() const noexcept { return pair_bytes_; }

private:
    struct PairGeometry {
        std::string object;
        std::uint64_t base         = 0;
        std::uint64_t matrix_code  = 0;
        std::uint64_t matrix_scale = 0;
        std::uint64_t scale_plane  = 0;
        std::uint64_t row_begin    = 0;
        std::uint64_t row_count    = 0;
        std::uint64_t row_bytes    = 0;
        std::uint64_t tile_bytes   = 0;
    };

    artifact::Reader reader_;
    std::uint64_t pair_bytes_;
    std::vector<std::array<PairGeometry, 3>> geometry_;
};
class ExpertCache {
public:
    ExpertCache(const ExpertCacheOptions& options, std::uint64_t pair_bytes,
                std::uint64_t available_bytes, HostPairSource& source);

    [[nodiscard]] std::optional<std::uint32_t> find(std::uint32_t layer, std::uint32_t expert) noexcept;
    [[nodiscard]] std::uint32_t load(std::uint32_t layer, std::uint32_t expert);

    [[nodiscard]] std::uint64_t slot_offset(std::uint32_t slot) const noexcept;
    [[nodiscard]] std::uint32_t slots() const noexcept;
    [[nodiscard]] const ExpertCachePlan& plan() const noexcept { return plan_; }
    [[nodiscard]] HostPairSource& source() noexcept { return source_; }
    [[nodiscard]] CacheStats stats() const noexcept { return stats_; }

private:
    std::uint64_t key(std::uint32_t layer, std::uint32_t expert) const noexcept;
    std::uint32_t place(std::uint32_t layer, std::uint32_t expert);
    void assign(std::uint32_t slot, std::uint32_t layer, std::uint32_t expert);
    void seed(const std::filesystem::path& profile);

    std::uint64_t pair_bytes_;
    ExpertCachePlan plan_;
    HostPairSource& source_;
    std::vector<std::int32_t> slot_of_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> resident_;
    std::vector<std::uint8_t> reference_;
    std::uint32_t hand_      = 0;
    std::uint32_t next_free_ = 0;
    CacheStats stats_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
