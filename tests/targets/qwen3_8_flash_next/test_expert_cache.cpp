// Host-only planning layer of the per-rank expert residency cache; the device layer needs two
// devices and is skipped when they cannot be taken.
//
// The cache owns one rank's own expert set: at tp 2 that is 256 whole matrices per layer (R27),
// so the pair universe is 48 * 256 = 12288 pairs and every byte count here is per rank.
//
// Every expected byte mapping is computed here from the storage-layout formulas of
// docs/maintainer/storage-layouts.md (sections 4 and 11) over the artifact's own stored directory,
// read through the reader. The package's own arithmetic is only the thing under test.

#include "artifact/reader.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/load/expert_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <variant>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace artifact = ninfer::artifact;
namespace cache = ninfer::targets::qwen3_8_flash_next::detail;

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
constexpr std::uint64_t kGroup      = 16;

std::uint64_t align_up(std::uint64_t value, std::uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

std::uint64_t hash_of(std::span<const std::byte> bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : bytes) {
        hash = (hash ^ static_cast<std::uint64_t>(byte)) * 1099511628211ULL;
    }
    return hash;
}

// storage-layouts.md sections 4 and 11: a batched NVFP4 object is a matrix-major stack whose code
// plane is row-major over the logical columns, whose swizzled scale plane follows the code plane
// on a 256-byte alignment, and whose per-matrix scale block is a run of 128-row tiles of
// (columns / 64) * 512 bytes.
struct Segment {
    std::string object;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

std::vector<Segment> oracle_pair(const artifact::Reader& reader, std::uint32_t layer,
                                 std::uint32_t expert) {
    static const char* kRoles[3] = {"gate", "up", "down"};
    std::vector<Segment> out;
    for (const char* role : kRoles) {
        const std::string name = "text/layers/" + std::to_string(layer) + "/moe/experts/" + role;
        const auto& parts = reader.v3_directory().part_bindings.at(name);
        const auto& part  = parts[expert];
        const auto* object = reader.find(part.object);
        if (object == nullptr) { throw std::runtime_error("no object " + part.object); }
        const auto& tensor = std::get<artifact::TensorDescriptor>(*object);
        const auto shape = tensor.shape;
        const std::uint64_t matrices = shape[0];
        const std::uint64_t rows     = shape[1];
        const std::uint64_t columns  = shape[2];
        const std::uint64_t elements = matrices * rows * columns;
        if (part.begin >= elements || part.end > elements) {
            throw std::runtime_error("the part of " + part.object + " is outside the tensor");
        }
        const std::uint64_t matrix    = part.begin / (rows * columns);
        const std::uint64_t row_begin = (part.begin % (rows * columns)) / columns;
        const std::uint64_t row_count = (part.end - part.begin) / columns;
        const std::uint64_t code_bytes  = rows * columns / 2;
        const std::uint64_t scale_bytes = rows * columns / kGroup;
        const std::uint64_t plane       = align_up(matrices * code_bytes, kAlign);
        const std::uint64_t base        = artifact::object_offset(*object);
        out.push_back({part.object, base + matrix * code_bytes + row_begin * (columns / 2),
                       row_count * (columns / 2)});
        const std::uint64_t tiles = columns / kTileColumn;
        out.push_back({part.object,
                       base + plane + matrix * scale_bytes + (row_begin / kRowTile) * tiles * kTileBytes,
                       (row_count / kRowTile) * tiles * kTileBytes});
    }
    return out;
}

std::uint64_t oracle_pair_bytes(const artifact::Reader& reader, std::uint32_t layer) {
    std::uint64_t total = 0;
    for (const auto& segment : oracle_pair(reader, layer, 0)) { total += segment.bytes; }
    return total;
}

// The measured routing skew (4 layers x 512 tokens, 20480 routings) as the test's own table.
struct CurvePoint {
    double fraction;
    double mass;
};

const CurvePoint kCurve[] = {{0.427, 0.990}, {0.477, 0.995}, {0.517, 0.999}};

std::uint64_t curve_pairs(double fraction) {
    return static_cast<std::uint64_t>(fraction * static_cast<double>(cache::kRankPairs) + 0.5);
}

void test_coverage_curve() {
    require(cache::kRankPairs == 48 * 256,
            "the per-rank pair universe is " + std::to_string(cache::kRankPairs) +
                "; tp 2 gives 48 layers x 256 experts = 12288");

    double previous_mass = 0.0;
    for (const auto& point : kCurve) {
        const auto pairs = curve_pairs(point.fraction);
        require(cache::curve_coverage(pairs) == point.mass,
                "coverage at " + std::to_string(pairs) + " pairs is " +
                    std::to_string(cache::curve_coverage(pairs)) + "; the measured curve gives " +
                    std::to_string(point.mass));
        require(cache::curve_coverage(pairs - 1) == previous_mass,
                "coverage at " + std::to_string(pairs - 1) + " pairs is " +
                    std::to_string(cache::curve_coverage(pairs - 1)) + "; the step below the " +
                    std::to_string(point.fraction) + " point is " + std::to_string(previous_mass));
        previous_mass = point.mass;
    }

    for (std::uint64_t pairs = 1; pairs <= cache::kRankPairs; pairs += 257) {
        require(cache::curve_coverage(pairs) >= cache::curve_coverage(pairs - 1),
                "the coverage curve is not monotonic at " + std::to_string(pairs));
    }

    require(cache::curve_coverage(cache::kRankPairs) == 1.0,
            "a cache holding every pair of the rank covers " +
                std::to_string(cache::curve_coverage(cache::kRankPairs)) + " of the routed mass");

    require(cache::pairs_for_coverage(cache::kCoverageTarget) == curve_pairs(0.477),
            "the coverage target needs " + std::to_string(cache::pairs_for_coverage(cache::kCoverageTarget)) +
                " pairs; the measured curve needs " + std::to_string(curve_pairs(0.477)));
}

void test_auto_sized(const char* path) {
    const artifact::Reader reader(path);
    const std::uint64_t pair_bytes = oracle_pair_bytes(reader, 0);
    require(pair_bytes == cache::kPairBytes,
            "the artifact sums to " + std::to_string(pair_bytes) + " bytes per pair; the cache says " +
                std::to_string(cache::kPairBytes));

    struct Case {
        std::uint64_t available;
        std::uint64_t expected_pairs;
        double expected_mass;
        bool budget_governs;
    };

    const std::uint64_t measured_budget = 24800000000ULL;
    const std::uint64_t short_budget    = curve_pairs(0.427) * pair_bytes;
    const std::uint64_t full_budget     = (cache::kRankPairs + 4) * pair_bytes;

    const Case cases[] = {
        {measured_budget, curve_pairs(0.477), 0.995, false},
        {short_budget, curve_pairs(0.427), 0.990, true},
        {full_budget, curve_pairs(0.477), 0.995, false},
    };

    for (const auto& case_ : cases) {
        const auto plan = cache::ExpertCachePlan::auto_sized(case_.available, pair_bytes);
        require(plan.pairs == case_.expected_pairs,
                "auto sizing at " + std::to_string(case_.available) + " available bytes picks " +
                    std::to_string(plan.pairs) + " pairs; the curve and the budget give " +
                    std::to_string(case_.expected_pairs));
        require(plan.bytes == plan.pairs * pair_bytes,
                "the plan covers " + std::to_string(plan.bytes) + " bytes; " +
                    std::to_string(plan.pairs) + " pairs is " +
                    std::to_string(plan.pairs * pair_bytes));
        require(plan.budget_pairs == case_.available / pair_bytes,
                "auto sizing reports " + std::to_string(plan.budget_pairs) +
                    " budget pairs at " + std::to_string(case_.available) + " bytes");
        require(plan.coverage_mass == case_.expected_mass,
                "auto sizing at " + std::to_string(case_.available) + " bytes reports " +
                    std::to_string(plan.coverage_mass) + " mass; the curve gives " +
                    std::to_string(case_.expected_mass));

        require(plan.target_pairs == cache::pairs_for_coverage(cache::kCoverageTarget),
                "auto sizing reports target pairs " + std::to_string(plan.target_pairs) +
                    "; the coverage target needs " +
                    std::to_string(cache::pairs_for_coverage(cache::kCoverageTarget)));

        require(plan.coverage_target_met || plan.budget_exhausted,
                "auto sizing at " + std::to_string(case_.available) + " bytes satisfies neither "
                "the coverage target nor the budget");
        require(plan.budget_exhausted == case_.budget_governs,
                "auto sizing at " + std::to_string(case_.available) + " bytes reports governed_by " +
                    (plan.budget_exhausted ? "budget" : "coverage target") + "; the case expects " +
                    (case_.budget_governs ? "budget" : "coverage target"));

        if (case_.budget_governs) {
            require(plan.budget_pairs < plan.target_pairs,
                    "the case calls the budget governing at " + std::to_string(case_.available) +
                        " bytes, where the budget holds " + std::to_string(plan.budget_pairs) +
                        " pairs and the coverage target needs " + std::to_string(plan.target_pairs));
            require(plan.pairs == plan.budget_pairs,
                    "a governing budget at " + std::to_string(case_.available) + " bytes plans " +
                        std::to_string(plan.pairs) + " pairs; it holds " +
                        std::to_string(plan.budget_pairs));
        } else {
            require(plan.budget_pairs > plan.target_pairs,
                    "the case calls the coverage target governing at " +
                        std::to_string(case_.available) + " bytes, where the budget holds " +
                        std::to_string(plan.budget_pairs) + " pairs and the target needs " +
                        std::to_string(plan.target_pairs));
            require(plan.pairs == plan.target_pairs,
                    "a governing coverage target plans " + std::to_string(plan.pairs) +
                        " pairs at " + std::to_string(case_.available) + " bytes; it needs " +
                        std::to_string(plan.target_pairs) + " and the budget would hold " +
                        std::to_string(plan.budget_pairs));
        }

        std::cout << "flash-next expert cache: " << std::to_string(plan.pairs) << " pairs ("
                  << std::to_string(plan.bytes) << " B) at " << std::to_string(case_.available)
                  << " available bytes, coverage " << std::to_string(plan.coverage_mass * 100.0)
                  << "% mass, budget " << std::to_string(plan.budget_pairs) << " pairs, governed by "
                  << (plan.budget_exhausted ? "budget" : "coverage target") << '\n';
    }
}

void test_explicit_capacity_over_budget(const char* path) {
    const std::uint64_t available = 24800000000ULL;

    cache::ExpertCacheOptions options;
    options.capacity_bytes = 30000000000ULL;

    bool threw = false;
    try {
        cache::HostPairSource source(path, cache::kPairBytes);
        cache::ExpertCache expert(options, cache::kPairBytes, available, source);
        (void)expert;
    } catch (const std::invalid_argument& error) {
        threw = true;
        const std::string message = error.what();
        require(message.find(std::to_string(available)) != std::string::npos,
                "the rejection does not report the available bytes: " + message);
        require(message.find(std::to_string(*options.capacity_bytes)) != std::string::npos,
                "the rejection does not report the requested bytes: " + message);
    }
    require(threw, "an explicit capacity over the available memory is accepted");

    options.capacity_bytes = 6000000000ULL;
    options.vram_reserve_bytes = 1000000000ULL;
    cache::HostPairSource source(path, cache::kPairBytes);
    cache::ExpertCache expert(options, cache::kPairBytes, available, source);
    require(expert.plan().pairs == 6000000000ULL / cache::kPairBytes,
            "an explicit capacity inside the budget plans " + std::to_string(expert.plan().pairs) +
                " pairs; 6000000000 bytes is " +
                std::to_string(6000000000ULL / cache::kPairBytes) + " pairs");

    options.capacity_bytes = std::nullopt;
    options.vram_reserve_bytes = 0;
    cache::ExpertCache automatic(options, cache::kPairBytes, available, source);
    require(automatic.plan().pairs == cache::pairs_for_coverage(cache::kCoverageTarget),
            "auto capacity at " + std::to_string(available) + " bytes plans " +
                std::to_string(automatic.plan().pairs) + " pairs; the coverage target needs " +
                std::to_string(cache::pairs_for_coverage(cache::kCoverageTarget)) +
                " and the budget would hold " + std::to_string(available / cache::kPairBytes));
    require(!automatic.plan().budget_exhausted && automatic.plan().coverage_target_met,
            "auto capacity at " + std::to_string(available) + " bytes reports itself governed by " +
                (automatic.plan().budget_exhausted ? "the budget" : "the coverage target"));
    require(automatic.plan().bytes < available,
            "auto capacity at " + std::to_string(available) + " bytes plans " +
                std::to_string(automatic.plan().bytes) + " bytes, leaving nothing for the resident KV "
                "and the MTP proposal head");
}

// find and load each record one lookup; load additionally records the placement.
void test_slot_table(const char* path) {
    cache::ExpertCacheOptions options;
    options.capacity_bytes = 100 * cache::kPairBytes;
    cache::HostPairSource source(path, cache::kPairBytes);
    cache::ExpertCache expert(options, cache::kPairBytes, 24800000000ULL, source);

    require(expert.slots() == 100, "the cache holds " + std::to_string(expert.slots()) +
        " slots; the capacity asked for 100");
    for (std::uint32_t slot = 0; slot < expert.slots(); ++slot) {
        require(expert.slot_offset(slot) == static_cast<std::uint64_t>(slot) * cache::kPairBytes,
                "slot " + std::to_string(slot) + " starts at " +
                    std::to_string(expert.slot_offset(slot)) + "; the fixed slab layout gives " +
                    std::to_string(static_cast<std::uint64_t>(slot) * cache::kPairBytes));
    }

    const auto miss = expert.find(7, 41);
    require(!miss.has_value(), "find on a pair the cache does not hold returns a slot");
    require(expert.stats().lookups == 1, "find recorded " +
        std::to_string(expert.stats().lookups) + " lookups; it records one");
    require(expert.stats().miss_count == 1, "find on a miss counted " +
        std::to_string(expert.stats().miss_count) + " misses; it counts one");

    const auto slot = expert.load(7, 41);
    require(slot < expert.slots(), "load returned slot " + std::to_string(slot) +
        " outside the " + std::to_string(expert.slots()) + " slots");
    require(expert.stats().lookups == 2, "find and load recorded " +
        std::to_string(expert.stats().lookups) + " lookups; each records one");
    require(expert.stats().miss_count == 2, "load recorded " +
        std::to_string(expert.stats().miss_count) + " misses; find and load each record one");
    require(expert.stats().insertions == 1, "load counted " +
        std::to_string(expert.stats().insertions) + " insertions; it counts one");
    require(expert.stats().evictions == 0, "the first placement evicted " +
        std::to_string(expert.stats().evictions) + " pairs");
    require(expert.stats().residency == 1, "residency is " +
        std::to_string(expert.stats().residency) + " after one placement");

    const auto again = expert.find(7, 41);
    require(again == slot, "the pair moved from slot " + std::to_string(slot) + " to " +
        std::to_string(again.value_or(999)));
    require(expert.stats().hits == 1, "the hit after the placement counted " +
        std::to_string(expert.stats().hits) + " hits; it counts one");

    const auto other = expert.find(7, 42);
    require(!other.has_value(), "find on a second pair returns a slot");
    require(expert.stats().miss_count == 3, "the lookups recorded " +
        std::to_string(expert.stats().miss_count) + " misses; find, load and find record one each");
    require(expert.find(7, 41) == slot, "the resident pair left slot " + std::to_string(slot) +
        " after an unrelated lookup");
}

// Hand-worked clock trace over three slots, no profile, hand starts at slot 0. A placement sets
// the reference bit and moves the hand; a hit sets the reference bit only; a full-cache miss
// clears bits from the hand until it finds a clear one and evicts there.
void test_replacement_is_deterministic(const char* path) {
    cache::ExpertCacheOptions options;
    options.capacity_bytes = 3 * cache::kPairBytes;
    cache::HostPairSource source(path, cache::kPairBytes);

    const std::vector<std::pair<std::uint32_t, std::uint32_t>> sequence = {
        {0, 7}, {1, 3}, {2, 11}, {0, 7}, {3, 5}, {1, 3}, {4, 9}, {5, 1},
    };
    const std::uint32_t expected[] = {0, 1, 2, 0, 0, 1, 2, 1};

    cache::ExpertCache first(options, cache::kPairBytes, 3 * cache::kPairBytes, source);
    cache::ExpertCache second(options, cache::kPairBytes, 3 * cache::kPairBytes, source);

    for (std::size_t step = 0; step < sequence.size(); ++step) {
        const auto pair = sequence[step];
        const auto slot = first.load(pair.first, pair.second);
        require(slot == expected[step],
                "step " + std::to_string(step) + " put pair (" + std::to_string(pair.first) + "," +
                    std::to_string(pair.second) + ") in slot " + std::to_string(slot) +
                    "; the hand-worked clock trace gives " + std::to_string(expected[step]));
        require(second.load(pair.first, pair.second) == slot,
                "the same sequence allocated a different slot at step " + std::to_string(step));
    }
    require(first.stats().lookups == 8, "the trace recorded " +
        std::to_string(first.stats().lookups) + " lookups over 8 loads; it records one per load");
    require(first.stats().miss_count == 6, "the trace counted " +
        std::to_string(first.stats().miss_count) + " misses; the hand-worked trace counts 6");
    require(first.stats().hits == 2, "the trace counted " +
        std::to_string(first.stats().hits) + " hits; the hand-worked trace counts 2");
    require(first.stats().insertions == 6, "the trace counted " +
        std::to_string(first.stats().insertions) + " insertions; the hand-worked trace counts 6");
    require(first.stats().evictions == 3, "the trace evicted " +
        std::to_string(first.stats().evictions) + " pairs; the hand-worked trace evicts 3");
    require(first.stats().residency == 3, "the trace ends at residency " +
        std::to_string(first.stats().residency) + " over 3 slots");

    std::vector<std::pair<std::uint32_t, std::uint32_t>> long_sequence;
    std::uint64_t state = 88172645463325252ULL;
    for (std::size_t step = 0; step < 4096; ++step) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const std::uint32_t layer  = static_cast<std::uint32_t>(state % cache::kTextLayers);
        const std::uint32_t expert = static_cast<std::uint32_t>((state >> 32) % cache::kRankExperts);
        long_sequence.push_back({layer, expert});
    }

    cache::ExpertCache third(options, cache::kPairBytes, 3 * cache::kPairBytes, source);
    cache::ExpertCache fourth(options, cache::kPairBytes, 3 * cache::kPairBytes, source);
    for (const auto& pair : long_sequence) {
        const auto left  = third.load(pair.first, pair.second);
        const auto right = fourth.load(pair.first, pair.second);
        require(left == right, "the long sequence allocated different slots for pair (" +
            std::to_string(pair.first) + "," + std::to_string(pair.second) + ")");
    }
    require(third.stats().lookups == fourth.stats().lookups &&
                third.stats().hits == fourth.stats().hits &&
                third.stats().miss_count == fourth.stats().miss_count &&
                third.stats().insertions == fourth.stats().insertions &&
                third.stats().evictions == fourth.stats().evictions &&
                third.stats().residency == fourth.stats().residency,
            "the same sequence produced different statistics");
}

void test_profile_seeds_the_fill_order(const char* path, const std::filesystem::path& profile) {
    cache::ExpertCacheOptions options;
    options.capacity_bytes = 3 * cache::kPairBytes;
    options.profile = profile;
    cache::HostPairSource source(path, cache::kPairBytes);
    cache::ExpertCache expert(options, cache::kPairBytes, 3 * cache::kPairBytes, source);

    require(expert.stats().residency == 3, "the profile seeded " +
        std::to_string(expert.stats().residency) + " slots; the top three pairs fill all three");
    require(expert.stats().lookups == 0, "seeding counted " +
        std::to_string(expert.stats().lookups) + " lookups; seeding performs none");

    const std::vector<std::pair<std::uint32_t, std::uint32_t>> seeded = {{12, 200}, {3, 100}, {40, 7}};
    for (std::size_t index = 0; index < seeded.size(); ++index) {
        const auto slot = expert.find(seeded[index].first, seeded[index].second);
        require(slot.has_value(), "the profile did not seed pair (" +
            std::to_string(seeded[index].first) + "," + std::to_string(seeded[index].second) + ")");
        require(slot.value() == static_cast<std::uint32_t>(index),
                "the hottest profile pairs fill the slots out of order: (" +
                    std::to_string(seeded[index].first) + "," +
                    std::to_string(seeded[index].second) + ") is in slot " +
                    std::to_string(slot.value()));
    }

    const auto unseeded = expert.find(12, 201);
    require(!unseeded.has_value(), "a pair outside the profile is resident");
    const auto slot = expert.load(12, 201);
    require(slot < expert.slots(), "load past the seeded capacity returned slot " +
        std::to_string(slot));
    require(expert.stats().evictions == 1, "filling past the capacity evicted " +
        std::to_string(expert.stats().evictions) + " pair; the clock evicts one");
    require(expert.find(3, 100) == 1 && expert.find(40, 7) == 2,
            "the clock evicted a seeded pair that still had its reference bit");
    require(!expert.find(12, 200).has_value(), "the evicted slot still names the pair it lost");
    require(expert.find(12, 201) == slot, "the loaded pair is not in the slot load returned");
}

void test_empty_profile(const char* path, const std::filesystem::path& profile) {
    cache::ExpertCacheOptions options;
    options.capacity_bytes = 3 * cache::kPairBytes;
    options.profile = profile;
    cache::HostPairSource source(path, cache::kPairBytes);
    cache::ExpertCache expert(options, cache::kPairBytes, 3 * cache::kPairBytes, source);

    require(expert.stats().residency == 0, "an empty profile seeded " +
        std::to_string(expert.stats().residency) + " slots; it seeds none");
    require(expert.stats().insertions == 0, "an empty profile recorded " +
        std::to_string(expert.stats().insertions) + " insertions; it inserts nothing");

    const auto slot = expert.load(20, 200);
    require(slot == 0, "the first load after an empty profile took slot " +
        std::to_string(slot) + "; the clock takes the first free slot");
    require(expert.stats().evictions == 0, "the first load after an empty profile evicted " +
        std::to_string(expert.stats().evictions) + " pairs; the cache was not full");
    require(expert.stats().insertions == 1, "the first load after an empty profile recorded " +
        std::to_string(expert.stats().insertions) + " insertions");
    require(expert.stats().residency == 1, "residency is " +
        std::to_string(expert.stats().residency) + " after the first load past an empty profile");
}

std::vector<std::byte> direct_pair(const artifact::Reader& reader, std::uint32_t layer,
                                   std::uint32_t expert) {
    std::vector<std::byte> direct(static_cast<std::size_t>(cache::kPairBytes));
    std::size_t cursor = 0;
    for (const auto& segment : oracle_pair(reader, layer, expert)) {
        if (reader.find(segment.object) == nullptr) {
            throw std::runtime_error("no object " + segment.object);
        }
        for (const auto& view : reader.payload_views(
                 reader.payload_offset() + segment.offset, segment.bytes)) {
            std::copy(view.data.begin(), view.data.end(),
                      direct.begin() + static_cast<std::ptrdiff_t>(cursor));
            cursor += view.data.size();
        }
    }
    require(cursor == cache::kPairBytes, "the artifact pair at layer " + std::to_string(layer) +
        " expert " + std::to_string(expert) + " sums to " + std::to_string(cursor) + " bytes");
    return direct;
}

bool pair_spans_volumes(const artifact::Reader& reader, std::uint32_t layer, std::uint32_t expert) {
    for (const auto& segment : oracle_pair(reader, layer, expert)) {
        if (reader.payload_segments(reader.payload_offset() + segment.offset, segment.bytes).size() > 1) {
            return true;
        }
    }
    return false;
}

std::pair<std::uint32_t, std::uint32_t> find_spanning_pair(const artifact::Reader& reader) {
    for (std::uint32_t layer = 0; layer < cache::kTextLayers; ++layer) {
        for (std::uint32_t expert = 0; expert < cache::kRankExperts; ++expert) {
            if (pair_spans_volumes(reader, layer, expert)) { return {layer, expert}; }
        }
    }
    throw std::runtime_error("the artifact has no pair whose bytes span volumes");
}

void test_host_pair_source(const char* path) {
    const artifact::Reader reader(path);
    cache::HostPairSource source(path, cache::kPairBytes);

    std::vector<std::byte> buffer(cache::kPairBytes);
    for (const std::uint32_t layer : {0U, 17U, 47U}) {
        for (const std::uint32_t expert : {0U, 128U, 255U}) {
            source.read(layer, expert, buffer);
            require(hash_of(buffer) == hash_of(direct_pair(reader, layer, expert)),
                "the host pair source read a different pair than the artifact holds at layer " +
                    std::to_string(layer) + " expert " + std::to_string(expert));
        }
    }
    std::cout << "flash-next expert cache: the host pair source reads " << cache::kPairBytes
              << " bytes per pair and matches the artifact byte for byte\n";
}

void test_cross_volume_pair(const char* path) {
    const artifact::Reader reader(path);
    cache::HostPairSource source(path, cache::kPairBytes);

    const auto spanning = find_spanning_pair(reader);
    std::vector<std::byte> buffer(cache::kPairBytes);
    source.read(spanning.first, spanning.second, buffer);
    require(hash_of(buffer) == hash_of(direct_pair(reader, spanning.first, spanning.second)),
        "the host pair source misreads the pair that spans volumes at layer " +
            std::to_string(spanning.first) + " expert " + std::to_string(spanning.second));
    std::cout << "flash-next expert cache: the spanning pair at layer " << spanning.first
              << " expert " << spanning.second << " reads across volumes\n";
}

// The device layer: a fixed device slab whose slot addresses stay put, and the miss path landing
// the artifact bytes in the slot it resolved. Needs two devices.
int test_device_slots(const char* path) {
    ninfer::ExecutionContext execution({0, 1});
    if (execution.tp != 2) {
        std::cout << "skip: two devices are not available\n";
        return 77;
    }

    const std::uint32_t slots = 64;
    const std::uint64_t slab_bytes = static_cast<std::uint64_t>(slots) * cache::kPairBytes;
    std::byte* slab = nullptr;
    const auto error = cudaMalloc(reinterpret_cast<void**>(&slab), static_cast<std::size_t>(slab_bytes));
    if (error != cudaSuccess) {
        std::cout << "skip: the device slab cannot be taken: " << cudaGetErrorString(error) << '\n';
        return 77;
    }

    int failures = 0;
    const auto check = [&](bool condition, const std::string& message) {
        if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
    };

    cache::ExpertCacheOptions options;
    options.capacity_bytes = slab_bytes;
    cache::HostPairSource source(path, cache::kPairBytes);
    cache::ExpertCache expert(options, cache::kPairBytes, slab_bytes, source);
    const artifact::Reader reader(path);

    std::vector<std::byte> staging(static_cast<std::size_t>(cache::kPairBytes));
    std::vector<std::byte> readback(static_cast<std::size_t>(cache::kPairBytes));
    std::vector<std::byte*> slot_address(static_cast<std::size_t>(slots), nullptr);

    for (std::uint32_t step = 0; step < slots; ++step) {
        const std::uint32_t layer        = step % cache::kTextLayers;
        const std::uint32_t expert_index = step % cache::kRankExperts;
        const auto slot = expert.load(layer, expert_index);
        check(slot < slots, "load returned slot " + std::to_string(slot) +
            " outside the " + std::to_string(slots) + " device slots");

        const auto address = slab + static_cast<std::size_t>(expert.slot_offset(slot));
        if (slot_address[slot] != nullptr) {
            check(address == slot_address[slot], "slot " + std::to_string(slot) +
                " changed its device address between placements");
        }
        slot_address[slot] = address;
        check(expert.slot_offset(slot) + cache::kPairBytes <= slab_bytes,
              "slot " + std::to_string(slot) + " runs past the end of the slab");

        source.read(layer, expert_index, staging);
        if (cudaMemcpy(address, staging.data(), staging.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("the miss path could not fill slot " + std::to_string(slot));
        }
        execution.primary().synchronize();
        if (cudaMemcpy(readback.data(), address, readback.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("the device slot could not be read back");
        }

        check(hash_of(readback) == hash_of(direct_pair(reader, layer, expert_index)),
              "the device slot holds bytes the artifact does not have for pair (" +
                  std::to_string(layer) + "," + std::to_string(expert_index) + ")");
    }

    check(expert.stats().residency == slots, "the device slab reports residency " +
        std::to_string(expert.stats().residency) + " after " + std::to_string(slots) +
        " placements");
    check(expert.stats().miss_count == slots, "the device slab counted " +
        std::to_string(expert.stats().miss_count) + " misses over " + std::to_string(slots) +
        " placements");

    const auto spanning = find_spanning_pair(reader);
    const auto evicted_slot = expert.load(spanning.first, spanning.second);
    check(expert.stats().evictions == 1, "the load after a full slab evicted " +
        std::to_string(expert.stats().evictions) + " pair; a full-cache miss evicts one");
    check(expert.find(spanning.first, spanning.second) == evicted_slot,
          "the pair that caused the eviction is not in the slot load returned");

    source.read(spanning.first, spanning.second, staging);
    std::byte* evicted_address = slab + static_cast<std::size_t>(expert.slot_offset(evicted_slot));
    if (cudaMemcpy(evicted_address, staging.data(), staging.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        throw std::runtime_error("the spanning pair could not be copied to its device slot");
    }
    execution.primary().synchronize();
    if (cudaMemcpy(readback.data(), evicted_address, readback.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        throw std::runtime_error("the spanning pair could not be read back from the device");
    }
    check(hash_of(readback) == hash_of(direct_pair(reader, spanning.first, spanning.second)),
          "the device slot does not hold the bytes of the pair that spans volumes");

    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        check(slab + static_cast<std::size_t>(expert.slot_offset(slot)) == slot_address[slot],
              "slot " + std::to_string(slot) + " moved to a different device address after an eviction");
    }

    cudaFree(slab);
    if (failures == 0) {
        std::cout << "flash-next expert cache: " << slots << " device slots at fixed addresses, "
                  << "the miss path lands " << cache::kPairBytes << " bytes per slot\n";
    }
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool artifact_mode = argc > 1 && std::string_view(argv[1]) == "artifact";
    const bool device_mode   = argc > 1 && std::string_view(argv[1]) == "device";

    try {
        test_coverage_curve();
        if (!artifact_mode && !device_mode) { return g_failures == 0 ? 0 : 1; }

        const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n";
            return 77;
        }

        test_auto_sized(artifact);
        test_explicit_capacity_over_budget(artifact);
        test_slot_table(artifact);
        test_replacement_is_deterministic(artifact);

        const auto profile = std::filesystem::temp_directory_path() / "ninfer_expert_cache_profile.txt";
        {
            std::ofstream out(profile);
            out << "12 200 512\n3 100 310\n40 7 90\n12 201 12\n";
        }
        test_profile_seeds_the_fill_order(artifact, profile);

        const auto empty_profile =
            std::filesystem::temp_directory_path() / "ninfer_expert_cache_empty_profile.txt";
        {
            std::ofstream out(empty_profile);
        }
        test_empty_profile(artifact, empty_profile);
        test_cross_volume_pair(artifact);

        if (device_mode) { return test_device_slots(artifact); }
        test_host_pair_source(artifact);

        if (g_failures != 0) { return 1; }
        std::cout << "flash-next expert cache: planning layer over " << cache::kRankPairs
                  << " per-rank pairs at " << cache::kPairBytes << " bytes per pair\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
