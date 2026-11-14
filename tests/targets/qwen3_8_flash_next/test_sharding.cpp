#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include "targets/qwen3_8_flash_next/impl/load/sharding.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
namespace sharding = ninfer::targets::qwen3_8_flash_next::detail::sharding;

using Axis    = sharding::ShardAxis;
using Split   = sharding::SplitAxis;
using Role    = roles::Role;
using Mapping = sharding::ShardMapping;

constexpr std::int64_t kBand = 128;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

const char* axis_name(Axis axis) {
    switch (axis) {
    case Axis::Replicated: return "replicated";
    case Axis::Vocab:      return "vocab";
    case Axis::Head:       return "head";
    case Axis::Expert:     return "expert";
    }
    return "unknown";
}

struct ExpectedMapping {
    Role role;
    Axis axis;
    Split split;
    std::int64_t begin;
    std::int64_t extent;
    std::int64_t unit;
};

constexpr ExpectedMapping kExpectedMappings[] = {
    {Role::TokenEmbedding,    Axis::Replicated, Split::Output, 0,      248320,  1},
    {Role::OutputHead,         Axis::Vocab,      Split::Output, 0,     248320,  1},
    {Role::FinalHcNorm,        Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::FinalHcDown,        Axis::Replicated, Split::Output, 0,     320,     1},
    {Role::FinalHcUp,          Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::AttentionHcNorm,    Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::AttentionHcDown,    Axis::Replicated, Split::Output, 0,     320,     1},
    {Role::AttentionHcInject,  Axis::Replicated, Split::Output, 0,     4,       1},
    {Role::AttentionHcUp,      Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::MoeHcNorm,          Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::MoeHcDown,          Axis::Replicated, Split::Output, 0,     320,     1},
    {Role::MoeHcInject,        Axis::Replicated, Split::Output, 0,     4,       1},
    {Role::MoeHcUp,            Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::GdnQuery,           Axis::Head,       Split::Output, 0,     2048,    128},
    {Role::GdnKey,             Axis::Head,       Split::Output, 2048,  2048,    128},
    {Role::GdnValue,           Axis::Head,       Split::Output, 4096,  6144,    128},
    {Role::GdnZ,               Axis::Head,       Split::Output, 10240, 6144,    128},
    {Role::GdnAProjection,     Axis::Replicated, Split::Output, 0,     48,      1},
    {Role::GdnBProjection,     Axis::Replicated, Split::Output, 0,     48,      1},
    {Role::GdnALog,            Axis::Replicated, Split::Output, 0,     48,      1},
    {Role::GdnDtBias,          Axis::Replicated, Split::Output, 0,     48,      1},
    {Role::GdnConvolution,     Axis::Replicated, Split::Output, 0,     4,       1},
    {Role::GdnNorm,            Axis::Replicated, Split::Output, 0,     128,     1},
    {Role::GdnOutput,          Axis::Head,       Split::Input,  0,     6144,    128},
    {Role::QsaQuery,           Axis::Head,       Split::Output, 0,     6144,    256},
    {Role::QsaKey,             Axis::Head,       Split::Output, 0,     512,     256},
    {Role::QsaValue,           Axis::Head,       Split::Output, 512,   512,     256},
    {Role::QsaIndexQuery,      Axis::Replicated, Split::Output, 1024,  512,     128},
    {Role::QsaIndexKey,        Axis::Replicated, Split::Output, 0,     128,     1},
    {Role::QsaOutput,          Axis::Head,       Split::Input,  0,     6144,    256},
    {Role::QsaOutputGate,      Axis::Head,       Split::Output, 6144,  6144,    256},
    {Role::QsaQueryNorm,       Axis::Replicated, Split::Output, 0,     256,     1},
    {Role::QsaIndexQueryNorm,  Axis::Replicated, Split::Output, 0,     128,     1},
    {Role::QsaKeyNorm,         Axis::Replicated, Split::Output, 0,     256,     1},
    {Role::QsaIndexKeyNorm,    Axis::Replicated, Split::Output, 0,     128,     1},
    {Role::MoeRouter,          Axis::Replicated, Split::Output, 0,     512,     1},
    {Role::MoeSharedScore,     Axis::Replicated, Split::Output, 0,     1,       1},
    {Role::MoeSharedGate,      Axis::Replicated, Split::Output, 513,   640,     1},
    {Role::MoeSharedUp,        Axis::Replicated, Split::Output, 1153,  640,     1},
    {Role::MoeSharedDown,      Axis::Replicated, Split::Input,  0,     640,     1},
    {Role::MoeRoutedGate,      Axis::Expert,     Split::Batch,  0,     512,     1},
    {Role::MoeRoutedUp,        Axis::Expert,     Split::Batch,  0,     512,     1},
    {Role::MoeRoutedDown,      Axis::Expert,     Split::Batch,  0,     512,     1},
    {Role::PleTable,           Axis::Replicated, Split::Output, 0,     2500012, 1},
    {Role::PleKey,             Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::PleValue,           Axis::Replicated, Split::Output, 0,     2560,    1},
    {Role::PleNormKey,         Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::PleNormQuery,       Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::PleNormConv,        Axis::Replicated, Split::Output, 0,     10240,   1},
    {Role::PleConvolution,     Axis::Replicated, Split::Output, 0,     4,       1},
};

const ExpectedMapping& expected_of(Role role) {
    for (const auto& entry : kExpectedMappings) {
        if (entry.role == role) { return entry; }
    }
    throw std::runtime_error("the expected table holds no entry for a role");
}

std::string name_of(Role role) { return roles::parameter_name(role, 0); }

std::string ranks_text(const Mapping& mapping) {
    std::string out = "[";
    for (const auto& shard : mapping.ranks) {
        if (out.size() > 1) { out += ","; }
        out += "[" + std::to_string(shard.begin) + "," + std::to_string(shard.end) + ")";
    }
    return out + "]";
}

void test_every_role_has_a_mapping() {
    require(std::size(kExpectedMappings) == roles::kRoleCount,
            "the expected table holds " + std::to_string(std::size(kExpectedMappings)) +
                " roles; the role table holds " + std::to_string(roles::kRoleCount));

    std::set<Role> seen;
    for (const auto& entry : kExpectedMappings) {
        require(seen.insert(entry.role).second, "the expected table repeats a role");
    }
    for (const auto role : roles::all_roles()) {
        require(seen.count(role) == 1, "the expected table holds no entry for " + name_of(role));
    }

    for (const auto& entry : kExpectedMappings) {
        const auto mapping = sharding::shard_mapping(entry.role, 2);
        require(mapping.axis == entry.axis, name_of(entry.role) + " maps to axis " +
            std::string(axis_name(mapping.axis)) + "; it maps to " + axis_name(entry.axis));
        require(mapping.split == entry.split,
                name_of(entry.role) + " splits the wrong stored axis");
        if (entry.axis == Axis::Replicated) {
            require(mapping.ranks.empty(),
                    name_of(entry.role) + " is a full copy but names a range");
            continue;
        }
        require(mapping.ranks.size() == 2, name_of(entry.role) + " names " +
            std::to_string(mapping.ranks.size()) + " ranks at tp=2; it names two");
        require(mapping.ranks[0].begin == entry.begin &&
                    mapping.ranks[1].end == entry.begin + entry.extent,
                name_of(entry.role) + " ranks do not join into the parent span " +
                    ranks_text(mapping));
    }
}

void test_head_roles_partition_by_head_width() {
    std::set<Role> head_roles;
    for (const auto& entry : kExpectedMappings) {
        if (entry.axis == Axis::Head) { head_roles.insert(entry.role); }
    }
    const std::set<Role> expected_head = {
        Role::GdnQuery,   Role::GdnKey,    Role::GdnValue,   Role::GdnZ,      Role::GdnOutput,
        Role::QsaQuery,   Role::QsaKey,    Role::QsaValue,
        Role::QsaOutput,  Role::QsaOutputGate,
    };
    require(head_roles == expected_head, "the head-axis set is not the head-axis set");

    for (const auto& entry : kExpectedMappings) {
        if (entry.axis != Axis::Head) { continue; }
        const auto mapping = sharding::shard_mapping(entry.role, 2);
        require(mapping.ranks[0].end == mapping.ranks[1].begin,
                name_of(entry.role) + " ranks do not join at one boundary");
        for (const auto& shard : mapping.ranks) {
            require(shard.end - shard.begin == entry.extent / 2, name_of(entry.role) +
                " rank covers " + std::to_string(shard.end - shard.begin) + " rows; it covers " +
                std::to_string(entry.extent / 2));
            const auto offset = shard.begin - entry.begin;
            require(offset % entry.unit == 0, name_of(entry.role) + " rank starts at offset " +
                std::to_string(offset) + ", which is not a " + std::to_string(entry.unit) +
                "-row head boundary");
        }
    }
}

void test_expert_roles_partition_by_expert_set() {
    std::set<Role> expert_roles;
    for (const auto& entry : kExpectedMappings) {
        if (entry.axis == Axis::Expert) { expert_roles.insert(entry.role); }
    }
    const std::set<Role> expected_expert = {
        Role::MoeRoutedGate,  Role::MoeRoutedUp,  Role::MoeRoutedDown,
    };
    require(expert_roles == expected_expert, "the expert-axis set is not the expert-axis set");

    for (const auto& entry : kExpectedMappings) {
        if (entry.axis != Axis::Expert) { continue; }
        const auto mapping = sharding::shard_mapping(entry.role, 2);
        require(mapping.ranks[0].begin == 0 && mapping.ranks[1].end == entry.extent,
                name_of(entry.role) + " ranks do not cover the whole stack " + ranks_text(mapping));
        require(mapping.ranks[0].end == mapping.ranks[1].begin,
                name_of(entry.role) + " ranks do not join at one boundary");
        for (const auto& shard : mapping.ranks) {
            require(shard.end - shard.begin == 256, name_of(entry.role) + " rank owns " +
                std::to_string(shard.end - shard.begin) + " matrices; it owns whole experts, 256");
        }
    }
}

void test_split_roles_start_on_band_boundaries() {
    for (const auto& entry : kExpectedMappings) {
        if (entry.axis == Axis::Replicated) { continue; }
        const auto mapping = sharding::shard_mapping(entry.role, 2);
        require(entry.begin % kBand == 0, name_of(entry.role) + " starts at " +
            std::to_string(entry.begin) + ", which is not a " + std::to_string(kBand) +
            "-row band boundary");
        for (const auto& shard : mapping.ranks) {
            require((shard.end - shard.begin) % kBand == 0, name_of(entry.role) + " rank covers " +
                std::to_string(shard.end - shard.begin) + " units, which is not a whole number "
                "of " + std::to_string(kBand) + "-row bands");
        }
    }
}

void test_replicated_set_is_exact() {
    std::set<Role> replicated;
    for (const auto role : roles::all_roles()) {
        if (sharding::shard_mapping(role, 2).axis == Axis::Replicated) { replicated.insert(role); }
    }
    const std::set<Role> expected_replicated = {
        Role::TokenEmbedding,     Role::FinalHcNorm,        Role::FinalHcDown,       Role::FinalHcUp,
        Role::AttentionHcNorm,    Role::AttentionHcDown,   Role::AttentionHcInject,
        Role::AttentionHcUp,      Role::MoeHcNorm,         Role::MoeHcDown,
        Role::MoeHcInject,        Role::MoeHcUp,           Role::GdnNorm,
        Role::GdnAProjection,     Role::GdnBProjection,    Role::GdnALog,
        Role::GdnDtBias,          Role::GdnConvolution,    Role::QsaIndexKey,
        Role::QsaIndexQuery,      Role::QsaIndexKeyNorm,    Role::QsaIndexQueryNorm,
        Role::QsaKeyNorm,
        Role::QsaQueryNorm,       Role::MoeRouter,         Role::MoeSharedScore,
        Role::MoeSharedGate,      Role::MoeSharedUp,       Role::MoeSharedDown,
        Role::PleTable,           Role::PleKey,            Role::PleValue,
        Role::PleNormKey,         Role::PleNormQuery,      Role::PleNormConv,
        Role::PleConvolution,
    };
    require(replicated == expected_replicated, "the replicated set is not the replicated set");
    require(sharding::shard_mapping(Role::OutputHead, 2).axis == Axis::Vocab,
            "the output head is not split on the vocab axis");
    require(sharding::shard_mapping(Role::MoeRoutedGate, 2).axis == Axis::Expert,
            "the routed experts are not split by the expert set");
}

void test_tp1_yields_the_whole_range() {
    for (const auto& entry : kExpectedMappings) {
        const auto mapping = sharding::shard_mapping(entry.role, 1);
        require(mapping.axis == entry.axis, name_of(entry.role) + " changes axis at tp=1");
        if (entry.axis == Axis::Replicated) {
            require(mapping.ranks.empty(), name_of(entry.role) +
                " is a full copy but names a range at tp=1");
            continue;
        }
        require(mapping.ranks.size() == 1, name_of(entry.role) + " names " +
            std::to_string(mapping.ranks.size()) + " ranks at tp=1; it names one");
        require(mapping.ranks[0].begin == entry.begin &&
                    mapping.ranks[0].end == entry.begin + entry.extent,
                name_of(entry.role) + " does not return the whole range at tp=1: " +
                    ranks_text(mapping));
    }
}

void test_illegal_boundary_names_the_role() {
    const std::vector<Role> indivisible = {
        Role::GdnKey,         Role::QsaKey,
        Role::MoeRoutedGate,  Role::MoeRoutedDown, Role::OutputHead,
    };
    for (const auto role : indivisible) {
        bool threw = false;
        try {
            const auto mapping = sharding::shard_mapping(role, 3);
            require(mapping.ranks.size() == 3, "tp=3 is accepted for " + name_of(role));
        } catch (const std::invalid_argument& error) {
            threw = true;
            const std::string message = error.what();
            require(message.find(name_of(role)) != std::string::npos,
                    "the rejection for " + name_of(role) + " does not name the role: " + message);
        }
        require(threw, "tp=3 is accepted for " + name_of(role));
    }

    bool threw = false;
    try {
        const auto mapping = sharding::shard_mapping(Role::MoeRoutedGate, 8);
        require(mapping.ranks.size() == 8, "tp=8 is accepted for " + name_of(Role::MoeRoutedGate));
    } catch (const std::invalid_argument& error) {
        threw = true;
        const std::string message = error.what();
        require(message.find(name_of(Role::MoeRoutedGate)) != std::string::npos &&
                    message.find("band") != std::string::npos,
                "the rejection for " + name_of(Role::MoeRoutedGate) +
                    " does not name the role and its band: " + message);
    }
    require(threw, "tp=8 is accepted for " + name_of(Role::MoeRoutedGate));

    threw = false;
    try {
        const auto mapping = sharding::shard_mapping(Role::GdnKey, 0);
        require(!mapping.ranks.empty(), "tp=0 is accepted for " + name_of(Role::GdnKey));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "tp=0 is accepted for " + name_of(Role::GdnKey));
}

} // namespace

int main() {
    try {
        test_every_role_has_a_mapping();
        test_head_roles_partition_by_head_width();
        test_expert_roles_partition_by_expert_set();
        test_split_roles_start_on_band_boundaries();
        test_replicated_set_is_exact();
        test_tp1_yields_the_whole_range();
        test_illegal_boundary_names_the_role();
        std::cout << "flash-next sharding: all " << roles::kRoleCount << " roles map at tp=2 over "
                  << "the head, expert and vocab axes, with the replicated embedding and the replicated "
                  << "set\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
