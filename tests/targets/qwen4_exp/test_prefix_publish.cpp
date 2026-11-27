// Host-only contract test for the qwen4_exp prefix publication side: committed 64-token blocks
// and one page-aligned state image become a probeable pinned-host manifest. No model, no artifact,
// no Engine is constructed; the expected frontiers and hashes are computed independently below.

#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/shared_host_tier.h"
#include "targets/qwen4_exp/impl/runtime/prefix_cache_impl.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace pc = ninfer::prefix_cache;
namespace q4 = ninfer::targets::qwen4_exp::detail;
using ninfer::SharedHostTier;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

struct RecordingSink final : pc::PrefixPageSink {
    int block_releases  = 0;
    int tail_releases   = 0;
    int slot_releases   = 0;
    void release_block_pages(const pc::PoolPages&) noexcept override { ++block_releases; }
    void release_tail_pages(const pc::PoolPages&) noexcept override { ++tail_releases; }
    void release_snapshot_slot(pc::SlotId) noexcept override { ++slot_releases; }
};

std::vector<ninfer::TokenId> prompt(std::size_t blocks) {
    std::vector<ninfer::TokenId> tokens(blocks * pc::kBlockTokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<ninfer::TokenId>(17 + (i * 37) % 9000);
    }
    return tokens;
}

SharedHostTier::Config tier_config() {
    SharedHostTier::Config config;
    config.host_bytes          = 64 * 1024;
    config.block_slab_bytes    = 1024;
    config.snapshot_slab_bytes = 2048;
    config.block_slabs         = 8;
    config.snapshot_slabs      = 4;
    config.block_tokens        = pc::kBlockTokens;
    config.tp                  = 1;
    return config;
}

void fill(pc::HostPageAllocator& allocator, pc::HostPageId page, std::uint8_t byte) {
    std::memset(allocator.data(page), byte, allocator.bytes(page));
}

} // namespace

int main() {
    auto tokens = prompt(2);
    const auto hashes = pc::chain_block_hashes(tokens);
    check(hashes.size() == 2, "two full prompt blocks produce two chained hashes");
    // Independent recomputation of the chained hash contract (parent chaining, exact tail).
    const auto first = pc::block_hash(pc::kBlockHashSeed,
                                      std::span<const ninfer::TokenId>(tokens.data(), 64), 0);
    const auto second = pc::block_hash(first,
                                       std::span<const ninfer::TokenId>(tokens.data() + 64, 64), 0);
    check(hashes[0] == first && hashes[1] == second, "hashes chain from the documented seed");

    auto tier = SharedHostTier::create(tier_config());
    const std::uint64_t owner = tier->allocate_owner();
    RecordingSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 3}, &sink);
    index.set_host_allocator(tier.get());
    index.set_shared_host_tier(tier.get());

    const auto slot = index.reserve_snapshot_slot(128);
    check(slot != pc::kInvalidSlot, "a device slot can be reserved before publication");
    const auto published =
        q4::commit_prefix_blocks(index, hashes, tokens, slot);
    check(published.block_nodes.size() == 2, "both prompt blocks are committed");
    check(published.anchor == published.block_nodes.back(), "the snapshot anchors on the last block");
    check(published.frontier == 128, "the page-aligned frontier is 128 tokens");
    check(published.snapshot != pc::kInvalidSnapshot, "an endpoint snapshot is published locally");

    for (const pc::NodeId node : published.block_nodes) {
        const auto write = index.begin_host_write(node);
        check(write.page != pc::kInvalidHostPage, "each block obtains a pinned host slab");
        if (write.page == pc::kInvalidHostPage) { continue; }
        fill(*tier, write.page, 0x5A);
        index.finish_host_write(node, true);
    }
    const auto image = index.begin_snapshot_write(published.snapshot);
    check(image.page != pc::kInvalidHostPage, "the state image obtains a pinned host slab");
    if (image.page != pc::kInvalidHostPage) {
        fill(*tier, image.page, 0x3C);
        index.finish_snapshot_write(published.snapshot, true);
    }
    const std::uint32_t generation = index.snapshot_generation(published.snapshot);
    const std::uint64_t manifest = index.publish_snapshot_manifest(
        published.snapshot, owner, 1, generation);
    check(manifest != 0, "a fully host-backed snapshot becomes a shared manifest");

    const auto probe = q4::probe_prefix(index, tier.get(), 1, hashes);
    check(probe.block_tokens == 64, "block granularity is 64");
    check(probe.device_frontier == 128, "local device frontier matches the committed chain");
    check(probe.manifest_frontier == 128, "host manifest frontier matches the committed chain");
    check(probe.manifest_id == manifest, "probe names the published manifest");

    auto foreign = hashes;
    foreign[1] ^= 0xABCDEF0123456789ULL;
    const auto miss = q4::probe_prefix(index, tier.get(), 1, foreign);
    check(miss.manifest_id == 0 && miss.manifest_frontier == 0,
          "a divergent block hash has no host manifest");
    check(miss.device_frontier == 64, "local device probe stops at the divergent block");

    // host-cache-mib 0: no tier is constructed, so no manifest can exist even after the same
    // blocks are committed to the local index.
    pc::PrefixIndex device_only(pc::PrefixIndexOptions{.device_snapshot_slots = 3}, &sink);
    const auto cold = q4::commit_prefix_blocks(device_only, hashes, tokens, pc::kInvalidSlot);
    check(cold.frontier == 128 && cold.snapshot != pc::kInvalidSnapshot,
          "device-only publication still records the chain");
    const auto cold_probe = q4::probe_prefix(device_only, nullptr, 1, hashes);
    check(cold_probe.manifest_id == 0 && cold_probe.manifest_frontier == 0,
          "without a host tier no manifest is published or matched");
    check(cold_probe.device_frontier == 128, "device-only probe uses local residency");

    // Publishing the same frontier twice must not strand the second reserved slot: add_snapshot
    // returns the existing snapshot without consuming the reservation.
    pc::PrefixIndex repeat_index(pc::PrefixIndexOptions{.device_snapshot_slots = 3}, &sink);
    repeat_index.set_host_allocator(tier.get());
    repeat_index.set_shared_host_tier(tier.get());
    const std::uint32_t free_before = repeat_index.device_snapshot_slots_free();
    const auto first_slot = repeat_index.reserve_snapshot_slot(128);
    const auto first_pub = q4::commit_prefix_blocks(repeat_index, hashes, tokens, first_slot);
    check(first_pub.snapshot != pc::kInvalidSnapshot, "the first publication creates a snapshot");
    check(repeat_index.device_snapshot_slots_free() == free_before - 1,
          "the first publication consumes exactly one device slot");
    const auto second_slot = repeat_index.reserve_snapshot_slot(128);
    const auto second_pub = q4::commit_prefix_blocks(repeat_index, hashes, tokens, second_slot);
    check(second_pub.snapshot == first_pub.snapshot, "a duplicate frontier resolves to the same snapshot");
    check(repeat_index.snapshot(second_pub.snapshot)->device_slot == first_slot,
          "the duplicate publication keeps the original slot owner");
    check(repeat_index.device_snapshot_slots_free() == free_before - 1,
          "a duplicate frontier returns its reserved slot instead of leaking it");

    if (failures != 0) {
        std::fprintf(stderr, "qwen4_exp prefix publish: %d failure(s)\n", failures);
        return 1;
    }
    std::cout << "OK qwen4_exp prefix publish/probe\n";
    return 0;
}
