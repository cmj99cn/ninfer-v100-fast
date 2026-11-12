#pragma once

// Shared pinned-host cache tier for multi-instance prefix reuse.
//
// A SharedHostTier owns the slab pool and the manifest records that describe which pages make up
// a reusable prompt prefix. Publishers pin every page while the manifest exists; consumers must
// retain a manifest before using the pages returned by match() and release it afterwards.
// Manifest refcounts are independent of the slab pin refcount:
//   - publish pins each page once for the lifetime of the manifest;
//   - the last release returns the pages to the pool in a deferred state;
//   - eviction (or erase_manifest) drops the final pin and frees the pages.
// All manifest-map and pool mutations are serialized by one mutex.

#include "runtime/prefix_cache/host_slab_pool.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer {

// Host-only unit tests use this friend accessor to inspect the raw slab pool. Production code
// must go through the SharedHostTier HostPageAllocator overrides, which serialize the pool.
struct SharedHostTierTestAccess;

class SharedHostTier : public prefix_cache::HostPageAllocator,
                       public std::enable_shared_from_this<SharedHostTier> {
public:
    struct Config {
        std::size_t host_bytes = 0;
        std::size_t block_slab_bytes = 0;
        std::size_t snapshot_slab_bytes = 0;
        std::size_t block_slabs = 0;
        std::size_t snapshot_slabs = 0;
        std::uint32_t block_tokens = 64;
        std::uint32_t tp = 1;
        std::chrono::milliseconds manifest_ttl{std::chrono::minutes(30)};
    };
    struct ManifestView {
        std::uint64_t id = 0;
        std::uint32_t tp = 0;
        std::uint32_t block_tokens = 64;
        std::uint32_t frontier = 0;
        std::vector<std::uint64_t> hashes;
        std::vector<prefix_cache::HostPageId> block_pages;
        prefix_cache::HostPageId state_page = prefix_cache::kInvalidHostPage;
    };
    struct PublishData {
        std::uint32_t tp = 0;
        std::uint32_t frontier = 0;
        std::vector<std::uint64_t> hashes;
        std::vector<prefix_cache::HostPageId> block_pages;
        prefix_cache::HostPageId state_page = prefix_cache::kInvalidHostPage;
    };

    static std::shared_ptr<SharedHostTier> create(Config config);

    ~SharedHostTier() override = default;
    SharedHostTier(const SharedHostTier&) = delete;
    SharedHostTier& operator=(const SharedHostTier&) = delete;

    std::uint64_t allocate_owner();
    std::uint64_t publish(std::uint64_t owner, PublishData data);
    bool retain_manifest(std::uint64_t manifest_id);
    bool release_manifest(std::uint64_t manifest_id);
    bool erase_manifest(std::uint64_t manifest_id);
    void retire_owner(std::uint64_t owner);
    std::optional<ManifestView> match(std::uint32_t tp, std::span<const std::uint64_t> hashes,
                                      std::uint32_t max_frontier);
    std::size_t manifest_count() const;
    std::size_t evict_one();
    bool enabled() const noexcept;
    std::uint32_t tp() const noexcept;
    std::size_t block_slab_bytes() const noexcept;
    std::size_t snapshot_slab_bytes() const noexcept;

    // HostPageAllocator overrides: serialize access to the non-thread-safe slab pool.
    [[nodiscard]] prefix_cache::SlabAllocation allocate_block() override;
    [[nodiscard]] prefix_cache::SlabAllocation allocate_snapshot() override;
    void release(prefix_cache::HostPageId page) noexcept override;
    void pin(prefix_cache::HostPageId page) noexcept override;
    void unpin(prefix_cache::HostPageId page) noexcept override;
    void mark_dead(prefix_cache::HostPageId page) noexcept override;
    void mark_alive(prefix_cache::HostPageId page) noexcept override;
    void supersede(prefix_cache::HostPageId page) noexcept override;
    void clear_superseded(prefix_cache::HostPageId page) noexcept override;
    void hit(prefix_cache::HostPageId page, std::uint64_t tick) noexcept override;
    [[nodiscard]] const prefix_cache::HostSlab* slab(
        prefix_cache::HostPageId page) const noexcept override;
    [[nodiscard]] void* data(prefix_cache::HostPageId page) const noexcept override;
    [[nodiscard]] std::size_t bytes(prefix_cache::HostPageId page) const noexcept override;
    [[nodiscard]] std::size_t used_bytes() const noexcept override;
    [[nodiscard]] std::size_t pinned_bytes() const noexcept override;
    [[nodiscard]] std::size_t block_slab_count() const noexcept override;
    [[nodiscard]] std::size_t snapshot_slab_count() const noexcept override;
    [[nodiscard]] std::uint64_t block_alloc_failures() const noexcept override;
    [[nodiscard]] std::uint64_t snapshot_alloc_failures() const noexcept override;

private:
    struct Entry {
        std::uint64_t id = 0;
        std::uint64_t owner = 0;
        std::uint32_t tp = 0;
        std::uint32_t block_tokens = 64;
        std::uint32_t frontier = 0;
        std::vector<std::uint64_t> hashes;
        std::vector<prefix_cache::HostPageId> block_pages;
        prefix_cache::HostPageId state_page = prefix_cache::kInvalidHostPage;
        std::uint32_t refs = 0;
        std::chrono::steady_clock::time_point expires_at;
        std::uint64_t freq = 0;
        std::uint64_t last_hit = 0;
    };

    explicit SharedHostTier(Config config);

    // Raw pool access is host-test-only; see SharedHostTierTestAccess above.
    friend struct SharedHostTierTestAccess;
    prefix_cache::HostSlabPool& pool() noexcept { return pool_; }

    using ManifestMap = std::unordered_map<std::uint64_t, Entry>;
    ManifestMap::iterator erase_locked(ManifestMap::iterator it) noexcept;
    void release_pages_locked(const Entry& entry) noexcept;
    std::size_t evict_one_locked() noexcept;

    Config config_;
    prefix_cache::HostSlabPool pool_;
    mutable std::mutex mutex_;
    ManifestMap manifests_;
    std::uint64_t next_manifest_id_ = 1;
    std::uint64_t next_owner_ = 1;
    std::uint64_t access_tick_ = 0;
    std::uint64_t block_alloc_failures_ = 0;
    std::uint64_t snapshot_alloc_failures_ = 0;
};

} // namespace ninfer
