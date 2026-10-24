#include "runtime/prefix_cache/prefix_index.h"

#include "runtime/prefix_cache/shared_host_tier.h"

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::prefix_cache {
namespace {

static_assert(kBlockTokens == kPagedKVPageSize,
              "prefix-cache block must equal the paged KV page size");

[[nodiscard]] std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

} // namespace

std::uint64_t block_hash(std::uint64_t parent_hash, std::span<const TokenId> tokens,
                         std::uint64_t extra) noexcept {
    std::uint64_t hash = mix(parent_hash ^ 0xA076'1D64'CF15'1E25ULL);
    for (TokenId token : tokens) {
        hash = mix(hash ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(token)) *
                           0xE703'7B9D'2C1F'1235ULL));
    }
    return mix(hash ^ extra);
}

std::vector<std::uint64_t> chain_block_hashes(std::span<const TokenId> tokens,
                                              std::span<const std::uint64_t> extras) {
    const std::size_t blocks = tokens.size() / static_cast<std::size_t>(kBlockTokens);
    if (!extras.empty() && extras.size() != blocks) {
        throw std::invalid_argument("block extras must supply one key per full block");
    }
    std::vector<std::uint64_t> hashes(blocks);
    std::uint64_t parent = kBlockHashSeed;
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint64_t extra = extras.empty() ? 0U : extras[b];
        parent = block_hash(parent,
                            tokens.subspan(b * static_cast<std::size_t>(kBlockTokens),
                                           static_cast<std::size_t>(kBlockTokens)),
                            extra);
        hashes[b] = parent;
    }
    return hashes;
}

PrefixIndex::PrefixIndex(PrefixIndexOptions options, PrefixPageSink* sink)
    : options_(options), sink_(sink), tick_(options.now_tick) {
    nodes_.reserve(256);
    snapshots_.reserve(64);
    slot_owner_.assign(options_.device_snapshot_slots, kInvalidSnapshot);
    free_slots_.reserve(options_.device_snapshot_slots);
    for (std::int32_t slot = static_cast<std::int32_t>(options_.device_snapshot_slots) - 1;
         slot >= 0; --slot) {
        free_slots_.push_back(slot);
    }

    root_ = allocate_node();
    BlockNode& root = nodes_[root_ - 1U];
    root.parent       = kInvalidNode;
    root.depth        = -1;
    root.lookup_hash  = kBlockHashSeed;
    root.device_state = Residency::Absent;
}

PrefixIndex::~PrefixIndex() {
    if (shared_tier_ == nullptr) { return; }
    for (auto& [manifest_id, record] : foreign_records_) {
        (void)manifest_id;
        for (NodeId id : record.nodes) {
            if (!valid(id)) { continue; }
            BlockNode& node = nodes_[id - 1U];
            if (node.device_state == Residency::Resident && sink_ != nullptr) {
                sink_->release_block_pages(node.device_pages);
            }
            node.host_page          = kInvalidHostPage;
            node.host_state         = Residency::Absent;
            node.host_shared        = false;
            node.shared_manifests.clear();
            free_node(id);
        }
        if (record.snapshot != kInvalidSnapshot && record.snapshot <= snapshots_.size() &&
            snapshots_[record.snapshot - 1U].live) {
            Snapshot& s = snapshots_[record.snapshot - 1U];
            if (s.device_slot != kInvalidSlot && sink_ != nullptr) {
                sink_->release_snapshot_slot(s.device_slot);
            }
            s.host_page          = kInvalidHostPage;
            s.host_state         = Residency::Absent;
            s.host_shared        = false;
            s.shared_manifest_id = 0;
            s.live               = false;
        }
        shared_tier_->release_manifest(record.manifest_id);
    }
    foreign_records_.clear();
    for (Snapshot& record : snapshots_) {
        if (record.live && record.host_shared && record.shared_manifest_id != 0) {
            shared_tier_->release_manifest(record.shared_manifest_id);
            record.host_shared          = false;
            record.shared_manifest_id   = 0;
            record.host_page            = kInvalidHostPage;
            record.host_state           = Residency::Absent;
        }
    }
}

const BlockNode* PrefixIndex::node(NodeId id) const noexcept {
    if (!valid(id)) { return nullptr; }
    return &nodes_[id - 1U];
}

const Snapshot* PrefixIndex::snapshot(SnapshotId id) const noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return nullptr; }
    const Snapshot& record = snapshots_[id - 1U];
    return record.live ? &record : nullptr;
}

SnapshotId PrefixIndex::snapshot_id_at(std::size_t index) const noexcept {
    if (index >= snapshots_.size()) { return kInvalidSnapshot; }
    const Snapshot& record = snapshots_[index];
    return record.live ? static_cast<SnapshotId>(index + 1U) : kInvalidSnapshot;
}

NodeId PrefixIndex::allocate_node() {
    NodeId id;
    if (!free_nodes_.empty()) {
        id = free_nodes_.back();
        free_nodes_.pop_back();
        BlockNode& reused = nodes_[id - 1U];
        const std::uint32_t generation = reused.generation + 1U;
        reused = BlockNode{};
        reused.generation = generation;
    } else {
        nodes_.emplace_back();
        id = static_cast<NodeId>(nodes_.size());
    }
    ++live_nodes_;
    return id;
}

void PrefixIndex::free_node(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    record.live       = false;
    record.children.clear();
    record.snapshots.clear();
    record.device_state = Residency::Absent;
    free_nodes_.push_back(id);
    --live_nodes_;
}

void PrefixIndex::link_lru_tail(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    // A foreign node is not part of the local eviction LRU; its manifest-owned host page is
    // reclaimed only through the tier, and its device pages only through `commit_foreign` or
    // `release_foreign`.
    if (record.foreign) { return; }
    // A node is evictable iff it is device-resident and held by no one: no active sequence
    // (active_refs) and no in-flight transfer (restore_pins). Linking is conditional so every
    // state-change site can simply call this after mutating the counters.
    if (record.lru_linked || record.device_state != Residency::Resident ||
        record.active_refs != 0U || record.restore_pins != 0U) {
        return;
    }
    record.lru_prev   = lru_tail_;
    record.lru_next   = kInvalidNode;
    if (lru_tail_ != kInvalidNode) {
        nodes_[lru_tail_ - 1U].lru_next = id;
    } else {
        lru_head_ = id;
    }
    lru_tail_          = id;
    record.lru_linked  = true;
    ++evictable_pages_;
}

void PrefixIndex::unlink_lru(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    if (!record.lru_linked) { return; }
    if (record.lru_prev != kInvalidNode) {
        nodes_[record.lru_prev - 1U].lru_next = record.lru_next;
    } else {
        lru_head_ = record.lru_next;
    }
    if (record.lru_next != kInvalidNode) {
        nodes_[record.lru_next - 1U].lru_prev = record.lru_prev;
    } else {
        lru_tail_ = record.lru_prev;
    }
    record.lru_prev   = kInvalidNode;
    record.lru_next   = kInvalidNode;
    record.lru_linked = false;
    if (evictable_pages_ > 0U) { --evictable_pages_; }
}

void PrefixIndex::pin_node(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.active_refs == 0U) { unlink_lru(id); }
    ++record.active_refs;
}

void PrefixIndex::unpin_node(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.active_refs == 0U) { return; }
    --record.active_refs;
    link_lru_tail(id);
}

void PrefixIndex::pin_restore(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.restore_pins == 0U) { unlink_lru(id); }
    ++record.restore_pins;
}

void PrefixIndex::unpin_restore(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.restore_pins == 0U) { return; }
    --record.restore_pins;
    link_lru_tail(id);
}

void PrefixIndex::pin_snapshot(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (record.live) { ++record.restore_pins; }
}

void PrefixIndex::unpin_snapshot(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (record.live && record.restore_pins != 0U) { --record.restore_pins; }
}

void PrefixIndex::inc_live_below(NodeId node) noexcept {
    for (NodeId cur = node; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
        BlockNode& record = nodes_[cur - 1U];
        if (record.live_snapshots_below == 0 && host_allocator_ != nullptr &&
            record.host_page != kInvalidHostPage) {
            host_allocator_->mark_alive(record.host_page);
        }
        ++record.live_snapshots_below;
    }
}

void PrefixIndex::dec_live_below(NodeId node, std::uint32_t count) noexcept {
    for (NodeId cur = node; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
        BlockNode& record = nodes_[cur - 1U];
        record.live_snapshots_below =
            record.live_snapshots_below >= count ? record.live_snapshots_below - count : 0U;
        // Dead-KV: a node no longer reachable from any snapshot is the lowest-priority host-resident
        // data. Marking it dead only adjusts eviction priority (the pool evicts dead slabs first);
        // the actual cleanup happens lazily when its slab is evicted (reconcile_host_eviction) or
        // its device copy is evicted (evict_device_lru). No eager structural change here, which keeps
        // this callable from inside snapshot deletion without reentrancy.
        if (record.live_snapshots_below == 0 && host_allocator_ != nullptr &&
            record.host_page != kInvalidHostPage) {
            host_allocator_->mark_dead(record.host_page);
        }
    }
}

void PrefixIndex::reconcile_host_eviction(HostPageId page) noexcept {
    if (page == kInvalidHostPage) { return; }
    // The pool already freed `page`; only the owner's bookkeeping must be cleared.
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        BlockNode& n = nodes_[i];
        if (!n.live || n.host_page != page) { continue; }
        n.host_page  = kInvalidHostPage;
        n.host_state = Residency::Absent;
        if (n.device_state != Residency::Resident && n.device_state != Residency::Filling &&
            n.active_refs == 0 && n.restore_pins == 0) {
            erase_subtree(static_cast<NodeId>(i + 1U));
        }
        return;
    }
    for (std::size_t i = 0; i < snapshots_.size(); ++i) {
        Snapshot& s = snapshots_[i];
        if (!s.live || s.host_page != page) { continue; }
        s.host_page  = kInvalidHostPage;
        s.host_state = Residency::Absent;
        if (s.device_state != Residency::Resident && s.device_state != Residency::Filling &&
            s.restore_pins == 0) {
            delete_snapshot(static_cast<SnapshotId>(i + 1U));
        }
        return;
    }
}

ChildMatch PrefixIndex::find_child(NodeId parent, std::uint64_t lookup_hash,
                                   std::span<const TokenId> tokens, std::uint64_t extra) const {
    const BlockNode* parent_node = node(parent);
    if (parent_node == nullptr || tokens.size() != kBlockTokens) { return {}; }
    ChildMatch match;
    for (NodeId child_id : parent_node->children) {
        const BlockNode& child = nodes_[child_id - 1U];
        if (!child.live || child.foreign) { continue; }
        if (child.lookup_hash != lookup_hash) { continue; }
        if (child.extra == extra &&
            std::equal(tokens.begin(), tokens.end(), child.tokens.begin())) {
            match.node = child_id;
            return match;
        }
        match.hash_collided = true;
    }
    return match;
}

std::uint32_t PrefixIndex::probe_device(std::span<const std::uint64_t> hashes) const {
    std::uint32_t frontier = 0;
    NodeId current = root_;
    for (const std::uint64_t hash : hashes) {
        const BlockNode* parent_node = node(current);
        if (parent_node == nullptr) { break; }
        NodeId next = kInvalidNode;
        for (const NodeId child_id : parent_node->children) {
            const BlockNode& child = nodes_[child_id - 1U];
            if (!child.live || child.foreign) { continue; }
            if (child.lookup_hash == hash) {
                next = child_id;
                break;
            }
        }
        if (next == kInvalidNode) { break; }
        if (nodes_[next - 1U].device_state != Residency::Resident) { break; }
        frontier += static_cast<std::uint32_t>(kBlockTokens);
        current = next;
    }
    return frontier;
}

PrefixMatch PrefixIndex::match(std::span<const TokenId> prompt_tokens,
                               std::span<const std::uint64_t> block_hashes,
                               std::span<const std::uint64_t> block_extras) const {
    PrefixMatch result;
    result.path.push_back(root_);

    const std::size_t token_count = prompt_tokens.size();
    std::size_t eligible = 0;
    if (token_count >= 1U) { eligible = (token_count - 1U) / kBlockTokens; }

    NodeId current = root_;
    std::size_t block = 0;
    for (; block < eligible && block < block_hashes.size(); ++block) {
        const std::span<const TokenId> tokens =
            prompt_tokens.subspan(block * kBlockTokens, kBlockTokens);
        const std::uint64_t extra = block < block_extras.size() ? block_extras[block] : 0U;
        const ChildMatch child = find_child(current, block_hashes[block], tokens, extra);
        if (child.node == kInvalidNode) { break; }
        current = child.node;
        result.path.push_back(current);
    }
    result.matched_blocks = result.path.size() - 1U;

    for (NodeId node_id : result.path) {
        const BlockNode& record = nodes_[node_id - 1U];
        for (SnapshotId snapshot_id : record.snapshots) {
            const Snapshot& candidate = snapshots_[snapshot_id - 1U];
            if (!candidate.live || candidate.superseded) { continue; }
            if (candidate.frontier > token_count - 1U) { continue; }

            const BlockNode& anchor = nodes_[candidate.anchor - 1U];
            const std::uint32_t tail_begin =
                static_cast<std::uint32_t>(kBlockTokens) *
                static_cast<std::uint32_t>(anchor.depth + 1);
            if (candidate.frontier != tail_begin + candidate.tail_len) { continue; }
            if (candidate.tail_len != 0U &&
                !std::equal(candidate.tail_tokens.begin(),
                            candidate.tail_tokens.begin() + candidate.tail_len,
                            prompt_tokens.begin() + tail_begin)) {
                continue;
            }

            // A snapshot is valid iff every node on its anchor path, its tail bundle and its image
            // are resident. Path residency is checked against the matched path prefix.
            bool device_resident = candidate.device_state == Residency::Resident &&
                                    (candidate.tail_len == 0U ||
                                     candidate.tail_state == Residency::Resident);
            bool host_resident = candidate.host_state == Residency::Host &&
                                  (candidate.tail_len == 0U ||
                                   candidate.tail_state == Residency::Host);
            if (device_resident) {
                const std::size_t anchor_index = static_cast<std::size_t>(anchor.depth + 1);
                // path[0] is the root, which owns no KV pages; residency of a prefix is carried by
                // the deepest matching node.
                for (std::size_t i = 1; i <= anchor_index && i < result.path.size(); ++i) {
                    if (nodes_[result.path[i] - 1U].device_state != Residency::Resident) {
                        device_resident = false;
                        break;
                    }
                }
            }
            if (host_resident) {
                // Every node on the path must have data available somewhere (device or host) so it
                // can be made device-resident for the restore.
                const std::size_t anchor_index = static_cast<std::size_t>(anchor.depth + 1);
                for (std::size_t i = 1; i <= anchor_index && i < result.path.size(); ++i) {
                    const BlockNode& n = nodes_[result.path[i] - 1U];
                    if (n.device_state != Residency::Resident && n.host_state != Residency::Host) {
                        host_resident = false;
                        break;
                    }
                }
            }

            result.candidates.push_back(SnapshotCandidate{snapshot_id, candidate.frontier,
                                                          device_resident, host_resident});
            if (device_resident && candidate.frontier >= result.best_device_frontier) {
                result.best_device          = snapshot_id;
                result.best_device_frontier = candidate.frontier;
            }
            if (host_resident && candidate.frontier >= result.best_host_frontier) {
                result.best_host          = snapshot_id;
                result.best_host_frontier = candidate.frontier;
            }
        }
    }
    return result;
}

NodeId PrefixIndex::commit_block(const BlockCommit& commit, bool& adopted) {
    adopted = false;
    if (!valid(commit.parent) || commit.tokens.size() != kBlockTokens) { return kInvalidNode; }

    const ChildMatch existing =
        find_child(commit.parent, commit.lookup_hash, commit.tokens, commit.extra);
    if (existing.node != kInvalidNode) {
        BlockNode& record = nodes_[existing.node - 1U];
        // Adopt the offered pages only when no transfer is in flight on this node: a concurrent
        // restore already owns the destination pages and must not be retargeted underneath it.
        if (record.device_state != Residency::Resident && record.restore_pins == 0U) {
            record.device_pages  = commit.pages;
            record.device_state  = Residency::Resident;
            link_lru_tail(existing.node);
            adopted = true;
        }
        record.last_access_tick = ++tick_;
        return existing.node;
    }

    const NodeId id = allocate_node();
    BlockNode& record = nodes_[id - 1U];
    record.parent      = commit.parent;
    record.depth       = nodes_[commit.parent - 1U].depth + 1;
    record.lookup_hash = commit.lookup_hash;
    record.extra       = commit.extra;
    std::copy(commit.tokens.begin(), commit.tokens.end(), record.tokens.begin());
    record.device_pages     = commit.pages;
    record.device_state     = Residency::Resident;
    record.last_access_tick = ++tick_;
    nodes_[commit.parent - 1U].children.push_back(id);
    link_lru_tail(id);
    adopted = true;
    return id;
}

SlotId PrefixIndex::acquire_slot(std::uint64_t priority) {
    if (!free_slots_.empty()) {
        const SlotId slot = free_slots_.back();
        free_slots_.pop_back();
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }

    // Oldest superseded snapshot's slot first.
    SnapshotId superseded_victim = kInvalidSnapshot;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& record = snapshots_[index];
        if (!record.live || !record.superseded || record.restore_pins != 0U ||
            record.device_slot == kInvalidSlot) {
            continue;
        }
        if (record.superseded_at < oldest) {
            oldest           = record.superseded_at;
            superseded_victim = static_cast<SnapshotId>(index + 1U);
        }
    }
    if (superseded_victim != kInvalidSnapshot) {
        // A superseded snapshot that still has a host copy is demoted (kept alive as
        // host-resident) rather than deleted, so its image survives the device-slot eviction.
        const SlotId slot = snapshots_[superseded_victim - 1U].host_state == Residency::Host
                                ? demote_snapshot_device(superseded_victim)
                                : detach_snapshot(superseded_victim);
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }

    // Device-only snapshot with the lowest GDSF priority, if the new one is at least as valuable.
    SnapshotId lowest_victim = kInvalidSnapshot;
    std::uint64_t lowest     = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& record = snapshots_[index];
        if (!record.live || record.restore_pins != 0U || record.device_slot == kInvalidSlot) {
            continue;
        }
        if (record.gdsf_h < lowest) {
            lowest        = record.gdsf_h;
            lowest_victim = static_cast<SnapshotId>(index + 1U);
        }
    }
    if (lowest_victim != kInvalidSnapshot && lowest <= priority) {
        const SlotId slot = snapshots_[lowest_victim - 1U].host_state == Residency::Host
                                ? demote_snapshot_device(lowest_victim)
                                : detach_snapshot(lowest_victim);
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }
    return kInvalidSlot;
}

SnapshotId PrefixIndex::add_snapshot(const SnapshotRequest& request) {
    if (!valid(request.anchor)) {
        return kInvalidSnapshot;
    }
    const BlockNode& anchor = nodes_[request.anchor - 1U];
    const std::uint32_t tail_begin =
        static_cast<std::uint32_t>(kBlockTokens) * static_cast<std::uint32_t>(anchor.depth + 1);
    if (request.frontier < tail_begin) {
        return kInvalidSnapshot;
    }
    const std::uint32_t tail_len = request.frontier - tail_begin;
    if (tail_len >= kBlockTokens || request.tail_tokens.size() != tail_len) {
        // tail_len == 0 is a frontier exactly on a block boundary (a tap snapshot).
        return kInvalidSnapshot;
    }

    // At most one snapshot per (anchor, exact tail tokens).
    for (SnapshotId existing : anchor.snapshots) {
        const Snapshot& record = snapshots_[existing - 1U];
        if (!record.live) { continue; }
        if (record.frontier == request.frontier &&
            record.tail_len == tail_len &&
            std::equal(record.tail_tokens.begin(), record.tail_tokens.begin() + tail_len,
                       request.tail_tokens.begin())) {
            hit(existing);
            return existing;
        }
    }

    // Take a slot. A reserved slot was already removed from free_slots_ by reserve_snapshot_slot,
    // so use it directly. Otherwise acquire a new one from the free/GDSF paths.
    SlotId slot = request.reserved_slot;
    if (slot == kInvalidSlot) {
        slot = acquire_slot(request.priority);
        if (slot == kInvalidSlot && request.supersede != kInvalidSnapshot) {
            supersede(request.supersede);
            slot = acquire_slot(request.priority);
        }
    }
    if (slot == kInvalidSlot) { return kInvalidSnapshot; }

    // Never recycle the id of the snapshot just superseded in this call: a caller may still hold it
    // to supersede on finish.
    SnapshotId id = kInvalidSnapshot;
    for (std::size_t index = free_snapshots_.size(); index-- > 0U;) {
        if (free_snapshots_[index] == request.supersede) { continue; }
        id                  = free_snapshots_[index];
        free_snapshots_[index] = free_snapshots_.back();
        free_snapshots_.pop_back();
        break;
    }
    if (id != kInvalidSnapshot) {
        Snapshot& reused               = snapshots_[id - 1U];
        const std::uint32_t generation = reused.generation + 1U;
        reused                         = Snapshot{};
        reused.generation              = generation;
    } else {
        snapshots_.emplace_back();
        id = static_cast<SnapshotId>(snapshots_.size());
    }
    Snapshot& record = snapshots_[id - 1U];
    record.anchor       = request.anchor;
    record.frontier     = request.frontier;
    record.tail_len     = static_cast<std::uint8_t>(tail_len);
    std::copy(request.tail_tokens.begin(), request.tail_tokens.end(), record.tail_tokens.begin());
    record.tail_pages    = request.tail_pages;
    record.tail_state    = tail_len > 0U ? Residency::Resident : Residency::Absent;
    record.device_slot   = slot;
    record.device_state  = Residency::Resident;
    record.kind          = request.kind;
    record.superseded    = false;
    record.superseded_at = 0;
    record.gdsf_h        = request.priority;
    record.freq          = 1;
    record.last_hit_tick = ++tick_;
    record.live          = true;

    slot_owner_[static_cast<std::size_t>(slot)] = id;
    nodes_[request.anchor - 1U].snapshots.push_back(id);
    inc_live_below(request.anchor);
    ++live_snapshots_;
    return id;
}

SlotId PrefixIndex::reserve_snapshot_slot(std::uint64_t priority) { return acquire_slot(priority); }

void PrefixIndex::release_reserved_slot(SlotId slot) noexcept {
    if (slot == kInvalidSlot) { return; }
    free_slots_.push_back(slot);
}

void PrefixIndex::supersede(SnapshotId id) noexcept {
    Snapshot* record = id == kInvalidSnapshot ? nullptr
                      : (id <= snapshots_.size() && snapshots_[id - 1U].live ? &snapshots_[id - 1U]
                                                                             : nullptr);
    if (record == nullptr || record->superseded) { return; }
    record->superseded    = true;
    record->superseded_at = ++tick_;
    // A superseded snapshot's host image is the next candidate for reclamation after dead KV.
    if (host_allocator_ != nullptr && record->host_page != kInvalidHostPage) {
        host_allocator_->supersede(record->host_page);
    }
}

void PrefixIndex::hit(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live) { return; }
    record.superseded    = false;
    record.superseded_at = 0;
    ++record.freq;
    record.last_hit_tick = ++tick_;
    if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
        host_allocator_->hit(record.host_page, tick_);
        host_allocator_->clear_superseded(record.host_page);
    }
}

std::uint64_t PrefixIndex::publish_snapshot_manifest(SnapshotId id, std::uint64_t owner,
                                                     std::uint32_t tp,
                                                     std::uint32_t expected_generation) noexcept {
    if (shared_tier_ == nullptr || id == kInvalidSnapshot || id > snapshots_.size()) { return 0; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live || record.generation != expected_generation ||
        record.host_shared || record.tail_len != 0 ||
        record.host_state != Residency::Host || record.host_page == kInvalidHostPage ||
        !valid(record.anchor)) {
        return 0;
    }

    try {
        std::vector<NodeId> path;
        for (NodeId cur = record.anchor; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
            path.push_back(cur);
            if (cur == root_) { break; }
        }
        std::reverse(path.begin(), path.end());

        ninfer::SharedHostTier::PublishData data;
        data.tp        = tp;
        data.frontier  = record.frontier;
        data.state_page = record.host_page;
        data.hashes.reserve(path.size() > 0 ? path.size() - 1U : 0U);
        data.block_pages.reserve(path.size() > 0 ? path.size() - 1U : 0U);
        for (NodeId cur : path) {
            if (cur == root_) { continue; }
            const BlockNode& node = nodes_[cur - 1U];
            if (!node.live || node.host_state != Residency::Host ||
                node.host_page == kInvalidHostPage) {
                return 0;
            }
            data.hashes.push_back(node.lookup_hash);
            data.block_pages.push_back(node.host_page);
        }
        if (data.hashes.empty()) { return 0; }

        const std::uint64_t manifest = shared_tier_->publish(owner, std::move(data));
        if (manifest == 0) { return 0; }

        record.host_shared        = true;
        record.shared_manifest_id = manifest;
        for (NodeId cur : path) {
            if (cur == root_) { continue; }
            BlockNode& node = nodes_[cur - 1U];
            if (!node.host_shared) {
                node.host_shared = true;
            }
            // Append this manifest id to the node's set (deduped).
            if (std::find(node.shared_manifests.begin(), node.shared_manifests.end(), manifest)
                    == node.shared_manifests.end()) {
                node.shared_manifests.push_back(manifest);
            }
        }
        return manifest;
    } catch (...) {
        return 0;
    }
}

SlotId PrefixIndex::detach_snapshot(SnapshotId id) {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return kInvalidSlot; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live) { return kInvalidSlot; }

    if (record.tail_state == Residency::Resident && sink_ != nullptr) {
        sink_->release_tail_pages(record.tail_pages);
    }
    const SlotId slot = record.device_slot;

    // A fully deleted snapshot releases its shared manifest or its private slab, never both. A
    // shared page is owned by the tier; dropping the local reference must not return it through
    // the allocator.
    if (record.host_shared) {
        const std::uint64_t manifest = record.shared_manifest_id;
        // A foreign snapshot's single manifest release belongs to `release_foreign`; detaching it
        // here must not double-release, and a foreign node's host reference stays owned by the
        // foreign record until that release.
        if (!record.foreign && shared_tier_ != nullptr && manifest != 0) {
            shared_tier_->release_manifest(manifest);
        }
        for (NodeId cur = record.anchor; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
            BlockNode& node = nodes_[cur - 1U];
            if (node.host_shared) {
                // Remove only this manifest's id from the node's set.
                node.shared_manifests.erase(
                    std::remove(node.shared_manifests.begin(), node.shared_manifests.end(), manifest),
                    node.shared_manifests.end());
                // Clear the local reference only when no other manifest still owns this node
                // and no restore pin is in flight.
                if (!node.foreign && node.shared_manifests.empty() && node.restore_pins == 0) {
                    node.host_page   = kInvalidHostPage;
                    node.host_state  = Residency::Absent;
                    node.host_shared = false;
                }
            }
            if (cur == root_) { break; }
        }
        record.host_page          = kInvalidHostPage;
        record.host_state         = Residency::Absent;
        record.host_shared        = false;
        record.shared_manifest_id = 0;
    } else if (record.host_page != kInvalidHostPage) {
        if (host_allocator_ != nullptr) {
            host_allocator_->unpin(record.host_page);
            host_allocator_->release(record.host_page);
        }
        record.host_page  = kInvalidHostPage;
        record.host_state = Residency::Absent;
    }

    if (valid(record.anchor)) {
        std::vector<SnapshotId>& list = nodes_[record.anchor - 1U].snapshots;
        list.erase(std::remove(list.begin(), list.end(), id), list.end());
    }
    dec_live_below(record.anchor, 1U);

    record.live = false;
    free_snapshots_.push_back(id);
    --live_snapshots_;
    return slot;
}

SlotId PrefixIndex::demote_snapshot_device(SnapshotId id) {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return kInvalidSlot; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live) { return kInvalidSlot; }

    // Release the tail pages if they are device-resident; the snapshot survives as host-resident.
    if (record.tail_state == Residency::Resident && sink_ != nullptr) {
        sink_->release_tail_pages(record.tail_pages);
        record.tail_state = Residency::Absent;
    }
    const SlotId slot = record.device_slot;
    record.device_slot  = kInvalidSlot;
    record.device_state = Residency::Absent;
    // The snapshot stays alive (still in its anchor's list, still counted in live_snapshots_below)
    // and can be restored later via begin_snapshot_restore.
    return slot;
}

void PrefixIndex::delete_snapshot(SnapshotId id) {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    // A snapshot with an in-flight host transfer is deferred, not deleted: its write must land
    // (or roll back) first. The caller's eviction pass simply makes no progress this round; the
    // snapshot becomes deletable once finish_* releases the pin.
    if (!record.live || record.restore_pins != 0U) { return; }
    const SlotId slot = detach_snapshot(id);
    if (slot == kInvalidSlot) { return; }
    if (sink_ != nullptr) { sink_->release_snapshot_slot(slot); }
    slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
    free_slots_.push_back(slot);
}

void PrefixIndex::remove_snapshot(SnapshotId id) { delete_snapshot(id); }

NodeId PrefixIndex::evict_device_lru() {
    if (lru_head_ == kInvalidNode) { return kInvalidNode; }

    // Pass 1: prefer a backed victim (host_state == Host, unpinned). Demote it: release device
    // pages, unlink from LRU, node stays alive as host-only.
    NodeId current = lru_head_;
    while (current != kInvalidNode) {
        BlockNode& record = nodes_[current - 1U];
        if (record.active_refs == 0 && record.restore_pins == 0 &&
            record.host_state == Residency::Host) {
            if (sink_ != nullptr) { sink_->release_block_pages(record.device_pages); }
            record.device_state = Residency::Absent;
            unlink_lru(current);
            return current;
        }
        current = record.lru_next;
    }

    // Pass 2: no backed victim. Device eviction never copies (HPC): take the first unpinned
    // victim and erase its subtree. The data is lost from the cache; if a host copy existed it
    // would have been found in pass 1. A pinned descendant makes erase_subtree a no-op, so skip
    // that candidate instead of reporting a phantom eviction the caller can never retry.
    current = lru_head_;
    while (current != kInvalidNode) {
        BlockNode& record = nodes_[current - 1U];
        // Save the successor before erasing: erase_subtree unlinks and frees `current`.
        const NodeId next = record.lru_next;
        if (record.active_refs == 0 && record.restore_pins == 0 && erase_subtree(current)) {
            return current;
        }
        current = next;
    }

    return kInvalidNode;
}

std::array<std::uint32_t, kMaxPools> PrefixIndex::evictable_page_counts() const noexcept {
    std::array<std::uint32_t, kMaxPools> counts{0U, 0U};
    for (NodeId id = lru_head_; id != kInvalidNode; id = nodes_[id - 1U].lru_next) {
        const BlockNode& record = nodes_[id - 1U];
        if (record.device_pages.page[0] >= 0) { ++counts[0]; }
        if (record.device_pages.page[1] >= 0) { ++counts[1]; }
    }
    return counts;
}

bool PrefixIndex::erase_subtree(NodeId node) {
    if (!valid(node) || node == root_) { return false; }

    std::vector<NodeId> order;
    std::vector<NodeId> stack{node};
    while (!stack.empty()) {
        const NodeId current = stack.back();
        stack.pop_back();
        const BlockNode& record = nodes_[current - 1U];
        if (!record.live || record.active_refs != 0U || record.restore_pins != 0U) { return false; }
        order.push_back(current);
        for (NodeId child : record.children) { stack.push_back(child); }
    }

    for (NodeId current : order) {
        BlockNode& record = nodes_[current - 1U];
        const std::vector<SnapshotId> anchored = record.snapshots;
        for (SnapshotId id : anchored) { delete_snapshot(id); }
    }

    const NodeId parent = nodes_[node - 1U].parent;
    if (valid(parent)) {
        std::vector<NodeId>& siblings = nodes_[parent - 1U].children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), node), siblings.end());
    }

    for (NodeId current : order) {
        BlockNode& record = nodes_[current - 1U];
        if (record.lru_linked) { unlink_lru(current); }
        if (record.device_state == Residency::Resident) {
            if (sink_ != nullptr) { sink_->release_block_pages(record.device_pages); }
            record.device_state = Residency::Absent;
        }
        // A shared page belongs to the tier manifest, not to this node; dropping the local
        // reference must not return it through the allocator. A private page goes back to the pool.
        if (record.host_shared) {
            record.host_page          = kInvalidHostPage;
            record.host_state         = Residency::Absent;
            record.host_shared        = false;
            record.shared_manifests.clear();
        } else if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
            host_allocator_->release(record.host_page);
            record.host_page  = kInvalidHostPage;
            record.host_state = Residency::Absent;
        }
        free_node(current);
    }
    return true;
}

// --- M2 async fill APIs ---

HostWriteTarget PrefixIndex::begin_host_write(NodeId id) {
    HostWriteTarget result;
    if (!valid(id) || host_allocator_ == nullptr) { return result; }
    BlockNode& record = nodes_[id - 1U];
    if (record.device_state != Residency::Resident || record.host_state != Residency::Absent) {
        return result;
    }

    // Allocate a host slab. If the pool evicted another slab to make room, reconcile its owner.
    SlabAllocation alloc = host_allocator_->allocate_block();
    if (alloc.page == kInvalidHostPage) { return result; }
    if (alloc.evicted != kInvalidHostPage) { reconcile_host_eviction(alloc.evicted); }

    record.host_page  = alloc.page;
    record.host_state = Residency::Filling;
    // Keep the node out of the eviction LRU for the duration of the transfer: its pages must stay
    // allocated while the D2H reads them.
    pin_restore(id);
    host_allocator_->pin(alloc.page);
    // Dead-KV: a block no snapshot reaches is the first host-resident data to go.
    if (record.live_snapshots_below == 0U) { host_allocator_->mark_dead(alloc.page); }
    result.node = id;
    result.page = alloc.page;
    return result;
}

void PrefixIndex::finish_host_write(NodeId id, bool ok) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.host_state != Residency::Filling) { return; }
    if (ok) {
        record.host_state = Residency::Host;
        // Transfer complete: release the in-flight pin so the slab is evictable by policy again.
        if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
            host_allocator_->unpin(record.host_page);
        }
    } else {
        // Rollback: release the host slab.
        if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
            host_allocator_->unpin(record.host_page);
            host_allocator_->release(record.host_page);
            record.host_page = kInvalidHostPage;
        }
        record.host_state = Residency::Absent;
    }
    unpin_restore(id);
}

void PrefixIndex::begin_device_fill(NodeId id, PoolPages pages) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.device_state != Residency::Absent) { return; }
    record.device_pages = pages;
    record.device_state = Residency::Filling;
    // Pin the node (pages must not be evicted mid-copy) and the source slab (the pool must not
    // reclaim it while the H2D reads from it).
    pin_restore(id);
    if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
        host_allocator_->pin(record.host_page);
    }
}

void PrefixIndex::finish_device_fill(NodeId id, bool ok) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.device_state != Residency::Filling) { return; }
    if (ok) {
        record.device_state = Residency::Resident;
    } else {
        // Rollback: release the device pages.
        if (sink_ != nullptr) { sink_->release_block_pages(record.device_pages); }
        record.device_pages = PoolPages{};
        record.device_state = Residency::Absent;
    }
    if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
        host_allocator_->unpin(record.host_page);
    }
    unpin_restore(id); // re-links into the LRU when the node is now fully unpinned
    // After the last restore pin is released, if no manifest still owns this node, clear the
    // local host reference so the page is not left dangling after the manifest is evicted. A
    // foreign node keeps its reference: `release_foreign` owns its manifest and its host page.
    if (!record.foreign && record.host_shared && record.shared_manifests.empty() &&
        record.restore_pins == 0) {
        record.host_page   = kInvalidHostPage;
        record.host_state  = Residency::Absent;
        record.host_shared = false;
    }
}

SnapshotWriteTarget PrefixIndex::begin_snapshot_write(SnapshotId id) {
    SnapshotWriteTarget result;
    if (id == kInvalidSnapshot || id > snapshots_.size() || host_allocator_ == nullptr) {
        return result;
    }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live || record.device_state != Residency::Resident ||
        record.host_state != Residency::Absent) {
        return result;
    }

    // Allocate a snapshot slab. If the pool evicted another slab to make room, reconcile its owner.
    SlabAllocation alloc = host_allocator_->allocate_snapshot();
    if (alloc.page == kInvalidHostPage) { return result; }
    if (alloc.evicted != kInvalidHostPage) { reconcile_host_eviction(alloc.evicted); }

    record.host_page  = alloc.page;
    record.host_state = Residency::Filling;
    ++record.restore_pins;
    // Pin the anchor node so neither it nor its subtree can be erased while the image is being
    // written out (erase_subtree aborts on a pinned node).
    pin_restore(record.anchor);
    host_allocator_->pin(alloc.page);
    result.id   = id;
    result.page = alloc.page;
    return result;
}

void PrefixIndex::finish_snapshot_write(SnapshotId id, bool ok) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live || record.host_state != Residency::Filling) { return; }
    if (ok) {
        record.host_state = Residency::Host;
        // Transfer complete: release the in-flight pin so the slab is evictable by policy again.
        if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
            host_allocator_->unpin(record.host_page);
        }
    } else {
        // Rollback: release the host slab.
        if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
            host_allocator_->unpin(record.host_page);
            host_allocator_->release(record.host_page);
            record.host_page = kInvalidHostPage;
        }
        record.host_state = Residency::Absent;
    }
    if (record.restore_pins > 0U) { --record.restore_pins; }
    unpin_restore(record.anchor);
}

SnapshotRestoreTarget PrefixIndex::begin_snapshot_restore(SnapshotId id) {
    SnapshotRestoreTarget result;
    if (id == kInvalidSnapshot || id > snapshots_.size() || host_allocator_ == nullptr) {
        return result;
    }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live || record.host_state != Residency::Host) { return result; }

    // Acquire a device slot. If no free slot is available, evict one per GDSF.
    SlotId slot = acquire_slot(0);
    if (slot == kInvalidSlot) { return result; }

    record.device_slot  = slot;
    record.device_state = Residency::Filling;
    ++record.restore_pins;
    host_allocator_->pin(record.host_page);
    result.id   = id;
    result.slot = slot;
    result.page = record.host_page;
    return result;
}

void PrefixIndex::finish_snapshot_restore(SnapshotId id, bool ok) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live || record.device_state != Residency::Filling) { return; }
    if (ok) {
        record.device_state = Residency::Resident;
    } else {
        // Rollback: return the slot to the free list.
        const SlotId slot = record.device_slot;
        record.device_slot  = kInvalidSlot;
        record.device_state = Residency::Absent;
        if (slot != kInvalidSlot) {
            slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
            free_slots_.push_back(slot);
        }
    }
    if (host_allocator_ != nullptr && record.host_page != kInvalidHostPage) {
        host_allocator_->unpin(record.host_page);
    }
    if (record.restore_pins > 0U) { --record.restore_pins; }
}

// --- foreign import (cross-instance restore) ---

bool PrefixIndex::import_foreign(const ForeignView& view) {
    if (shared_tier_ == nullptr || view.manifest_id == 0) { return false; }
    if (view.frontier == 0 || view.hashes.empty() || view.hashes.size() != view.block_pages.size() ||
        view.state_page == kInvalidHostPage) {
        return false;
    }
    const auto existing = foreign_records_.find(view.manifest_id);
    if (existing != foreign_records_.end()) { return false; }

    // The manifest ref is taken here and released exactly once by `release_foreign`. `match` itself
    // does not retain, so the caller must not retain twice.
    if (!shared_tier_->retain_manifest(view.manifest_id)) { return false; }

    ForeignRecord record;
    record.manifest_id = view.manifest_id;
    NodeId parent = root_;
    for (std::size_t block = 0; block < view.hashes.size(); ++block) {
        const NodeId id = allocate_node();
        BlockNode& node = nodes_[id - 1U];
        node.parent       = parent;   // foreign chain parent; not registered under the local root
        node.depth        = static_cast<std::int32_t>(block);
        node.lookup_hash  = view.hashes[block];
        node.host_page    = view.block_pages[block];
        node.host_state   = Residency::Host;
        node.host_shared  = true;      // owned by the manifest, never released through the allocator
        node.foreign      = true;
        node.shared_manifests.push_back(view.manifest_id);
        record.nodes.push_back(id);
        parent = id;
    }

    snapshots_.emplace_back();
    const SnapshotId sid = static_cast<SnapshotId>(snapshots_.size());
    Snapshot& snap = snapshots_[sid - 1U];
    snap.anchor             = record.nodes.back();
    snap.frontier           = view.frontier;
    snap.tail_len           = 0;
    snap.host_page          = view.state_page;
    snap.host_state         = Residency::Host;
    snap.host_shared        = true;
    snap.shared_manifest_id = view.manifest_id;
    snap.kind               = SnapshotKind::Tap;
    snap.freq               = 1;
    snap.last_hit_tick      = ++tick_;
    snap.live               = true;
    snap.foreign            = true;
    ++live_snapshots_;
    nodes_[snap.anchor - 1U].snapshots.push_back(sid);
    record.snapshot = sid;

    foreign_records_[view.manifest_id] = std::move(record);
    return true;
}

NodeId PrefixIndex::foreign_node(std::uint64_t manifest_id, std::size_t block) const noexcept {
    const auto it = foreign_records_.find(manifest_id);
    if (it == foreign_records_.end() || block >= it->second.nodes.size()) { return kInvalidNode; }
    return it->second.nodes[block];
}

SnapshotId PrefixIndex::foreign_snapshot(std::uint64_t manifest_id) const noexcept {
    const auto it = foreign_records_.find(manifest_id);
    if (it == foreign_records_.end()) { return kInvalidSnapshot; }
    return it->second.snapshot;
}

bool PrefixIndex::commit_foreign(std::uint64_t manifest_id, std::span<const TokenId> tokens) noexcept {
    const auto found = foreign_records_.find(manifest_id);
    if (found == foreign_records_.end()) { return false; }
    ForeignRecord& record = found->second;
    if (record.committed) { return true; }
    if (tokens.size() < record.nodes.size() * static_cast<std::size_t>(kBlockTokens)) { return false; }

    NodeId local_parent = root_;
    NodeId deepest      = root_;
    for (std::size_t block = 0; block < record.nodes.size(); ++block) {
        const NodeId foreign_id = record.nodes[block];
        const BlockNode& foreign = nodes_[foreign_id - 1U];
        const std::span<const TokenId> block_tokens =
            tokens.subspan(block * static_cast<std::size_t>(kBlockTokens),
                           static_cast<std::size_t>(kBlockTokens));
        const ChildMatch existing = find_child(local_parent, foreign.lookup_hash, block_tokens, 0);
        if (existing.node != kInvalidNode) {
            // Rule 6: the local node already owns this block. Keep it; the foreign node carried no
            // device pages for this block (the restore used the local node), so drop it.
            BlockNode& local = nodes_[existing.node - 1U];
            if (foreign.device_state == Residency::Resident &&
                local.device_state != Residency::Resident && local.restore_pins == 0) {
                local.device_pages  = foreign.device_pages;
                local.device_state  = Residency::Resident;
                link_lru_tail(existing.node);
            }
            free_node(foreign_id);
            deepest = existing.node;
            local_parent = existing.node;
            continue;
        }

        // No local node: the foreign node itself becomes the local node for this block. Its device
        // pages (filled during the restore) transfer to the local tree; its host reference is
        // dropped because the host page stays owned by the manifest. The node takes the block's
        // exact tokens so later `find_child` token comparisons can match it.
        BlockNode& node = nodes_[foreign_id - 1U];
        node.parent           = local_parent;
        std::copy(block_tokens.begin(), block_tokens.end(), node.tokens.begin());
        node.host_page        = kInvalidHostPage;
        node.host_state       = Residency::Absent;
        node.host_shared      = false;
        node.shared_manifests.clear();
        node.foreign          = false;
        nodes_[local_parent - 1U].children.push_back(foreign_id);
        link_lru_tail(foreign_id);
        deepest = foreign_id;
        local_parent = foreign_id;
    }

    // The foreign snapshot becomes a normal local snapshot anchored at the deepest committed node.
    // Its manifest ref was already accounted for by `release_foreign`; the index-side ownership of
    // the manifest ends here.
    if (record.snapshot != kInvalidSnapshot && record.snapshot <= snapshots_.size() &&
        snapshots_[record.snapshot - 1U].live) {
        Snapshot& snap = snapshots_[record.snapshot - 1U];
        std::vector<SnapshotId>& old_list = nodes_[snap.anchor - 1U].snapshots;
        old_list.erase(std::remove(old_list.begin(), old_list.end(), record.snapshot),
                       old_list.end());
        snap.anchor = deepest;
        nodes_[deepest - 1U].snapshots.push_back(record.snapshot);
        snap.host_page          = kInvalidHostPage;
        snap.host_state         = Residency::Absent;
        snap.host_shared        = false;
        snap.shared_manifest_id = 0;
        snap.foreign            = false;
        inc_live_below(deepest);
    }

    record.committed = true;
    return true;
}

void PrefixIndex::release_foreign(std::uint64_t manifest_id) noexcept {
    const auto found = foreign_records_.find(manifest_id);
    if (found == foreign_records_.end()) { return; }
    ForeignRecord& record = found->second;

    if (!record.committed) {
        for (NodeId id : record.nodes) {
            if (!valid(id)) { continue; }
            BlockNode& node = nodes_[id - 1U];
            if (node.device_state == Residency::Resident && sink_ != nullptr) {
                sink_->release_block_pages(node.device_pages);
            }
            // The host page belongs to the manifest; the target must never release it.
            node.host_page          = kInvalidHostPage;
            node.host_state         = Residency::Absent;
            node.host_shared        = false;
            node.shared_manifests.clear();
            free_node(id);
        }
        if (record.snapshot != kInvalidSnapshot && record.snapshot <= snapshots_.size() &&
            snapshots_[record.snapshot - 1U].live) {
            Snapshot& snap = snapshots_[record.snapshot - 1U];
            const SlotId slot = snap.device_slot;
            if (slot != kInvalidSlot) {
                if (sink_ != nullptr) { sink_->release_snapshot_slot(slot); }
                slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
                free_slots_.push_back(slot);
            }
            if (valid(snap.anchor)) {
                std::vector<SnapshotId>& list = nodes_[snap.anchor - 1U].snapshots;
                list.erase(std::remove(list.begin(), list.end(), record.snapshot), list.end());
            }
            snap.host_page          = kInvalidHostPage;
            snap.host_state         = Residency::Absent;
            snap.host_shared        = false;
            snap.shared_manifest_id = 0;
            snap.device_slot        = kInvalidSlot;
            snap.device_state       = Residency::Absent;
            snap.live               = false;
            free_snapshots_.push_back(record.snapshot);
            --live_snapshots_;
        }
    }

    // Exactly one release for the ref `import_foreign` took.
    shared_tier_->release_manifest(record.manifest_id);
    foreign_records_.erase(found);
}

} // namespace ninfer::prefix_cache
