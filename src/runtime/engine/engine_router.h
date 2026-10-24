#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::runtime {

// One ingress request awaiting a routing decision. `required_pages` is the authoritative filter
// input for `pick`: the caller computes it as
// `ceil((prompt_tokens + max_tokens + safety_margin_tokens) / 64)` from the router Config before
// enqueueing; `pick` consumes the stored value and never re-derives the formula.
struct PendingRequest {
    // Assigned by the pool on admission; callers leave it 0.
    std::uint64_t id = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t max_tokens = 0;
    std::uint32_t required_pages = 0;
    std::chrono::steady_clock::time_point enqueued{};
    std::uint32_t home_instance = 0;
    bool has_home = false;
    std::span<const std::uint64_t> block_hashes{};
};

// One routable Engine as a passive observation pair: its prefix probe and capacity snapshot. The
// router reads these values only; it owns no Engine state and mutates nothing. The probe is read at
// the (pending, instance) pair the provider is called for, so two pending requests with different
// block-hash chains see different probes on the same instance.
struct RouteCandidate {
    std::uint32_t instance = 0;
    PrefixProbe probe;
    CapacitySnapshot capacity;
};

// Builds the candidate view for one (pending, instance) pair. The caller owns the probe and capacity
// source; the router only scores what the provider returns.
using CandidateProvider = std::function<RouteCandidate(const PendingRequest&, std::uint32_t instance)>;

struct RouteDecision {
    std::size_t pending_index = 0;
    std::uint32_t instance = 0;
    bool use_hint = false;
    std::uint64_t manifest_id = 0;
    std::uint32_t manifest_frontier = 0;
};

// Pure host-side scoring over pending requests and candidate capacity views: no device work, no
// mutation of probes or snapshots, and no admission itself. The caller applies the decision.
class EngineRouter {
public:
    struct Config {
        std::uint32_t affinity_slack_tokens = 2048;
        std::chrono::milliseconds fairness_aging{1000};
        std::uint32_t safety_margin_tokens = 64;
        // Probe granularity the caller's `required_pages` formula divides by.
        std::uint32_t block_tokens = 64;
    };

    EngineRouter() = default;
    explicit EngineRouter(Config config) : config_(config) {}

    // Chooses the (pending, candidate) pair with the deepest usable prefix. For every pending the
    // provider is called once per instance in `instances`, so each (pending, candidate) pair is
    // scored with its own probe. Candidates must have room for the pending's `required_pages` (the
    // single authority, computed by the caller) and a free lane. The home bonus applies only to an
    // idle home candidate. Equal scores break by `(pending_requests, active_requests, instance)`
    // ascending, and equal best scores across pendings keep the earlier (queue-head) pending. An aged
    // queue head dispatches to its own best candidate regardless of deeper prefixes elsewhere in the
    // queue. `nullopt` only when no pending has any fitting candidate.
    [[nodiscard]] std::optional<RouteDecision> pick(std::span<const PendingRequest> pending,
                                      const CandidateProvider& provider,
                                      std::span<const std::uint32_t> instances,
                                      std::chrono::steady_clock::time_point now) const;

private:
    Config config_;
};

} // namespace ninfer::runtime
