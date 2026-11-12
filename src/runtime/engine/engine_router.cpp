#include "runtime/engine/engine_router.h"

#include <algorithm>
#include <vector>

namespace ninfer::runtime {

namespace {

// Net device pages the request newly allocates on this candidate. The device-resident prefix
// (device_frontier) is shared with the candidate's cache, so only tokens beyond it need fresh
// pages; a manifest (host-tier) restore still allocates device pages and does not reduce the
// page bill. `required_pages` on the pending request stays the reuse-blind gross upper bound;
// the router derives the per-candidate net requirement from the same formula minus the reuse.
[[nodiscard]] std::uint32_t net_required_pages(const RouteCandidate& candidate,
                                               const PendingRequest& request,
                                               const EngineRouter::Config& config) noexcept {
    const std::uint64_t reuse =
        std::min<std::uint64_t>(candidate.probe.device_frontier, request.prompt_tokens);
    const std::uint64_t tokens = static_cast<std::uint64_t>(request.prompt_tokens) +
                                 request.max_tokens + config.safety_margin_tokens - reuse;
    return static_cast<std::uint32_t>((tokens + config.block_tokens - 1) / config.block_tokens);
}

[[nodiscard]] bool has_room(const RouteCandidate& candidate, const PendingRequest& request,
                            const EngineRouter::Config& config) noexcept {
    return candidate.capacity.free_pages >= net_required_pages(candidate, request, config) &&
           candidate.capacity.active_requests + candidate.capacity.pending_requests <
               candidate.capacity.max_concurrency;
}

// A tap state image carries only the GDN state at its frontier, so a manifest that covers the
// whole prompt cannot restore a shorter prefix. Such a manifest is not a hint; the device frontier
// (if any) is the only usable local match.
[[nodiscard]] std::uint32_t usable_manifest_frontier(const RouteCandidate& candidate,
                                                     const PendingRequest& request) noexcept {
    return candidate.probe.manifest_frontier > 0 &&
                   candidate.probe.manifest_frontier < request.prompt_tokens
               ? candidate.probe.manifest_frontier
               : 0;
}

[[nodiscard]] std::uint64_t score(const RouteCandidate& candidate, const PendingRequest& request,
                                  const EngineRouter::Config& config) noexcept {
    const std::uint32_t manifest = usable_manifest_frontier(candidate, request);
    const std::uint32_t frontier =
        candidate.probe.device_frontier > manifest ? candidate.probe.device_frontier : manifest;
    const std::uint64_t home_bonus =
        request.has_home && request.home_instance == candidate.instance &&
            candidate.capacity.pending_requests == 0
            ? static_cast<std::uint64_t>(config.affinity_slack_tokens)
            : 0;
    return frontier + home_bonus;
}

// True when `left` beats `right` on the equal-score key (pending_requests, active_requests,
// instance) ascending.
[[nodiscard]] bool tie_breaks_before(const RouteCandidate& left, const RouteCandidate& right) noexcept {
    if (left.capacity.pending_requests != right.capacity.pending_requests) {
        return left.capacity.pending_requests < right.capacity.pending_requests;
    }
    if (left.capacity.active_requests != right.capacity.active_requests) {
        return left.capacity.active_requests < right.capacity.active_requests;
    }
    return left.instance < right.instance;
}

// Index of the best fitting candidate for one pending, or nullopt when none has room. The filter
// derives the per-candidate net page requirement (gross `required_pages` minus the candidate's
// device-resident prefix); a warm home is no longer evicted by a reuse-blind gross check.
[[nodiscard]] std::optional<std::size_t> best_candidate(const PendingRequest& request,
                                        std::span<const RouteCandidate> candidates,
                                        const EngineRouter::Config& config) {
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const RouteCandidate& candidate = candidates[i];
        if (!has_room(candidate, request, config)) { continue; }
        if (!best.has_value() || score(candidate, request, config) > score(candidates[*best], request, config) ||
            (score(candidate, request, config) == score(candidates[*best], request, config) &&
             tie_breaks_before(candidate, candidates[*best]))) {
            best = i;
        }
    }
    return best;
}

[[nodiscard]] RouteDecision decision_for(std::size_t pending_index, const RouteCandidate& candidate,
                                         const PendingRequest& request,
                                         const EngineRouter::Config& config) {
    const std::uint32_t manifest = usable_manifest_frontier(candidate, request);
    return RouteDecision{
        .pending_index = pending_index,
        .instance = candidate.instance,
        .use_hint = manifest > 0,
        .manifest_id = manifest > 0 ? candidate.probe.manifest_id : 0,
        .manifest_frontier = manifest,
        .reserved_pages = net_required_pages(candidate, request, config),
    };
}

} // namespace

std::optional<RouteDecision> EngineRouter::pick(std::span<const PendingRequest> pending,
                                                const CandidateProvider& provider,
                                                std::span<const std::uint32_t> instances,
                                                std::chrono::steady_clock::time_point now) const {
    if (pending.empty() || instances.empty()) { return std::nullopt; }

    std::vector<RouteCandidate> head_candidates;
    head_candidates.reserve(instances.size());
    for (const std::uint32_t instance : instances) {
        head_candidates.push_back(provider(pending[0], instance));
    }
    const auto head_best = best_candidate(pending[0], head_candidates, config_);
    if (head_best.has_value() && now - pending[0].enqueued >= config_.fairness_aging) {
        return decision_for(0, head_candidates[*head_best], pending[0], config_);
    }

    std::optional<std::size_t> chosen_pending;
    RouteCandidate chosen_candidate{};
    std::uint64_t chosen_score = 0;
    for (std::size_t i = 0; i < pending.size(); ++i) {
        std::vector<RouteCandidate> candidates;
        candidates.reserve(instances.size());
        for (const std::uint32_t instance : instances) {
            candidates.push_back(provider(pending[i], instance));
        }
        const auto best = best_candidate(pending[i], candidates, config_);
        if (!best.has_value()) { continue; }
        const std::uint64_t s = score(candidates[*best], pending[i], config_);
        if (!chosen_pending.has_value() || s > chosen_score) {
            chosen_pending   = i;
            chosen_candidate = candidates[*best];
            chosen_score     = s;
        }
    }
    if (!chosen_pending.has_value()) { return std::nullopt; }
    return decision_for(*chosen_pending, chosen_candidate, pending[*chosen_pending], config_);
}

} // namespace ninfer::runtime
