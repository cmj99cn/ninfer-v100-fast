#include "runtime/engine/engine_router.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ninfer::runtime::CandidateProvider;
using ninfer::runtime::EngineRouter;
using ninfer::runtime::PendingRequest;
using ninfer::runtime::RouteCandidate;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

RouteCandidate make_candidate(std::uint32_t instance, std::uint32_t device_frontier,
                              std::uint32_t free_pages, std::uint32_t active,
                              std::uint32_t pending, std::uint32_t max_concurrency) {
    return RouteCandidate{
        .instance = instance,
        .probe = {.device_frontier = device_frontier},
        .capacity = {
            .free_pages = free_pages,
            .total_pages = 4096,
            .active_requests = active,
            .pending_requests = pending,
            .max_concurrency = max_concurrency,
        },
    };
}

// The router reads one (pending, instance) pair through the provider, so the test injects the
// scripted view instead of a fixed candidate array. Scoring itself is the production router.
CandidateProvider provider_for(std::vector<RouteCandidate> candidates) {
    return [candidates = std::move(candidates)](const PendingRequest&, std::uint32_t instance) {
        for (const RouteCandidate& candidate : candidates) {
            if (candidate.instance == instance) { return candidate; }
        }
        return RouteCandidate{.instance = instance};
    };
}

std::vector<std::uint32_t> instance_ids(const std::vector<RouteCandidate>& candidates) {
    std::vector<std::uint32_t> ids;
    for (const RouteCandidate& candidate : candidates) { ids.push_back(candidate.instance); }
    return ids;
}

int test_deepest_prefix_wins() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 2> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .has_home = false,
        },
        PendingRequest{
            .id = 2, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .has_home = false,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(1, 8192, 1024, 0, 0, 4),
        make_candidate(2, 4096, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "deepest-prefix routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->instance == 1, "the shallower candidate won the deepest-prefix routing");
    failures += check(decision->pending_index == 0, "the deepest prefix did not route the queue head");
    failures += check(!decision->use_hint, "a zero manifest frontier set use_hint");
    return failures;
}

// Two pending requests with different block-hash chains see different probes on the same instance:
// the deeper chain belongs to the second pending, so the provider must be read per pair.
int test_probe_is_per_pending_pair() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 2> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .has_home = false,
        },
        PendingRequest{
            .id = 2, .prompt_tokens = 4096, .max_tokens = 64, .required_pages = 67,
            .enqueued = now, .has_home = false,
        },
    };
    // Per (pending, instance) frontier. Pending 1 sees 1024/4096 and pending 2 sees 4096/8192, so a
    // single per-candidate probe could not reproduce this decision.
    auto provider = [](const PendingRequest& request, std::uint32_t instance) {
        const std::uint32_t frontier = request.id == 1 ? (instance == 1 ? 1024 : 4096)
                                                       : (instance == 1 ? 4096 : 8192);
        return RouteCandidate{
            .instance = instance,
            .probe = {.device_frontier = frontier},
            .capacity = {
                .free_pages = 1024,
                .total_pages = 4096,
                .max_concurrency = 4,
            },
        };
    };

    const EngineRouter router;
    const std::array<std::uint32_t, 2> ids{1, 2};
    const auto decision = router.pick(pending, provider, ids, now);
    int failures = check(decision.has_value(), "per-pair probe routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->pending_index == 1, "the per-pair probe did not score the deeper pending");
    failures += check(decision->instance == 2, "the per-pair probe did not route to its best instance");
    return failures;
}

int test_no_prefix_does_not_idle() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 1> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 64, .max_tokens = 64, .required_pages = 3,
            .enqueued = now, .has_home = false,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(3, 0, 1024, 0, 0, 4),
        make_candidate(2, 0, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "zero-frontier routing idled instead of dispatching the head");
    if (failures != 0) { return failures; }
    failures += check(decision->pending_index == 0, "the head pending was not dispatched");
    failures += check(decision->instance == 2, "equal-score candidates were not ordered by instance ascending");
    return failures;
}

int test_cross_pending_tie_keeps_head() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 2> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .home_instance = 1, .has_home = true,
        },
        PendingRequest{
            .id = 2, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .home_instance = 2, .has_home = true,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(1, 4096, 1024, 0, 0, 4),
        make_candidate(2, 4096, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "cross-pending tie routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->pending_index == 0, "the cross-pending tie did not keep the queue head");
    failures += check(decision->instance == 1, "the queue head did not route to its own best candidate");
    return failures;
}

int test_aging_beats_affinity() {
    const Clock::time_point base = Clock::now();
    const Clock::time_point now = base + std::chrono::milliseconds(1500);
    std::array<PendingRequest, 2> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = base, .has_home = false,
        },
        PendingRequest{
            .id = 2, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = base + std::chrono::milliseconds(1200), .home_instance = 2, .has_home = true,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(1, 1024, 1024, 0, 0, 4),
        make_candidate(2, 2048, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "aged-head routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->pending_index == 0, "aging did not dispatch the aged head pending");
    return failures;
}

int test_home_busy_loses_slack() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 1> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 1024, .max_tokens = 64, .required_pages = 18,
            .enqueued = now, .home_instance = 1, .has_home = true,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(1, 4096, 1024, 0, 1, 2),
        make_candidate(2, 4096, 1024, 0, 0, 2),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "home-busy routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->instance == 2, "the busy home candidate kept the affinity slack");
    return failures;
}

int test_capacity_filter_and_home_slack() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 1> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 4096, .max_tokens = 128, .required_pages = 67,
            .enqueued = now, .home_instance = 3, .has_home = true,
        },
    };
    std::vector<RouteCandidate> candidates{
        make_candidate(1, 8192, 64, 0, 0, 4),
        make_candidate(2, 4096, 128, 0, 0, 4),
        RouteCandidate{
            .instance = 3,
            .probe = {.device_frontier = 4096, .manifest_frontier = 2048, .manifest_id = 42},
            .capacity = {
                .free_pages = 128,
                .total_pages = 4096,
                .max_concurrency = 4,
            },
        },
        make_candidate(4, 8192, 1024, 3, 1, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "capacity-filtered routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->instance == 3, "the home candidate did not win inside the affinity slack");
    failures += check(decision->use_hint, "a live manifest frontier did not set use_hint");
    failures += check(decision->manifest_id == 42, "the decision did not carry the selected manifest id");
    failures += check(decision->manifest_frontier == 2048, "the decision did not carry the selected frontier");

    std::vector<RouteCandidate> no_room{candidates[0], candidates[3]};
    failures += check(!router.pick(pending, provider_for(no_room), instance_ids(no_room), now).has_value(),
                      "routing returned a decision when no candidate had room");
    return failures;
}

// A tap state image exists only at the manifest frontier. A manifest that reaches the whole prompt
// cannot restore a shorter prefix, so it is not a hint and must not outscore a usable local prefix.
int test_full_prompt_manifest_is_not_a_hint() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 1> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 4096, .max_tokens = 64, .required_pages = 67,
            .enqueued = now, .has_home = false,
        },
    };
    std::vector<RouteCandidate> candidates{
        RouteCandidate{
            .instance = 1,
            .probe = {.device_frontier = 0, .manifest_frontier = 4096, .manifest_id = 42},
            .capacity = {
                .free_pages = 1024,
                .total_pages = 4096,
                .max_concurrency = 4,
            },
        },
        make_candidate(2, 2048, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "full-prompt manifest routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->instance == 2, "an unusable full-prompt manifest outscored the device prefix");
    failures += check(!decision->use_hint, "a full-prompt manifest was selected as a hint");
    failures += check(decision->manifest_id == 0, "a full-prompt manifest leaked its id into the decision");
    return failures;
}

// A manifest strictly below the prompt frontier is a real hint.
int test_sub_prompt_manifest_is_a_hint() {
    const Clock::time_point now = Clock::now();
    std::array<PendingRequest, 1> pending{
        PendingRequest{
            .id = 1, .prompt_tokens = 4096, .max_tokens = 64, .required_pages = 67,
            .enqueued = now, .has_home = false,
        },
    };
    std::vector<RouteCandidate> candidates{
        RouteCandidate{
            .instance = 1,
            .probe = {.device_frontier = 0, .manifest_frontier = 2048, .manifest_id = 42},
            .capacity = {
                .free_pages = 1024,
                .total_pages = 4096,
                .max_concurrency = 4,
            },
        },
        make_candidate(2, 1024, 1024, 0, 0, 4),
    };

    const EngineRouter router;
    const auto decision = router.pick(pending, provider_for(candidates), instance_ids(candidates), now);
    int failures = check(decision.has_value(), "sub-prompt manifest routing returned no decision");
    if (failures != 0) { return failures; }
    failures += check(decision->instance == 1, "the sub-prompt manifest did not win");
    failures += check(decision->use_hint, "a sub-prompt manifest did not set use_hint");
    failures += check(decision->manifest_id == 42, "the sub-prompt manifest id was lost");
    failures += check(decision->manifest_frontier == 2048, "the sub-prompt manifest frontier was lost");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_deepest_prefix_wins();
    failures += test_probe_is_per_pending_pair();
    failures += test_no_prefix_does_not_idle();
    failures += test_cross_pending_tie_keeps_head();
    failures += test_aging_beats_affinity();
    failures += test_home_busy_loses_slack();
    failures += test_capacity_filter_and_home_slack();
    failures += test_full_prompt_manifest_is_not_a_hint();
    failures += test_sub_prompt_manifest_is_a_hint();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
