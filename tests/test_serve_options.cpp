#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.groups == 1 && defaults.group_devices.empty() &&
                          defaults.max_concurrency == 1 && defaults.http_threads == 0 &&
                          defaults.global_pending_requests == 16 &&
                          defaults.affinity_slack_tokens == 2048 &&
                          defaults.fairness_aging_ms == 1000,
                      "single-group pooling/HTTP defaults changed");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.default_max_tokens == 65536,
                      "default output-token budget mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15",
                     "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash and Vision were accepted together");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "40", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    failures += check(to_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!to_request_options(request, configured).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");
    const ninfer::RequestOptions inherited_sampling = to_request_options(request, sampling);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling).execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    // --chat-template-kwargs: server-level chat-template defaults parsed from a JSON object.
    const ServeOptions template_kwargs = parse(
        {"ninfer-serve", "model.ninfer",
         "--chat-template-kwargs", R"({"preserve_thinking": true, "reasoning_effort": "low"})"});
    failures += check(template_kwargs.preserve_thinking &&
                          template_kwargs.reasoning_effort_default == RequestedReasoningEffort::Low,
                      "--chat-template-kwargs did not apply preserve_thinking/reasoning_effort");

    const ServeOptions template_enable_off = parse(
        {"ninfer-serve", "model.ninfer",
         "--chat-template-kwargs", R"({"enable_thinking": false})"});
    failures += check(!template_enable_off.enable_thinking,
                      "--chat-template-kwargs enable_thinking=false did not disable thinking");

    const ServeOptions template_none = parse(
        {"ninfer-serve", "model.ninfer",
         "--chat-template-kwargs", R"({"reasoning_effort": "none"})"});
    failures += check(template_none.reasoning_effort_default == RequestedReasoningEffort::Low,
                      "--chat-template-kwargs reasoning_effort=none did not fold to low");

    const ServeOptions template_high = parse(
        {"ninfer-serve", "model.ninfer",
         "--chat-template-kwargs", R"({"reasoning_effort": "high"})"});
    failures += check(template_high.reasoning_effort_default == RequestedReasoningEffort::Medium,
                      "--chat-template-kwargs reasoning_effort=high did not fold to medium");

    bool unknown_key_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--chat-template-kwargs", R"({"foo": 1})"});
    } catch (const std::invalid_argument&) { unknown_key_rejected = true; }
    failures += check(unknown_key_rejected, "unknown --chat-template-kwargs key was accepted");

    bool bad_json_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--chat-template-kwargs", "{not json"});
    } catch (const std::invalid_argument&) { bad_json_rejected = true; }
    failures += check(bad_json_rejected, "non-object --chat-template-kwargs JSON was accepted");

    bool non_object_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--chat-template-kwargs", "[1, 2]"});
    } catch (const std::invalid_argument&) { non_object_rejected = true; }
    failures += check(non_object_rejected, "non-object --chat-template-kwargs JSON was accepted");

    bool enable_conflict_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-thinking",
                     "--chat-template-kwargs", R"({"enable_thinking": true})"});
    } catch (const std::invalid_argument&) { enable_conflict_rejected = true; }
    failures += check(enable_conflict_rejected,
                      "--no-thinking conflicted with enable_thinking=true");

    bool preserve_conflict_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--preserve-thinking",
                     "--chat-template-kwargs", R"({"preserve_thinking": false})"});
    } catch (const std::invalid_argument&) { preserve_conflict_rejected = true; }
    failures += check(preserve_conflict_rejected,
                      "--preserve-thinking conflicted with preserve_thinking=false");

    bool effort_no_thinking_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--chat-template-kwargs",
                     R"({"reasoning_effort": "low", "enable_thinking": false})"});
    } catch (const std::invalid_argument&) { effort_no_thinking_rejected = true; }
    failures += check(effort_no_thinking_rejected,
                      "--chat-template-kwargs reasoning_effort was accepted with thinking off");

    // The server default reasoning_effort applies when a request omits it and is resolved exactly
    // like a per-request field (capability and enable_thinking checks).
    ninfer::PromptCapabilities effort_capabilities;
    effort_capabilities.enable_thinking = true;
    effort_capabilities.reasoning_effort.low = true;
    GenerationRequest effort_request;
    effort_request.max_tokens = 1;
    failures += check(
        resolve_prompt_semantics(effort_request, template_kwargs, effort_capabilities)
            .reasoning_effort == ninfer::ReasoningEffort::Low,
        "server reasoning_effort default was not resolved for a request that omits it");
    effort_request.reasoning_effort = RequestedReasoningEffort::Medium;
    bool request_medium_rejected = false;
    try {
        (void)resolve_prompt_semantics(effort_request, template_kwargs, effort_capabilities);
    } catch (const ApiException&) { request_medium_rejected = true; }
    failures += check(request_medium_rejected,
                      "request reasoning_effort=medium was not rejected when unsupported");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--chat-template-kwargs") !=
                          std::string::npos,
                      "serve help omits --chat-template-kwargs");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    // --groups / --group-devices: multi-instance pooling surface. --devices names every rank of
    // every group in group order when --group-devices is omitted.
    const ServeOptions four_groups = parse({"ninfer-serve", "model.ninfer", "--groups", "4",
                                            "--tp", "2", "--devices",
                                            "0,1,2,3,4,5,6,7"});
    failures += check(four_groups.groups == 4 && four_groups.tp == 2 &&
                          four_groups.devices == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}) &&
                          four_groups.group_devices.empty(),
                      "explicit --groups/--tp/--devices did not form four groups");

    const ServeOptions two_groups = parse(
        {"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "2", "--devices", "0,1,2,3"});
    failures += check(two_groups.groups == 2 && two_groups.tp == 2 &&
                          two_groups.devices == std::vector<int>({0, 1, 2, 3}),
                      "two explicit groups did not accept one rank per device");

    const ServeOptions inferred = parse(
        {"ninfer-serve", "model.ninfer", "--group-devices", "0,1;2,3"});
    failures += check(inferred.groups == 2 && inferred.tp == 2 &&
                          inferred.devices == std::vector<int>({0, 1, 2, 3}) &&
                          inferred.group_devices == std::vector<int>({0, 1, 2, 3}),
                      "--group-devices did not infer groups/tp from its group lengths");

    bool wrong_device_count_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "2", "--devices",
                     "0,1,2"});
    } catch (const std::invalid_argument&) { wrong_device_count_rejected = true; }
    failures += check(wrong_device_count_rejected,
                      "multi-group device count that does not match groups * tp was accepted");

    bool overlapping_devices_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "2", "--devices",
                     "0,1,1,2"});
    } catch (const std::invalid_argument&) { overlapping_devices_rejected = true; }
    failures += check(overlapping_devices_rejected, "overlapping multi-group devices were accepted");

    bool zero_groups_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--groups", "0"});
    } catch (const std::invalid_argument&) { zero_groups_rejected = true; }
    failures += check(zero_groups_rejected, "--groups 0 was accepted");

    bool uneven_groups_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--group-devices", "0,1;2"});
    } catch (const std::invalid_argument&) { uneven_groups_rejected = true; }
    failures += check(uneven_groups_rejected, "uneven --group-devices group lengths were accepted");

    bool dflash_groups_rejected = false;
    bool dflash_groups_message  = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "1", "--devices",
                     "0,1", "--spec", "dflash", "--draft-tokens", "15"});
    } catch (const std::invalid_argument& error) {
        dflash_groups_rejected = true;
        dflash_groups_message =
            std::string(error.what()).find("DFlash 多组暂不支持") != std::string::npos;
    }
    failures += check(dflash_groups_rejected, "DFlash was accepted with multiple groups");
    failures += check(dflash_groups_message, "multi-group DFlash rejection lacks its contract text");

    bool device_conflict_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--device", "5", "--groups", "2", "--tp",
                     "2", "--devices", "0,1,2,3"});
    } catch (const std::invalid_argument&) { device_conflict_rejected = true; }
    failures += check(device_conflict_rejected,
                      "multi-group --device silently disagreed with --devices");

    const ServeOptions dflash_single =
        parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15"});
    failures += check(dflash_single.groups == 1 &&
                          dflash_single.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "single-group DFlash was rejected");

    // Multi-group lane and KV defaults are per instance, not process totals.
    const ServeOptions tp2_defaults =
        parse({"ninfer-serve", "model.ninfer", "--groups", "4", "--tp", "2", "--max-context",
               "262144"});
    failures += check(tp2_defaults.max_concurrency == 4 &&
                          tp2_defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          tp2_defaults.kv_capacity.explicit_tokens == 1048576,
                      "tp2 multi-group defaults are not per-instance lanes/KV capacity");

    const ServeOptions tp4_defaults =
        parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "4", "--max-context",
               "262144"});
    failures += check(tp4_defaults.max_concurrency == 8 &&
                          tp4_defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          tp4_defaults.kv_capacity.explicit_tokens == 2097152,
                      "tp4 multi-group defaults are not per-instance lanes/KV capacity");

    const ServeOptions tp1_defaults =
        parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "1", "--max-context",
               "262144"});
    failures += check(tp1_defaults.max_concurrency == 1 &&
                          tp1_defaults.kv_capacity.explicit_tokens == 262144,
                      "tp1 multi-group defaults are not per-instance lanes/KV capacity");

    const ServeOptions explicit_concurrency =
        parse({"ninfer-serve", "model.ninfer", "--groups", "4", "--tp", "2",
               "--max-concurrency", "4", "--max-context", "262144"});
    failures += check(explicit_concurrency.max_concurrency == 4 &&
                          explicit_concurrency.kv_capacity.explicit_tokens == 1048576,
                      "--max-concurrency did not stay a per-instance lane count");

    const ServeOptions explicit_kv =
        parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "2",
               "--max-concurrency", "4", "--kv-capacity", "4096", "--max-context", "2048"});
    failures += check(explicit_kv.max_concurrency == 4 &&
                          explicit_kv.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          explicit_kv.kv_capacity.explicit_tokens == 4096,
                      "explicit --kv-capacity was not kept as one per-instance budget");

    const ServeOptions pool_controls =
        parse({"ninfer-serve", "model.ninfer", "--groups", "2", "--tp", "2", "--devices",
               "0,1,2,3", "--global-pending-requests", "7", "--http-threads", "5",
               "--affinity-slack-tokens", "512", "--fairness-aging-ms", "250"});
    failures += check(pool_controls.global_pending_requests == 7 &&
                          pool_controls.http_threads == 5 &&
                          pool_controls.affinity_slack_tokens == 512 &&
                          pool_controls.fairness_aging_ms == 250,
                      "pool/HTTP scheduling options did not reach serving options");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
