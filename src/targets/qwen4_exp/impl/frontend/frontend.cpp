#include <ninfer/targets/qwen4_exp/frontend.h>

#include "frontend/chat_template.h"
#include "frontend/tokenizer.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4_exp {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = ninfer::qwen_frontend;

constexpr std::size_t kTokenDomain = 248077;
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kReplacementCharacter = "\xef\xbf\xbd";
constexpr std::string_view kToolCallOpen = "<tool_call>";

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json config = parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (config.value("add_bos_token", true) || config.value("add_prefix_space", true)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen byte-level prefix semantics");
    }
    if (!config.contains("pad_token") || !config.at("pad_token").is_string() ||
        config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
    // The artifact's chat_template.jinja is the NInfer-adapted Flash-Next template. The upstream
    // tokenizer_config.json still embeds the unmodified official template; both are valid resources
    // and the frontend renders from the artifact template, so equality is not required here.
    if (!config.contains("chat_template") || !config.at("chat_template").is_string()) {
        throw std::invalid_argument(
            "tokenizer_config.json.chat_template must contain the loaded chat template");
    }
}

void validate_registered_tokenizer(const fi::Tokenizer& tokenizer) {
    if (!tokenizer.has_exact_token_domain(kTokenDomain)) {
        throw std::invalid_argument(
            "artifact tokenizer does not expose the registered 248077-token domain");
    }
    constexpr std::array<std::pair<std::string_view, TokenId>, 6> kStructuralTokens = {{
        {"<|im_start|>", 248045},
        {"<|im_end|>", 248046},
        {"<|vision_start|>", 248053},
        {"<|image_pad|>", 248056},
        {"<think>", 248068},
        {"</think>", 248069},
    }};
    for (const auto& [text, expected] : kStructuralTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected) {
            throw std::invalid_argument(
                "artifact tokenizer does not match registered structural token IDs");
        }
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources) {
    validate_tokenizer_config(resources);
    return fi::CompiledChatTemplate::resolve(resources.chat_template_jinja);
}

void check_preparation_control(const PreparationControl& control, std::string_view stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = source.role;
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            if (part.kind == MessagePartKind::Text) {
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                continue;
            }
            if (part.media.bytes.empty()) {
                throw std::invalid_argument("frontend media input contains no owning bytes");
            }
            fi::MediaData media;
            media.source_name = std::move(part.media.source_name);
            media.media_type  = std::move(part.media.media_type);
            media.bytes       = std::move(part.media.bytes);
            target.parts.push_back(part.media.kind == MediaKind::Image
                                       ? fi::ChatPart::image(std::move(media))
                                       : fi::ChatPart::video(std::move(media)));
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options) {
    return fi::ChatRenderOptions{.add_generation_prompt = options.add_generation_prompt,
                                 .enable_thinking       = options.enable_thinking,
                                 .reasoning_effort      = options.reasoning_effort,
                                 .preserve_thinking     = options.preserve_thinking,
                                 .add_vision_id         = options.add_vision_id,
                                 .tool_jsons            = options.tool_jsons};
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) { throw std::invalid_argument("prepared prompt must contain at least one token"); }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(std::vector<OutputDelta>& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

// Incremental UTF-8 decoding of a byte-level token stream. This scan is TOTAL: a trailing
// incomplete prefix is held, and bytes that cannot begin a code point become U+FFFD following the
// Unicode 15 maximal-subpart rule. It must never throw on the Engine worker thread.
std::size_t decode_utf8_prefix(std::string_view bytes, std::string& out) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto lead    = static_cast<unsigned char>(bytes[offset]);
        std::size_t length = 0;
        unsigned char low  = 0x80U;
        unsigned char high = 0xbfU;
        if (lead <= 0x7fU) {
            out.push_back(static_cast<char>(lead));
            ++offset;
            continue;
        }
        if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
            if (lead == 0xe0U) { low = 0xa0U; }
            if (lead == 0xedU) { high = 0x9fU; }
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
            if (lead == 0xf0U) { low = 0x90U; }
            if (lead == 0xf4U) { high = 0x8fU; }
        } else {
            out.append(kReplacementCharacter);
            ++offset;
            continue;
        }

        std::size_t index = 1;
        bool well_formed  = true;
        for (; index < length && offset + index < bytes.size(); ++index) {
            const auto byte              = static_cast<unsigned char>(bytes[offset + index]);
            const unsigned char minimum  = index == 1 ? low : 0x80U;
            const unsigned char maximum  = index == 1 ? high : 0xbfU;
            if (byte < minimum || byte > maximum) {
                well_formed = false;
                break;
            }
        }
        if (!well_formed) {
            out.append(kReplacementCharacter);
            offset += index;
            continue;
        }
        if (offset + length > bytes.size()) { return offset; }
        out.append(bytes.substr(offset, length));
        offset += length;
    }
    return offset;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning              = false;
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
};

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    std::vector<OutputDelta> output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, std::vector<OutputDelta>& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            std::vector<OutputDelta> candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, std::vector<OutputDelta>& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  std::vector<OutputDelta>& emitted, std::uint32_t committed_tokens,
                  StopMatch* best_match) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens, best_match);
}

bool is_line_boundary(char c) { return c == '\n' || c == '\r'; }

enum class ToolCallPrefix { No, Partial, Yes };

ToolCallPrefix tool_call_prefix(std::string_view text, std::size_t pos) {
    const std::string_view tail = text.substr(pos);
    if (tail.size() >= kToolCallOpen.size()) {
        return tail.starts_with(kToolCallOpen) ? ToolCallPrefix::Yes : ToolCallPrefix::No;
    }
    return kToolCallOpen.starts_with(tail) ? ToolCallPrefix::Partial : ToolCallPrefix::No;
}

// `</think>` closes reasoning only on its structural form: a blank line then the answer, or a tool
// call after at most one line boundary. A quoted copy inside code is published on the reasoning
// channel instead. A deciding byte still in flight holds the tag for the next feed.
void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       std::vector<OutputDelta>& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match) {
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match);
        return;
    }

    state.think_marker_pending.append(text);
    for (;;) {
        const std::size_t marker = state.think_marker_pending.find(kThinkClose);
        if (marker == std::string::npos) { break; }
        const std::size_t end            = marker + kThinkClose.size();
        const std::string_view pending   = state.think_marker_pending;

        bool close = false;
        bool wait  = false;
        if (end >= pending.size()) {
            wait = true;
        } else if (!is_line_boundary(pending[end])) {
            switch (tool_call_prefix(pending, end)) {
            case ToolCallPrefix::Yes: close = true; break;
            case ToolCallPrefix::Partial: wait = true; break;
            case ToolCallPrefix::No: break;
            }
        } else if (end + 1 >= pending.size()) {
            wait = true;
        } else if (is_line_boundary(pending[end + 1])) {
            close = true;
        } else {
            switch (tool_call_prefix(pending, end + 1)) {
            case ToolCallPrefix::Yes: close = true; break;
            case ToolCallPrefix::Partial: wait = true; break;
            case ToolCallPrefix::No: break;
            }
        }

        if (wait) {
            feed_channel(state, OutputChannel::Reasoning, pending.substr(0, marker), policy, emitted,
                         committed_tokens, best_match);
            state.think_marker_pending.erase(0, marker);
            return;
        }
        if (!close) {
            feed_channel(state, OutputChannel::Reasoning, pending.substr(0, end), policy, emitted,
                         committed_tokens, best_match);
            state.think_marker_pending.erase(0, end);
            continue;
        }
        feed_channel(state, OutputChannel::Reasoning, pending.substr(0, marker), policy, emitted,
                     committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(end);
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = true;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match);
        return;
    }

    const std::size_t hold = longest_suffix_prefix(state.think_marker_pending, kThinkClose, true);
    const std::size_t safe = state.think_marker_pending.size() - hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string bytes, const StopPolicy& policy,
                      std::vector<OutputDelta>& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match) {
    state.utf8_pending += bytes;
    std::string text;
    const std::size_t resolved = decode_utf8_prefix(state.utf8_pending, text);
    state.utf8_pending.erase(0, resolved);
    if (text.empty()) { return; }
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match);
}

void terminalize(DecoderState& state, const StopPolicy& policy, std::vector<OutputDelta>& emitted,
                 std::uint32_t committed_tokens) {
    if (!state.utf8_pending.empty()) {
        state.utf8_pending.clear();
        feed_decoded_text(state, kReplacementCharacter, policy, emitted, committed_tokens, nullptr);
    }
    if (state.in_reasoning) {
        const std::string_view pending = state.think_marker_pending;
        const bool held_close =
            pending.starts_with(kThinkClose) &&
            std::all_of(pending.begin() + static_cast<std::ptrdiff_t>(kThinkClose.size()),
                        pending.end(), is_line_boundary);
        if (!held_close) {
            feed_channel(state, OutputChannel::Reasoning, state.think_marker_pending, policy,
                         emitted, committed_tokens, nullptr);
        }
        state.think_marker_pending.clear();
        close_channel(state, OutputChannel::Reasoning, emitted);
    } else {
        close_channel(state, OutputChannel::Content, emitted);
    }
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, FrontendOptions options) {
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        // A token-only frontend has no artifact resources: the bench/NLL routes go through
        // prepare_tokens() and must stay usable without a compiled template or tokenizer.
        if (resources.tokenizer_json.empty()) { return; }
        chat_template = compile_chat_template(resources);
        tokenizer = std::make_shared<const fi::Tokenizer>(
            fi::TokenizerResources{.tokenizer_json         = resources.tokenizer_json,
                                   .tokenizer_config_json  = resources.tokenizer_config_json,
                                   .generation_config_json = resources.generation_config_json});
        validate_registered_tokenizer(*tokenizer);
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
    }

    std::optional<fi::CompiledChatTemplate> chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy defaults;
};

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_,
         const OutputOptions& output, bool starts_in_reasoning)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          preserve_special(output.raw || output.preserve_special_tokens) {
        state.in_reasoning = starts_in_reasoning && !output.raw;
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    bool preserve_special = false;
    DecoderState state;
    DecoderState preview_state;
    std::vector<OutputDelta> preview_output;
    bool preview_ready = false;
};

PreparedPrompt::PreparedPrompt() noexcept = default;
PreparedPrompt::~PreparedPrompt()         = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PromptSummary PreparedPrompt::summary() const {
    PromptSummary summary;
    if (data_ != nullptr) {
        summary.prompt_tokens = static_cast<std::uint32_t>(data_->token_ids.size());
    }
    return summary;
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept { return {}; }

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

OutputSession::OutputSession() noexcept = default;
OutputSession::~OutputSession()         = default;
OutputSession::OutputSession(OutputSession&&) noexcept = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview(std::span<const TokenId> tokens,
                                               std::uint32_t budget_remaining,
                                               FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (tokens.empty()) { throw std::invalid_argument("cannot preview an empty generated-token round"); }
    if (tokens.size() > budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit && limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state  = impl_->state;
    impl_->preview_output.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason) {
        impl_->preview_ready = true;
        return runtime::OutputDecision{.accepted_tokens = count, .finish_reason = reason};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count = static_cast<std::uint32_t>(index + 1);
        const TokenId token       = tokens[index];
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("generated token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }

        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        DecoderState before_state;
        std::vector<OutputDelta> before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const std::string bytes =
            impl_->tokenizer->decode_token_bytes(token, !impl_->preserve_special);
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, impl_->preview_output, count,
                         &match);

        if (match.found) {
            impl_->preview_state  = terminal_state(std::move(impl_->preview_state));
            impl_->preview_output = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            return complete(count, FinishReason::StopToken);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        return complete(count, limit_reason);
    }
    return complete(count, FinishReason::None);
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state = impl_->state;
    impl_->preview_output.clear();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

std::vector<OutputDelta> OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    std::vector<OutputDelta> output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
    return output;
}

std::uint32_t OutputSession::reasoning_tokens() const {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

Frontend::Frontend() noexcept = default;

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr || !impl_->tokenizer) {
        throw std::logic_error("qwen4_exp frontend has no loaded chat resources");
    }
    check_preparation_control(control, "prompt preparation");
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media) {
        throw std::invalid_argument("qwen4_exp family frontend does not prepare media input");
    }

    const fi::RenderedChat rendered = impl_->chat_template->render(messages, render_options(options));
    check_preparation_control(control, "tokenization");
    auto data = std::make_unique<PreparedPromptData>();
    data->token_ids = impl_->tokenizer->encode(rendered.text);
    checked_token_count(data->token_ids.size());
    data->block_hashes = ninfer::prefix_cache::chain_block_hashes(data->token_ids);
    data->starts_in_reasoning = options.add_generation_prompt && options.enable_thinking;
    return PreparedPrompt(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr || !impl_->tokenizer) {
        throw std::logic_error("qwen4_exp frontend has no loaded chat resources");
    }
    check_preparation_control(control, "prompt preparation");
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    for (const fi::ChatMessage& message : messages) {
        if (message.has_media()) {
            throw std::invalid_argument("qwen4_exp family frontend does not count media input");
        }
    }
    const fi::RenderedChat rendered = impl_->chat_template->render(messages, render_options(options));
    check_preparation_control(control, "tokenization");
    return checked_token_count(impl_->tokenizer->encode(rendered.text).size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    (void)allow_prefix_identity;
    auto data = std::make_unique<PreparedPromptData>();
    data->token_ids = std::move(token_ids);
    data->block_hashes = ninfer::prefix_cache::chain_block_hashes(data->token_ids);
    return PreparedPrompt(std::move(data));
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    if (impl_ == nullptr || !impl_->chat_template) { return {}; }
    return impl_->chat_template->capabilities();
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    if (impl_ == nullptr || !impl_->tokenizer) {
        throw std::logic_error("qwen4_exp frontend has no loaded chat resources");
    }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, prompt.data_->starts_in_reasoning));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    static const StopPolicy empty;
    return impl_ != nullptr && impl_->tokenizer ? impl_->defaults : empty;
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("PreparedPrompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("PreparedPrompt is empty"); }
    return std::move(*prompt.data_);
}

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

namespace {

artifact::ObjectHandle require_component_resource(artifact::Binder& binder,
                                                  const std::string& component,
                                                  const std::string& role) {
    const auto& resources = binder.directory().component_resources;
    const auto component_it = resources.find(component);
    if (component_it == resources.end()) {
        throw std::invalid_argument("the artifact has no frontend component resources for " + component);
    }
    const auto role_it = component_it->second.find(role);
    if (role_it == component_it->second.end()) {
        throw std::invalid_argument("the artifact has no frontend resource role " + role);
    }
    const artifact::ObjectHandle handle =
        binder.require_resource(role_it->second, artifact::ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

std::string take_string(artifact::MaterializedArtifact& materialized, artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json           = require_component_resource(binder, "text", "tokenizer.json"),
        .tokenizer_config_json    = require_component_resource(binder, "text", "tokenizer_config.json"),
        .chat_template_jinja      = require_component_resource(binder, "text", "chat_template.jinja"),
        .generation_config_json   = require_component_resource(binder, "text", "generation_config.json"),
        .preprocessor_config_json = require_component_resource(binder, "vision", "preprocessor_config.json"),
        .video_preprocessor_config_json =
            require_component_resource(binder, "vision", "video_preprocessor_config.json"),
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = take_string(materialized, plan.preprocessor_config_json),
        .video_preprocessor_config_json =
            take_string(materialized, plan.video_preprocessor_config_json),
    };
}

} // namespace ninfer::targets::qwen4_exp
