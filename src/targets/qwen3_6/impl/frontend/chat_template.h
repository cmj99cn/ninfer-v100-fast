#pragma once

#include "frontend/chat_template.h"

// The Qwen chat-template compiler and its wire-independent conversation types are shared with the
// Flash-Next family. Qwen3.6 keeps this internal alias so its processor/media state stays in the
// family namespace while the compiler itself has a single implementation.
namespace ninfer::targets::qwen3_6::frontend_internal {

using ninfer::qwen_frontend::ChatMessage;
using ninfer::qwen_frontend::ChatPart;
using ninfer::qwen_frontend::ChatPartKind;
using ninfer::qwen_frontend::ChatRenderOptions;
using ninfer::qwen_frontend::ChatTemplateSemantics;
using ninfer::qwen_frontend::CompiledChatTemplate;
using ninfer::qwen_frontend::MediaData;
using ninfer::qwen_frontend::RenderedChat;
using ninfer::qwen_frontend::RewriteCheckpointByteSpec;
using ninfer::qwen_frontend::RewriteCheckpointKind;
using ninfer::qwen_frontend::ToolCall;

} // namespace ninfer::targets::qwen3_6::frontend_internal
