# Plan & SDD: Connecting OpenCode to Flash-MoE (Qwen 3.5-397B)

## Executive Summary

Connect [OpenCode](https://opencode.ai/) — an open-source terminal AI coding agent — to the locally running Qwen 3.5-397B-A17B model served by Flash-MoE's built-in HTTP server. OpenCode expects an OpenAI-compatible `/v1/chat/completions` endpoint with **tool/function calling** in the OpenAI JSON format. Flash-MoE already serves this endpoint but has **three critical gaps** that must be bridged.

---

## Current State Analysis

### What Already Works

| Feature | Status | Location |
|---------|--------|----------|
| HTTP server (`--serve <port>`) | Working | `infer.m:5965` |
| `POST /v1/chat/completions` | Working | `infer.m:6181` |
| `GET /v1/models` | Working | `infer.m:6166` |
| `GET /health` | Working | `infer.m:6152` |
| SSE streaming (`text/event-stream`) | Working | `infer.m:5803` |
| CORS headers | Working | `infer.m:5845` |
| Session persistence (KV cache reuse) | Working | `infer.m:6123` |
| System prompt caching | Working | `infer.m:6000` |
| `max_tokens` / `max_completion_tokens` | Working | `infer.m:5725` |

### What's Missing (Critical Gaps)

#### Gap 1: Multi-message Conversation Handling

**Problem:** `extract_last_content()` only extracts the **last** `"content"` value from the messages array. OpenCode sends the **full conversation history** on every request (system + user + assistant + tool results). The server ignores all prior context and only sees the last message.

**Impact:** The server has no way to process multi-turn conversations sent by OpenCode. Since the server manages its own KV cache with a session continuation model, it needs to either:
- (a) Parse the full messages array and build the proper ChatML prompt, OR
- (b) Rely on session_id continuations (but OpenCode doesn't send `session_id`)

**Solution:** Implement full messages array parsing. Build the ChatML prompt from the entire conversation history, including system, user, assistant, and tool messages.

#### Gap 2: OpenAI Tool Calling Format

**Problem:** The model natively outputs tool calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>` XML blocks. OpenCode expects the OpenAI API format where tool calls are structured in `delta.tool_calls[]` with `id`, `type`, `function.name`, and `function.arguments`.

**Impact:** OpenCode's agentic features (file editing, shell execution, code search) won't work because it can't parse tool invocations from raw content.

**Solution:** Detect `<tool_call>` blocks in the generated output and convert them to OpenAI-format `tool_calls` chunks in the SSE stream. Also parse incoming `tool` role messages and format them as `<tool_response>` blocks in the ChatML prompt.

#### Gap 3: `tools` Parameter Parsing

**Problem:** OpenCode sends a `tools` array in the request body defining available functions (JSON Schema). The server currently ignores this entirely. The model needs the tool definitions injected into the system prompt to know what tools are available.

**Impact:** Without tool definitions in context, the model won't know it can call tools or what schema to use.

**Solution:** Parse the `tools` array from the request body and inject tool definitions into the system prompt using Qwen's expected format.

---

## Software Design Document (SDD)

### 1. Architecture Overview

```
┌─────────────┐     HTTP/SSE      ┌──────────────────────────────────┐
│  OpenCode    │ ◄──────────────► │  Flash-MoE Server (infer.m)      │
│  (terminal)  │                  │                                    │
│              │  /v1/chat/       │  ┌────────────────────────────┐   │
│  @ai-sdk/    │  completions     │  │  NEW: OpenAI Compat Layer  │   │
│  openai-     │                  │  │                            │   │
│  compatible  │                  │  │  • Full messages[] parser  │   │
│              │                  │  │  • tools[] → system prompt │   │
│              │                  │  │  • <tool_call> → delta     │   │
│              │                  │  │    tool_calls[] converter  │   │
│              │                  │  └────────────────────────────┘   │
│              │                  │                │                   │
│              │                  │  ┌─────────────▼──────────────┐   │
│              │                  │  │  Existing Inference Engine │   │
│              │                  │  │  (tokenizer → model → gen) │   │
│              │                  │  └────────────────────────────┘   │
└─────────────┘                  └──────────────────────────────────┘
```

### 2. Component Design

#### 2.1 Full Messages Array Parser

**Function:** `build_chatml_from_messages(char *json_body) → char *chatml_prompt`

Parses the OpenAI `messages` array and builds a ChatML-formatted prompt string:

```
Input (OpenAI format):
{
  "messages": [
    {"role": "system", "content": "You are a coding assistant"},
    {"role": "user", "content": "List files in /tmp"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls /tmp\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt\nfile2.txt"}
    {"role": "user", "content": "Now delete file1.txt"}
  ]
}

Output (ChatML):
<|im_start|>system
You are a coding assistant

# Tools
...tool definitions...
<|im_end|>
<|im_start|>user
List files in /tmp<|im_end|>
<|im_start|>assistant
<tool_call>{"name":"bash","arguments":{"command":"ls /tmp"}}</tool_call><|im_end|>
<|im_start|>user
<tool_response>
file1.txt
file2.txt
</tool_response><|im_end|>
<|im_start|>user
Now delete file1.txt<|im_end|>
<|im_start|>assistant
```

**Key behaviors:**
- System message → `<|im_start|>system\n{content}<|im_end|>`
- User message → `<|im_start|>user\n{content}<|im_end|>`
- Assistant message → `<|im_start|>assistant\n{content}<|im_end|>`
- Assistant with tool_calls → Convert to `<tool_call>` XML format
- Tool result → `<|im_start|>user\n<tool_response>{content}</tool_response><|im_end|>`
- Always append `\n<|im_start|>assistant\n` at end to prompt generation

**JSON parsing approach:** Minimal state-machine JSON parser (no external library). Walk the messages array, extract role/content/tool_calls/tool_call_id fields. This follows the existing codebase pattern of inline JSON parsing.

#### 2.2 Tools Array → System Prompt Injection

**Function:** `extract_tools_for_prompt(char *json_body) → char *tools_block`

Parses the `"tools"` array from the request and formats it for injection into the system prompt using Qwen3's expected tool format:

```
# Tools

You are provided with the following tools. Call them by outputting <tool_call>...</tool_call>.

## bash

Execute a bash command.

Parameters:
- command (string, required): The command to execute

## read_file

Read a file from the filesystem.

Parameters:
- path (string, required): The file path to read
```

This block is appended to the system message content before building ChatML.

#### 2.3 Tool Call Output Converter (SSE Transform)

**Function:** `sse_send_delta_with_tool_detection(fd, request_id, token, &tool_state) → int`

Wraps `sse_send_delta()` with a streaming state machine that detects `<tool_call>` blocks and converts them:

**States:**
1. `NORMAL` — Pass tokens through as `delta.content`
2. `TOOL_TAG_DETECTING` — Accumulating chars that might be `<tool_call>`
3. `IN_TOOL_CALL` — Inside `<tool_call>...</tool_call>`, buffering JSON
4. `TOOL_TAG_CLOSING` — Detecting `</tool_call>`

**When a complete `<tool_call>{...}</tool_call>` is detected:**

Send SSE chunks in OpenAI tool_calls format:
```json
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"bash","arguments":""}}]}}]}
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"ls\"}"}}]}}]}
```

Then send `finish_reason: "tool_calls"` instead of `"stop"`.

#### 2.4 Tokenization Path Change

Currently the server has two paths:
- `tokenize_user_turn()` — For new sessions (system prompt in snapshot)
- `tokenize_continuation_turn()` — For continuing sessions

**New path:** When the request contains a full messages array (detected by presence of `"role"` in messages), use:
- `tokenize_full_chatml(chatml_prompt)` — Tokenizes the entire ChatML string
- **Bypass** the snapshot restore (since the full conversation is re-encoded)
- This is less efficient than session continuation but is the correct approach for stateless API clients like OpenCode

**Optimization (Phase 2):** Implement prompt prefix caching — hash the messages array minus the last message, and if the hash matches the previous request, use continuation mode instead of full re-encode.

### 3. OpenCode Configuration

Create the config file at `~/.config/opencode/opencode.json` (or project-local `opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "flash-moe": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Flash-MoE (Qwen 3.5-397B)",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-local"
      },
      "models": {
        "qwen3.5-397b-a17b": {
          "name": "Qwen 3.5 397B-A17B (4-bit, local)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "flash-moe/qwen3.5-397b-a17b"
}
```

### 4. System Prompt for Coding

Create `~/.flash-moe/system.md` with a coding-optimized system prompt that includes thinking mode and tool-use instructions compatible with how OpenCode structures its requests.

---

## Implementation Plan

### Phase 1: Core API Compatibility (Required — MVP)

| Step | Task | Estimated Complexity | Files Modified |
|------|------|---------------------|----------------|
| 1.1 | Implement full messages array JSON parser | Medium | `infer.m` |
| 1.2 | Build ChatML prompt from parsed messages | Medium | `infer.m` |
| 1.3 | Parse `tools` array and inject into system prompt | Medium | `infer.m` |
| 1.4 | Replace `extract_last_content()` path with full messages parser when messages array detected | Low | `infer.m` |
| 1.5 | Add `tokenize_full_chatml()` path that encodes arbitrary ChatML string | Low | `infer.m` |
| 1.6 | Stream tool call detection + OpenAI format conversion | High | `infer.m` |
| 1.7 | Handle `tool` role messages in input (tool results) | Medium | `infer.m` |
| 1.8 | Add `finish_reason: "tool_calls"` when tool calls detected | Low | `infer.m` |
| 1.9 | Create OpenCode config file | Low | `opencode.json` |
| 1.10 | Test end-to-end with OpenCode | — | — |

### Phase 2: Optimization (Optional — Performance)

| Step | Task | Description |
|------|------|-------------|
| 2.1 | Prompt prefix caching | Hash messages[0..n-1], reuse KV cache if same prefix |
| 2.2 | Non-streaming response mode | Return full JSON response when `stream: false` |
| 2.3 | Usage stats in response | Return `prompt_tokens`, `completion_tokens` in response |

### Phase 3: Robustness (Optional — Production)

| Step | Task | Description |
|------|------|-------------|
| 3.1 | Temperature/top_p sampling | Currently uses argmax only; add sampling support |
| 3.2 | Stop sequences | Parse `stop` parameter and halt generation |
| 3.3 | Request timeout handling | Kill generation if client disconnects |
| 3.4 | Concurrent request queuing | Queue requests instead of blocking |

---

## Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| JSON parsing bugs (no library) | High | Thorough testing with real OpenCode requests; consider using a minimal JSON parser like cJSON |
| Tool call detection false positives | Medium | Only activate tool parsing when `tools` array present in request |
| Full re-encode on every request is slow | Medium | Phase 2 prefix caching; initial latency acceptable at ~4 tok/s |
| Context window overflow | Medium | Truncate old messages; Qwen3.5 supports 32K context |
| Model outputs malformed tool JSON | Low | Already works in chat.m; 4-bit quantization handles JSON well |
| Thinking tokens leaked to OpenCode | Medium | Filter `<think>` blocks from SSE output (already filtered server-side) |

---

## Testing Strategy

1. **Unit test the messages parser** — Feed sample OpenCode request bodies, verify ChatML output
2. **Curl test** — Send OpenAI-format requests with tools to the server, verify SSE response format
3. **OpenCode smoke test** — Run `opencode` with the config, verify:
   - Model responds to basic prompts
   - File reading tool works
   - File editing tool works
   - Shell command execution works
4. **Multi-turn test** — Verify conversation context is maintained across turns

---

## Quick Start (After Implementation)

```bash
# Terminal 1: Start Flash-MoE server
cd metal_infer
./infer --serve 8000

# Terminal 2: Create OpenCode config and run
cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "flash-moe": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Flash-MoE (Qwen 3.5-397B)",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-local"
      },
      "models": {
        "qwen3.5-397b-a17b": {
          "name": "Qwen 3.5 397B-A17B (4-bit, local)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "flash-moe/qwen3.5-397b-a17b"
}
EOF

npx opencode-ai
```
