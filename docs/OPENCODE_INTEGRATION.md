# OpenCode Integration Guide

Connect [OpenCode](https://opencode.ai/) — an open-source terminal AI coding agent — to Flash-MoE's local Qwen 3.5-397B inference engine.

## Prerequisites

- Flash-MoE built and working (`cd metal_infer && make`)
- Model weights in place (`model_weights.bin`, `packed_experts/`)
- [OpenCode](https://opencode.ai/) installed: `npm i -g opencode-ai`

## Quick Start

```bash
# Terminal 1: Start Flash-MoE server
cd metal_infer
./infer --serve 8000

# Terminal 2: Run OpenCode from the repo root (picks up opencode.json)
cd flash-moe
opencode
```

That's it. The included `opencode.json` configures OpenCode to use the local server.

## How It Works

```
┌─────────────┐     HTTP/SSE      ┌─────────────────────────────┐
│  OpenCode    │ ◄──────────────► │  Flash-MoE Server            │
│  (terminal)  │                  │  ./infer --serve 8000        │
│              │  /v1/chat/       │                               │
│  @ai-sdk/    │  completions     │  • Full messages[] parser    │
│  openai-     │                  │  • tools[] → system prompt   │
│  compatible  │                  │  • <tool_call> → OpenAI      │
│              │                  │    tool_calls[] converter    │
└─────────────┘                  └─────────────────────────────┘
```

OpenCode sends standard OpenAI Chat Completions API requests. The Flash-MoE server:

1. **Parses the full `messages[]` array** — system, user, assistant, and tool messages are converted to Qwen3's ChatML format
2. **Injects tool definitions** — the `tools[]` parameter is parsed and appended to the system prompt so the model knows what tools are available
3. **Converts tool call output** — when the model outputs `<tool_call>` XML blocks, they're converted to OpenAI-format `delta.tool_calls[]` SSE chunks that OpenCode understands
4. **Streams via SSE** — tokens are streamed as Server-Sent Events in real-time

## Configuration

The included `opencode.json` in the repo root:

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

### Customizing

- **Port**: Change `8000` in both `--serve` flag and `baseURL`
- **Context window**: Adjust `limit.context` (max 32768 for Qwen3.5)
- **Output tokens**: Adjust `limit.output` (server caps at 32768)
- **System prompt**: Create `~/.flash-moe/system.md` to customize the system prompt for the legacy `chat.m` client. OpenCode sends its own system prompt.

### Global Config

To use Flash-MoE from any directory, copy the provider config to `~/.config/opencode/opencode.json`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with SSE streaming |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Example curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a hello world in Python"}
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

### With tools

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What files are in /tmp?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "The command to run"}
          },
          "required": ["command"]
        }
      }
    }],
    "max_tokens": 1024,
    "stream": true
  }'
```

## Architecture Notes

- **Stateless mode**: OpenCode sends the full conversation history on every request. The server resets all KV cache and attention state, then processes the complete ChatML prompt from scratch. This is slower than session continuation but correct for stateless clients.
- **Legacy compatibility**: The existing `chat.m` client with `session_id`-based continuation still works unchanged. The server auto-detects which mode to use.
- **Tool call format**: The model natively outputs `<tool_call>{"name":"...","arguments":{...}}</tool_call>`. A streaming state machine in the server converts these to OpenAI-format `delta.tool_calls[]` SSE chunks.
- **Thinking**: The model's `<think>` blocks are streamed as regular content. OpenCode displays them as-is.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OpenCode can't connect | Verify server is running: `curl http://localhost:8000/health` |
| "no content in messages" error | Ensure messages array has at least one message with content |
| Tool calls not working | Verify `tools` array is included in the request (OpenCode does this automatically) |
| Slow first response | Full messages mode re-encodes the entire conversation. This is expected; prefill takes ~230ms per token |
| Model outputs `<tool_call>` as text | Check that tools were sent in the request. Without tools, the server passes tokens through as-is |

## Technical Details

For the full design document including gap analysis, component design, and implementation plan, see [PLAN_OPENCODE_INTEGRATION.md](PLAN_OPENCODE_INTEGRATION.md).
