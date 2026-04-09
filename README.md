<p align="center">
  <img src="assets/logo.svg" alt="Luna Agent" width="120" height="120">
</p>

<h1 align="center">Luna Agent</h1>

<p align="center">
  A custom minimal AI agent with persistent memory, MCP tool integration, Discord/CLI interface, and structured observability.<br>
  Runs entirely on local hardware — no cloud API costs.
</p>

<p align="center">
  <a href="https://github.com/nonatofabio/luna-agent/actions/workflows/tests.yml"><img src="https://github.com/nonatofabio/luna-agent/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
</p>

---

~2300 lines of Python. No frameworks.

## Why Custom

We evaluated existing agent frameworks and rejected them all:

- **OpenClaw**: 400K lines of code, 42K exposed instances on Shodan. Too large to audit, too large to trust.
- **ZeroClaw**: 9 days old at time of evaluation. Too immature.
- **NanoClaw**: Too thin — would need to rebuild most of it anyway.

The core needs (memory, tools, chat, logging) are individually well-solved problems. No 400K-line framework needed.

## Architecture

```
Discord (discord.py)     CLI REPL (no token)
     |                        |
     v                        v
+---------------------------------+
|        Luna Agent Core          |
|                                 |
|  agent.py                       |  agent loop: msg → memory → prompt → LLM → tools → respond
|    ├── llm.py                   |  single LLM client, configurable endpoint
|    ├── memory.py                |  SQLite + FTS5 + sqlite-vec hybrid search
|    ├── tools.py                 |  native tools: bash, files, web, delegate, code_task
|    ├── tool_output.py           |  smart output pipeline for large results
|    ├── mcp_manager.py           |  MCP client for community tool servers
|    └── observe.py               |  structured JSON logging
|                                 |
+---------------------------------+
              |
              v
        llama-server               Qwen3.5-35B-A3B on 2x RTX 3090
```

All LLM traffic flows through a single `LLMClient` with a configurable endpoint URL. Today it points at `localhost:8001` (llama-server). To insert an AI firewall later, change the URL to `localhost:9000` — zero code changes required.

**Thinking model support:** Luna handles reasoning models (Qwen3.5, etc.) automatically — extracting `reasoning_content`, falling back to cleaned reasoning when content is empty, and stripping leaked markup (`<thinking>`, `<tool_call>`, etc.) from output.

## Hardware

- Intel i7-13700K, 64GB DDR4
- 2x NVIDIA RTX 3090 (24GB each, 48GB total)
- Qwen3.5-35B-A3B Q8_0 via llama-server with layer split across both GPUs
- 131K context window, Q8_0 KV cache

## Quick Start

```bash
cd ~/luna-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run without Discord (interactive CLI REPL)
python -m luna

# Run with Discord
DISCORD_TOKEN=your-token-here python -m luna
```

## Project Structure

```
luna-agent/
├── config.toml              # All configuration
├── mcp_servers.json         # MCP server registry
├── pyproject.toml           # Dependencies
├── luna/
│   ├── __main__.py          # Entry point (python -m luna)
│   ├── agent.py             # Core agent loop
│   ├── llm.py               # LLM client (OpenAI-compatible)
│   ├── memory.py            # Memory (SQLite + FTS5 + sqlite-vec)
│   ├── tools.py             # Native tools (bash, files, web)
│   ├── tool_output.py       # Large output persistence + filtering
│   ├── mcp_manager.py       # MCP tool client
│   ├── discord_bot.py       # Discord interface
│   ├── observe.py           # Structured JSON logging
│   └── config.py            # Config loader
├── tests/
│   ├── test_agent.py        # Agent loop tests
│   ├── test_llm.py          # LLM client tests
│   ├── test_memory.py       # Memory system tests
│   ├── test_tools.py        # Native tool tests
│   └── test_tool_output.py  # Output pipeline tests
├── luna-agent.service        # systemd unit for the agent
├── worker-agent.service      # systemd unit for llama-server (Qwen3.5-35B-A3B)
└── data/                     # Created at runtime
    ├── memory.db             # SQLite database
    ├── logs/                 # JSON log files
    │   └── luna-YYYY-MM-DD.jsonl
    └── tool_outputs/         # Persisted large tool outputs
```

## Configuration

All settings live in `config.toml`. Environment variables override for secrets:

| Env Var | Overrides | Required |
|---------|-----------|----------|
| `DISCORD_TOKEN` | Discord bot token | Yes (for Discord) |
| `LLM_ENDPOINT` | `[llm] endpoint` | No |
| `LLM_MODEL` | `[llm] model` | No |
| `MEMORY_DB_PATH` | `[memory] db_path` | No |
| `LOG_DIR` | `[observe] log_dir` | No |
| `WEB_SEARCH_PROVIDER` | Web search backend (`duckduckgo` or `youcom`) | No |
| `YOUCOM_SEARCH_API_KEY` | API key for You.com web search (`X-API-Key`) | No |

See `config.toml` for all available settings and their defaults.

### Web search provider setup

By default, `web_search` uses DuckDuckGo. To use You.com Search API:

```bash
export WEB_SEARCH_PROVIDER=youcom
export YOUCOM_SEARCH_API_KEY=your_api_key_here
```

Fallback behavior: if You.com is selected but the API key is missing (or the API request fails), Luna falls back to DuckDuckGo and annotates the tool result.

## Components

### Agent (`agent.py`)

The orchestrator. Receives a message and session ID, then:

1. Saves the user message to memory
2. Searches for relevant memories (hybrid FTS + vector)
3. Retrieves the session summary (if any)
4. Builds a system prompt with memories, summary, and current time
5. Loads the last 20 messages for context
6. Calls the LLM with all available tools (native + MCP)
7. Enters a tool call loop (max 25 rounds):
   - Executes each tool call (native or MCP)
   - Feeds results back to the LLM
   - Repeats until the LLM responds without tool calls
8. Saves the assistant response
9. Triggers conversation summarization if enough messages have accumulated

### LLM Client (`llm.py`)

Thin async wrapper around the OpenAI-compatible API. Single `chat()` method that handles tool calls, thinking model output, and per-call temperature overrides. This is the only code that talks to the LLM — the AI firewall insertion point.

Returns structured `LLMResponse` objects with content, reasoning, tool calls, and token usage.

### Memory (`memory.py`)

SQLite-based persistent memory with three search strategies combined via Reciprocal Rank Fusion:

1. **FTS5 keyword search** — fast exact/stemmed term matching (Porter stemmer + Unicode61 tokenizer)
2. **sqlite-vec cosine similarity** — semantic search via nomic-embed-text-v1.5 embeddings
3. **Recency + importance weighting** — recent and important memories rank higher

**Scoring formula:**
```
final_score = rrf_score + (recency_weight × 2^(-age_days / 7)) + (importance / 10 × 0.1)
```

**Database tables:**

| Table | Purpose |
|-------|---------|
| `messages` | Every message persisted per session |
| `memories` | Extracted facts with embeddings and importance scores |
| `summaries` | LLM-generated compression of old message blocks |
| `memories_fts` | FTS5 virtual table for keyword search |
| `memories_vec` | sqlite-vec virtual table for vector search |

**Conversation compression:** Every N messages (default 50), the LLM summarizes the conversation and extracts facts with importance scores (1-10). Facts above the threshold (default 3.0) are stored as memories. This enables effectively infinite conversations — the agent always has a summary of what came before plus searchable memory of key facts.

All retrieval parameters (top_k, RRF k, recency weight, importance threshold, etc.) are in `config.toml` for experimentation.

### Native Tools (`tools.py`)

Built-in tools that don't require external MCP servers:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with safety guardrails |
| `read_file` | Read files with optional offset/limit for large files |
| `write_file` | Write or append to files, creates parent directories |
| `list_directory` | List files/directories, optional recursion with depth limits |
| `web_fetch` | Fetch a URL and convert HTML to markdown via html2text |
| `web_search` | Search the web via DuckDuckGo (default) or You.com (`WEB_SEARCH_PROVIDER=youcom`) |
| `delegate` | Hand off a self-contained subtask to a sub-agent with its own tool loop |
| `code_task` | Delegate a coding task to a sub-agent with a write-run-fix loop |
| `summarize_paper` | Fetch and summarize an arXiv paper |
| `list_available_tools` | Discover MCP tools available from connected servers |
| `use_tool` | Call a specific MCP tool by name |

**Bash safety:** Commands are checked against blocked patterns before execution:
- `rm -rf /`, `mkfs`, `dd if=`, `shutdown`, `reboot`, fork bombs, writes to `/dev/sda`
- Timeout enforcement: default 30s, max 120s
- Output capped at 50KB

### Tool Output Pipeline (`tool_output.py`)

Handles large tool outputs so they don't overwhelm the LLM context:

1. **Small outputs** (< 10KB) — passed through directly
2. **Large outputs** — processed through a pipeline:
   - **Persist** — full output saved to `data/tool_outputs/` with a deterministic filename (content hash + source label)
   - **Python filter** — keyword matching against the user's query context, with structural detection (headers, code blocks). Includes 1 line of surrounding context per match.
   - **LLM extraction** — if the Python filter finds fewer than 5 keyword matches, the LLM extracts relevant parts from the raw output
   - **File reference** — a footer with the persisted file path is appended so the agent can inspect the full output later

### MCP Manager (`mcp_manager.py`)

Connects to community MCP servers via stdio transport. On startup it spawns configured servers, discovers their tools, and converts schemas to OpenAI function-calling format. Tool calls from the LLM are routed to the correct server automatically.

**Tool namespacing:** Tools are prefixed with the server name (`browser__navigate`, `filesystem__read_file`) to avoid collisions between servers.

Configure servers in `mcp_servers.json`:

```json
{
  "servers": {
    "browser": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp"],
      "transport": "stdio"
    }
  }
}
```

Adding a new tool is editing JSON — no code changes.

### Discord Bot (`discord_bot.py`)

Responds to DMs, @mentions, and replies in threads it created. Shows a typing indicator while the agent is processing.

**Session isolation:** Session IDs are derived from message context to keep memory separate:

| Context | Session ID |
|---------|-----------|
| Thread | `thread-{thread_id}` |
| DM | `dm-{user_id}` |
| Channel | `ch-{channel_id}-{user_id}` |

Long responses are split at newlines (preferred), spaces, or hard-split at 2000 characters to stay within Discord's limit.

### Observability (`observe.py`)

Every LLM call, tool execution, memory operation, and Discord message is logged as structured JSON.

**Dual output:**
- **File** — `data/logs/luna-YYYY-MM-DD.jsonl`, one file per day, machine-parseable
- **Console** — human-readable format for development

**What's logged:**

| Component | Events |
|-----------|--------|
| LLM | `llm_call`, `llm_response` (tokens, latency, tools used) |
| Memory | `memory_search` (hits, method breakdown), `memory_stored`, `summary_stored` |
| Tools | `tool_executing`, `native_tool_call`, `tool_call` (server, tool, duration, errors) |
| Discord | `discord_ready`, `discord_message` (session, author, channel) |
| MCP | `server_connected`, `tools_refreshed`, `mcp_shutdown` |
| Agent | `agent_process` (latency), `agent_response` (memory hits, tool rounds) |
| Output | `output_persisted`, `llm_extraction_triggered` |

**Inspection:**

```bash
# Watch logs in real-time
tail -f data/logs/luna-*.jsonl

# Search with jq
jq 'select(.event == "llm_response")' data/logs/luna-*.jsonl
jq 'select(.latency_ms > 5000)' data/logs/luna-*.jsonl
```

### Config (`config.py`)

Dataclass-based configuration loaded from `config.toml` with environment variable overrides. Relative paths are resolved against the project root. All fields have sensible defaults — the agent starts with zero configuration if a `config.toml` is present.

## Deployment

Copy the systemd service files and enable them:

```bash
sudo cp luna-agent.service /etc/systemd/system/
sudo cp worker-agent.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now worker-agent    # Start LLM server (Qwen3.5-35B-A3B) first
sudo systemctl enable --now luna-agent      # Then the agent (depends on worker-agent)
```

**Monitor:**

```bash
journalctl -u luna-agent -f
journalctl -u worker-agent -f
```

**CLI mode** (no Discord token): The agent starts an interactive REPL where tool calls print inline as they execute, then the final response prints below. Useful for testing without Discord.

## Dependencies

8 runtime packages, no heavy frameworks:

| Package | Purpose |
|---------|---------|
| `discord.py` | Discord API client |
| `openai` | OpenAI-compatible HTTP client |
| `mcp[cli]` | Model Context Protocol SDK |
| `sentence-transformers` | Embedding model runtime |
| `einops` | Tensor operations for embeddings |
| `sqlite-vec` | Vector search in SQLite |
| `html2text` | HTML to markdown conversion |
| `duckduckgo-search` | Web search |

**Dev:** `pytest`, `pytest-asyncio`

**Python:** >= 3.11

## What This Doesn't Include (By Design)

- No AI firewall (future — just don't block the insertion point)
- No web dashboard (future phase of observability)
- No multi-user auth (single user)
- No cloud LLM fallback (local only)
- No containers for the agent (systemd is simpler)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull request guidelines.

## License

[MIT](LICENSE) — Fabio Nonato, 2026
