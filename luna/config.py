"""Load configuration from config.toml with env var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:8001/v1"
    model: str = "worker-agent"
    max_tokens: int = 16384
    temperature: float = 1.0


@dataclass
class DiscordConfig:
    token: str = ""


@dataclass
class MemoryConfig:
    db_path: str = "data/memory.db"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimensions: int = 384
    top_k: int = 10
    chunk_size: int = 500
    summary_interval: int = 50
    rrf_k: int = 60
    importance_threshold: float = 3.0
    recency_weight: float = 0.3


@dataclass
class ObserveConfig:
    log_dir: str = "data/logs"
    log_level: str = "INFO"
    web_dashboard: bool = False
    web_port: int = 8900


@dataclass
class AgentConfig:
    workspace: str = "data/workspace"
    allow_read_outside: bool = True
    recent_messages: int = 50


@dataclass
class ClaudeCodeConfig:
    enabled: bool = True
    max_sessions: int = 3
    session_timeout: int = 600       # seconds of inactivity before expiry
    turn_timeout: int = 300          # seconds per turn
    max_budget_usd: float = 2.0      # per-session cost cap
    claude_path: str = "claude"      # path to claude CLI binary


@dataclass
class VisionConfig:
    enabled: bool = False
    device: int = 0                   # /dev/videoN index
    capture_dir: str = "data/vision"  # where captured images are saved
    width: int = 1280
    height: int = 720
    llm_endpoint: str = ""            # vision LLM endpoint (empty = use main LLM)
    llm_model: str = ""               # vision model name (empty = use main LLM model)
    fallback_api_key: str = ""        # OpenAI API key for GPT-4o fallback
    serve_port: int = 8900            # port to serve captured images for sharing


@dataclass
class WikiConfig:
    wiki_dir: str = "data/wiki"
    enabled: bool = True
    ingest_after_messages: int = 50
    max_context_chars: int = 4000
    auto_init: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observe: ObserveConfig = field(default_factory=ObserveConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    wiki: WikiConfig = field(default_factory=WikiConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    root_dir: Path = _ROOT


def _apply_section(target, data: dict) -> None:
    for key, value in data.items():
        if hasattr(target, key):
            expected = type(getattr(target, key))
            setattr(target, key, expected(value))


def load_config(config_path: Path | None = None) -> Config:
    """Load config from TOML file, then override with env vars."""
    if config_path is None:
        config_path = _ROOT / "config.toml"

    cfg = Config()

    if config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        if "llm" in raw:
            _apply_section(cfg.llm, raw["llm"])
        if "discord" in raw:
            _apply_section(cfg.discord, raw["discord"])
        if "memory" in raw:
            _apply_section(cfg.memory, raw["memory"])
        if "observe" in raw:
            _apply_section(cfg.observe, raw["observe"])
        if "agent" in raw:
            _apply_section(cfg.agent, raw["agent"])
        if "vision" in raw:
            _apply_section(cfg.vision, raw["vision"])
        if "wiki" in raw:
            _apply_section(cfg.wiki, raw["wiki"])
        if "claude_code" in raw:
            _apply_section(cfg.claude_code, raw["claude_code"])

    # Env var overrides
    if token := os.environ.get("DISCORD_TOKEN"):
        cfg.discord.token = token
    if endpoint := os.environ.get("LLM_ENDPOINT"):
        cfg.llm.endpoint = endpoint
    if model := os.environ.get("LLM_MODEL"):
        cfg.llm.model = model
    if db_path := os.environ.get("MEMORY_DB_PATH"):
        cfg.memory.db_path = db_path
    if log_dir := os.environ.get("LOG_DIR"):
        cfg.observe.log_dir = log_dir
    if wiki_dir := os.environ.get("WIKI_DIR"):
        cfg.wiki.wiki_dir = wiki_dir
    if claude_path := os.environ.get("CLAUDE_PATH"):
        cfg.claude_code.claude_path = claude_path
    if vision_api_key := os.environ.get("OPENAI_API_KEY"):
        cfg.vision.fallback_api_key = vision_api_key

    # Resolve relative paths against project root
    if not Path(cfg.memory.db_path).is_absolute():
        cfg.memory.db_path = str(cfg.root_dir / cfg.memory.db_path)
    if not Path(cfg.observe.log_dir).is_absolute():
        cfg.observe.log_dir = str(cfg.root_dir / cfg.observe.log_dir)
    if not Path(cfg.agent.workspace).is_absolute():
        cfg.agent.workspace = str(cfg.root_dir / cfg.agent.workspace)
    if not Path(cfg.wiki.wiki_dir).is_absolute():
        cfg.wiki.wiki_dir = str(cfg.root_dir / cfg.wiki.wiki_dir)
    if not Path(cfg.vision.capture_dir).is_absolute():
        cfg.vision.capture_dir = str(cfg.root_dir / cfg.vision.capture_dir)

    return cfg
