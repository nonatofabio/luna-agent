"""Memory manager: SQLite + FTS5 + sqlite-vec for persistent, searchable memory."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from luna.config import MemoryConfig
from luna.observe import get_logger, log_event

logger = get_logger("memory")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL DEFAULT 5.0,
    source_session TEXT,
    created_at REAL NOT NULL,
    accessed_at REAL,
    access_count INTEGER DEFAULT 0,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    msg_range_start INTEGER,
    msg_range_end INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS thread_intents (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    original_request TEXT NOT NULL,
    interpreted_intent TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_thread_intents_session ON thread_intents(session_id);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tokenize='porter unicode61',
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


@dataclass
class MemoryResult:
    id: int
    content: str
    memory_type: str
    importance: float
    score: float  # combined retrieval score
    created_at: float


class MemoryManager:
    """Persistent memory with hybrid keyword + vector search."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._embedder = None  # lazy load
        self._vec_initialized = False

        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(db_path))
        self.db.enable_load_extension(True)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self._init_schema()
        log_event(logger, "memory_initialized", db_path=config.db_path)

    def _init_schema(self) -> None:
        self.db.executescript(SCHEMA_SQL)
        self.db.executescript(FTS_SQL)

        # sqlite-vec table — check if it exists first
        exists = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        ).fetchone()
        if not exists:
            self.db.execute(
                f"CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[{self.config.embedding_dimensions}])"
            )
        self._vec_initialized = True
        self.db.commit()

    @property
    def embedder(self):
        """Lazy-load the embedding model on first use."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.config.embedding_model,
                truncate_dim=self.config.embedding_dimensions,
                trust_remote_code=True,
                device="cpu",
            )
            log_event(logger, "embedder_loaded", model=self.config.embedding_model)
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        # nomic-embed-text requires "search_query: " or "search_document: " prefix
        vec = self.embedder.encode(f"search_query: {text}", normalize_embeddings=True)
        return vec.tolist()

    def _embed_document(self, text: str) -> list[float]:
        """Generate embedding for a document to be stored."""
        vec = self.embedder.encode(f"search_document: {text}", normalize_embeddings=True)
        return vec.tolist()

    # --- Message History ---

    def save_message(self, session_id: str, role: str, content: str, tool_calls: list | None = None) -> int:
        now = time.time()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        cur = self.db.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, tc_json, now),
        )
        self.db.commit()
        log_event(logger, "message_saved", session_id=session_id, role=role, msg_id=cur.lastrowid)
        return cur.lastrowid

    def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent messages for a session, formatted for the LLM."""
        rows = self.db.execute(
            "SELECT role, content, tool_calls FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

        messages = []
        for role, content, tc_json in reversed(rows):
            msg: dict[str, Any] = {"role": role, "content": content}
            if tc_json:
                msg["tool_calls"] = json.loads(tc_json)
            messages.append(msg)
        return messages

    def get_message_count(self, session_id: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0]

    # --- Memory CRUD ---

    def store_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 5.0,
        source_session: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Store a new memory with embedding."""
        now = time.time()
        meta_json = json.dumps(metadata) if metadata else None
        cur = self.db.execute(
            "INSERT INTO memories (content, memory_type, importance, source_session, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, memory_type, importance, source_session, now, meta_json),
        )
        mem_id = cur.lastrowid

        # Store embedding
        embedding = self._embed_document(content)
        self.db.execute(
            "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
            (mem_id, json.dumps(embedding)),
        )
        self.db.commit()

        log_event(logger, "memory_stored", mem_id=mem_id, memory_type=memory_type, importance=importance)
        return mem_id

    # --- Search ---

    def _fts_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """Full-text search, returns (id, rank) pairs."""
        rows = self.db.execute(
            "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _vec_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """Vector similarity search, returns (id, distance) pairs."""
        embedding = self._embed(query)
        rows = self.db.execute(
            "SELECT rowid, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (json.dumps(embedding), limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def search(self, query: str, top_k: int | None = None) -> list[MemoryResult]:
        """Hybrid search with Reciprocal Rank Fusion."""
        if top_k is None:
            top_k = self.config.top_k

        # Get results from both search methods
        try:
            fts_results = self._fts_search(query)
        except sqlite3.OperationalError:
            fts_results = []

        vec_results = self._vec_search(query)

        # RRF: score = sum(1 / (k + rank)) across methods
        k = self.config.rrf_k
        scores: dict[int, float] = {}

        for rank, (mem_id, _) in enumerate(fts_results):
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (k + rank + 1)

        for rank, (mem_id, _) in enumerate(vec_results):
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (k + rank + 1)

        if not scores:
            return []

        # Fetch memory details and apply recency/importance weighting
        mem_ids = list(scores.keys())
        placeholders = ",".join("?" * len(mem_ids))
        rows = self.db.execute(
            f"SELECT id, content, memory_type, importance, created_at FROM memories WHERE id IN ({placeholders})",
            mem_ids,
        ).fetchall()

        now = time.time()
        results = []
        for row in rows:
            mem_id, content, mtype, importance, created_at = row
            base_score = scores[mem_id]

            # Recency boost: exponential decay, halves every 7 days
            age_days = (now - created_at) / 86400
            recency = 2 ** (-age_days / 7)
            final_score = base_score + self.config.recency_weight * recency

            # Importance boost (normalized)
            final_score += (importance / 10.0) * 0.1

            results.append(MemoryResult(
                id=mem_id,
                content=content,
                memory_type=mtype,
                importance=importance,
                score=final_score,
                created_at=created_at,
            ))

            # Update access tracking
            self.db.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, mem_id),
            )

        self.db.commit()
        results.sort(key=lambda r: r.score, reverse=True)

        log_event(logger, "memory_search", query_len=len(query), fts_hits=len(fts_results),
                  vec_hits=len(vec_results), returned=min(top_k, len(results)))

        return results[:top_k]

    # --- Thread Intents ---

    def get_thread_intent(self, session_id: str) -> dict[str, str] | None:
        """Get the persisted intent for a session/thread."""
        row = self.db.execute(
            "SELECT original_request, interpreted_intent FROM thread_intents WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row:
            return {"original_request": row[0], "interpreted_intent": row[1]}
        return None

    def save_thread_intent(self, session_id: str, original_request: str, interpreted_intent: str) -> None:
        """Save or update the thread intent for a session."""
        now = time.time()
        self.db.execute(
            "INSERT INTO thread_intents (session_id, original_request, interpreted_intent, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "original_request = excluded.original_request, "
            "interpreted_intent = excluded.interpreted_intent, "
            "updated_at = ?",
            (session_id, original_request, interpreted_intent, now, now),
        )
        self.db.commit()
        log_event(logger, "thread_intent_saved", session_id=session_id)

    def has_thread_intent(self, session_id: str) -> bool:
        """Check if a thread already has a persisted intent."""
        row = self.db.execute(
            "SELECT 1 FROM thread_intents WHERE session_id = ?", (session_id,),
        ).fetchone()
        return row is not None

    # --- Session Summaries ---

    def get_session_summary(self, session_id: str) -> str | None:
        """Get the most recent summary for a session."""
        row = self.db.execute(
            "SELECT summary FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return row[0] if row else None

    def store_summary(self, session_id: str, summary: str, msg_start: int, msg_end: int) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO summaries (session_id, summary, msg_range_start, msg_range_end, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, summary, msg_start, msg_end, now),
        )
        self.db.commit()
        log_event(logger, "summary_stored", session_id=session_id, msg_range=f"{msg_start}-{msg_end}")
        return cur.lastrowid

    def should_summarize(self, session_id: str) -> bool:
        """Check if enough unsummarized messages have accumulated."""
        # Find the last summarized message ID
        row = self.db.execute(
            "SELECT COALESCE(MAX(msg_range_end), 0) FROM summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        last_summarized = row[0]

        # Count messages since then
        row = self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND id > ?",
            (session_id, last_summarized),
        ).fetchone()
        return row[0] >= self.config.summary_interval

    async def summarize_and_extract(self, session_id: str, llm) -> None:
        """Summarize recent messages and extract facts. Called periodically."""
        # Get unsummarized messages
        row = self.db.execute(
            "SELECT COALESCE(MAX(msg_range_end), 0) FROM summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        last_summarized = row[0]

        rows = self.db.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, last_summarized),
        ).fetchall()

        if not rows:
            return

        msg_start = rows[0][0]
        msg_end = rows[-1][0]
        conversation = "\n".join(f"{role}: {content}" for _, role, content in rows)

        # Ask LLM to summarize and extract facts
        extract_prompt = [
            {"role": "system", "content": (
                "You are a memory extraction system. Given a conversation, do two things:\n"
                "1. Write a brief summary (2-3 sentences) of what was discussed.\n"
                "2. Extract key facts as a JSON array of objects with 'content' (the fact) and "
                "'importance' (1-10, where 10 is critical).\n\n"
                "Respond in this exact JSON format:\n"
                '{"summary": "...", "facts": [{"content": "...", "importance": 5}, ...]}'
            )},
            {"role": "user", "content": f"Extract from this conversation:\n\n{conversation}"},
        ]

        try:
            response = await llm.chat(extract_prompt)
            data = json.loads(response.content)

            # Store summary
            self.store_summary(session_id, data["summary"], msg_start, msg_end)

            # Store extracted facts
            for fact in data.get("facts", []):
                if fact.get("importance", 0) >= self.config.importance_threshold:
                    self.store_memory(
                        content=fact["content"],
                        memory_type="fact",
                        importance=fact["importance"],
                        source_session=session_id,
                    )

            log_event(logger, "extraction_complete", session_id=session_id,
                      facts_extracted=len(data.get("facts", [])))
        except Exception:
            logger.exception("Failed to summarize/extract")

    def close(self) -> None:
        self.db.close()
