"""LLM Wiki — persistent knowledge base as LLM-maintained markdown files.

Implements Karpathy's LLM Wiki pattern: synthesis happens at write time,
not read time. The wiki is a directory of interlinked markdown pages that
the LLM maintains as new knowledge arrives.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from luna.config import WikiConfig
from luna.observe import get_logger, log_event

logger = get_logger("wiki")

# ---------------------------------------------------------------------------
# Seed file content
# ---------------------------------------------------------------------------

_SCHEMA_SEED = """\
# Wiki Schema

Conventions for maintaining this knowledge wiki.

## Page Naming
- Lowercase with hyphens: `homelab-setup.md`, `user-preferences.md`
- Use `.md` extension
- Keep names short and descriptive

## Page Structure
- H1 title matching the page name
- H2 sections for subtopics
- Focus on WHY and HOW, not just WHAT
- Cross-link related pages using `[Title](path.md)` syntax
- Concise, factual content — no filler

## What to Track
- User preferences, working style, and decision rationale
- Project architecture and WHY it was designed that way
- Relationships between projects, tools, and concepts
- Non-obvious context that helps future conversations
- Recurring patterns and lessons learned

## What NOT to Track
- Runtime state: PIDs, current VRAM usage, uptimes, specific IPs
- Feature lists or tech stacks easily found in README or code
- Verbatim tool output, command results, or file listings
- Step-by-step debugging sessions (just the conclusion)
- Ephemeral information (weather, greetings, time-specific queries)
- Information that changes every conversation

## Quality Over Quantity
- A smaller wiki of high-quality pages beats a bloated one
- Prefer updating existing pages over creating new ones
- Merge thin related pages rather than keeping many stubs
- Every page should contain knowledge you can't get from `git log` or reading the code

## Merge Rules
- Preserve existing facts unless contradicted by newer info
- When contradicted, update the fact and briefly note what changed
- Keep the index.md up to date with every page change

## Index Format
Each entry in index.md follows this format:
```
- [Page Title](filename.md) -- one-line summary
```
"""

_INDEX_SEED = """\
# Wiki Index

Pages in this wiki, organized by topic.

## Pages

(No pages yet. Pages will appear here as knowledge accumulates.)
"""

_LOG_SEED = """\
# Wiki Changelog

Chronological record of wiki updates.

---
"""


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------

_INDEX_ENTRY_RE = re.compile(
    r"-\s+\[([^\]]+)\]\(([^)]+)\)\s*(?:--|—)\s*(.*)",
)


def _parse_index_entries(text: str) -> list[dict[str, str]]:
    """Parse index.md into structured entries."""
    entries: list[dict[str, str]] = []
    for m in _INDEX_ENTRY_RE.finditer(text):
        entries.append({
            "title": m.group(1).strip(),
            "path": m.group(2).strip(),
            "summary": m.group(3).strip(),
        })
    return entries


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through and or but not no nor "
    "so yet both each all any some this that these those it its i me "
    "my we our you your he she they them his her".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def _score_entry(query_tokens: list[str], entry: dict[str, str]) -> float:
    """Score an index entry against query tokens using term overlap."""
    entry_text = f"{entry['title']} {entry['summary']} {entry['path']}"
    entry_tokens = _tokenize(entry_text)
    if not entry_tokens or not query_tokens:
        return 0.0
    entry_counts = Counter(entry_tokens)
    score = sum(entry_counts.get(t, 0) for t in query_tokens)
    # Normalize by entry length to avoid bias toward long summaries
    return score / (len(entry_tokens) ** 0.5)


def _rrf_fuse(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each list is [(path, score)] sorted by score descending.
    Returns fused [(path, fused_score)] sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (path, _) in enumerate(ranked):
            scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# WikiManager
# ---------------------------------------------------------------------------


class WikiManager:
    """Manages a directory of LLM-maintained markdown wiki pages."""

    def __init__(self, config: WikiConfig) -> None:
        self.config = config
        self.wiki_dir = Path(config.wiki_dir)
        if config.auto_init:
            self._ensure_initialized()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.wiki_dir.exists()

    # --- Initialization ---

    def _ensure_initialized(self) -> None:
        """Create wiki directory and seed files if they don't exist."""
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        seeds = {
            "schema.md": _SCHEMA_SEED,
            "index.md": _INDEX_SEED,
            "log.md": _LOG_SEED,
        }
        for name, content in seeds.items():
            path = self.wiki_dir / name
            if not path.exists():
                path.write_text(content)

        log_event(logger, "wiki_initialized", wiki_dir=str(self.wiki_dir))

    # --- Read / Write ---

    def _validate_path(self, page_path: str) -> Path:
        """Resolve and validate a page path. Raises ValueError on escape."""
        resolved = (self.wiki_dir / page_path).resolve()
        try:
            resolved.relative_to(self.wiki_dir.resolve())
        except ValueError:
            raise ValueError(f"Path escapes wiki directory: {page_path}")
        return resolved

    def read_page(self, page_path: str) -> str:
        """Read a wiki page by relative path."""
        path = self._validate_path(page_path)
        if not path.exists():
            raise FileNotFoundError(f"Wiki page not found: {page_path}")
        return path.read_text()

    def write_page(self, page_path: str, content: str) -> None:
        """Write a wiki page and update index/log."""
        path = self._validate_path(page_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not path.exists()
        path.write_text(content)

        # Extract title from first H1 or use filename
        title = page_path.replace(".md", "").replace("-", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        if is_new and page_path not in ("index.md", "log.md", "schema.md"):
            summary = content.splitlines()[0] if content.strip() else ""
            # Use first non-heading line as summary
            for line in content.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    summary = stripped[:100]
                    break
            self._update_index([{"title": title, "path": page_path, "summary": summary}])

        op = "created" if is_new else "updated"
        self._append_log(op, f"{page_path} ({title})")
        log_event(logger, f"wiki_page_{op}", path=page_path)

    def list_pages(self) -> list[str]:
        """List all .md files in wiki_dir recursively, relative paths."""
        pages = []
        for p in sorted(self.wiki_dir.rglob("*.md")):
            pages.append(str(p.relative_to(self.wiki_dir)))
        return pages

    # --- Index management ---

    def _read_index(self) -> list[dict[str, str]]:
        """Parse index.md into structured entries."""
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            return []
        return _parse_index_entries(index_path.read_text())

    def _update_index(self, new_entries: list[dict[str, str]]) -> None:
        """Append new entries to index.md."""
        index_path = self.wiki_dir / "index.md"
        content = index_path.read_text() if index_path.exists() else _INDEX_SEED

        # Remove placeholder text
        content = content.replace(
            "(No pages yet. Pages will appear here as knowledge accumulates.)\n", ""
        )

        existing_paths = {e["path"] for e in _parse_index_entries(content)}

        additions = []
        for entry in new_entries:
            if entry["path"] not in existing_paths:
                additions.append(
                    f"- [{entry['title']}]({entry['path']}) -- {entry['summary']}"
                )

        if additions:
            content = content.rstrip() + "\n" + "\n".join(additions) + "\n"
            index_path.write_text(content)

    # --- Log ---

    def _append_log(self, operation: str, details: str) -> None:
        """Append an entry to log.md."""
        log_path = self.wiki_dir / "log.md"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{ts}] {operation} | {details}\n"

        if log_path.exists():
            content = log_path.read_text()
        else:
            content = _LOG_SEED
        content += entry
        log_path.write_text(content)

    # --- Query (with LLM-expanded search strings + RRF) ---

    async def query(self, question: str, llm=None) -> str:
        """Search wiki for content relevant to a question.

        If an LLM is provided, expands the question into 3 diverse search
        strings and fuses the results via RRF. Otherwise falls back to
        direct keyword matching.
        """
        entries = self._read_index()
        if not entries:
            return ""

        search_strings = [question]  # fallback: just the original question

        if llm is not None:
            try:
                search_strings = await self._expand_query(question, llm)
            except Exception:
                logger.debug("Query expansion failed, using original", exc_info=True)

        # Score entries against each search string independently
        ranked_lists: list[list[tuple[str, float]]] = []
        for query_str in search_strings:
            tokens = _tokenize(query_str)
            scored = [(e["path"], _score_entry(tokens, e)) for e in entries]
            scored = [(p, s) for p, s in scored if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_lists.append(scored)

        # RRF fuse the ranked lists
        fused = _rrf_fuse(ranked_lists)
        if not fused:
            return ""

        # Read top pages up to max_context_chars
        context_parts: list[str] = []
        total_chars = 0
        for page_path, _score in fused[:3]:
            try:
                content = self.read_page(page_path)
                if total_chars + len(content) > self.config.max_context_chars:
                    remaining = self.config.max_context_chars - total_chars
                    if remaining > 200:
                        content = content[:remaining] + "\n...(truncated)"
                    else:
                        break
                context_parts.append(content)
                total_chars += len(content)
            except (FileNotFoundError, ValueError):
                continue

        return "\n\n---\n\n".join(context_parts)

    async def _expand_query(self, question: str, llm) -> list[str]:
        """Ask the LLM to produce 3 diverse search strings for a question."""
        messages = [
            {"role": "system", "content": (
                "Given a user's question, produce 3 diverse search strings that would help "
                "find relevant wiki pages. Each should capture a different aspect or phrasing "
                "of the question. Respond with ONLY a JSON array of 3 strings, no other text."
            )},
            {"role": "user", "content": question},
        ]
        response = await llm.chat(messages, temperature=0.3)
        strings = json.loads(response.content)
        if isinstance(strings, list) and len(strings) >= 1:
            return [str(s) for s in strings[:3]]
        return [question]

    # --- Background Ingest ---

    async def ingest(self, conversation: str, session_id: str, llm) -> dict:
        """After a conversation, update wiki with new knowledge.

        Two-step process:
        1. Plan: LLM reads conversation + index → decides which pages to update/create
        2. Execute: For each page, LLM merges new info into existing content

        Returns dict with counts of pages created/updated.
        """
        index_content = (self.wiki_dir / "index.md").read_text()
        plan = await self._plan_updates(conversation, index_content, llm)

        if plan.get("skip_reason"):
            log_event(logger, "wiki_ingest_skipped", reason=plan["skip_reason"])
            return {"created": 0, "updated": 0, "skipped": plan["skip_reason"]}

        created = 0
        updated = 0

        for entry in plan.get("creates", []):
            try:
                content = await self._create_page(
                    entry["path"], entry["title"],
                    entry.get("info", conversation[:2000]), llm,
                )
                self.write_page(entry["path"], content)
                created += 1
            except Exception:
                logger.exception(f"Failed to create wiki page: {entry.get('path')}")

        for entry in plan.get("updates", []):
            try:
                current = self.read_page(entry["path"])
                content = await self._update_page(
                    entry["path"], current, entry["reason"], llm,
                )
                self.write_page(entry["path"], content)
                updated += 1
            except Exception:
                logger.exception(f"Failed to update wiki page: {entry.get('path')}")

        log_event(logger, "wiki_ingest_complete",
                  session_id=session_id, created=created, updated=updated)
        return {"created": created, "updated": updated}

    async def _plan_updates(self, conversation: str, index_content: str, llm) -> dict:
        """Ask LLM what wiki changes are needed."""
        messages = [
            {"role": "system", "content": (
                "You are a knowledge manager. Given a conversation and a wiki index, "
                "determine what wiki updates are needed.\n\n"
                "Respond with a JSON object:\n"
                '{"updates": [{"path": "existing-page.md", "reason": "Add info about X"}], '
                '"creates": [{"path": "new-page.md", "title": "Page Title", '
                '"summary": "One-line for index", "info": "Key facts to include"}], '
                '"skip_reason": null}\n\n'
                "RECORD durable knowledge:\n"
                "- User preferences, working style, decisions and rationale\n"
                "- Project architecture, design choices, why something was built a certain way\n"
                "- Relationships between projects, tools, and concepts\n"
                "- Non-obvious context that would help a future conversation\n\n"
                "DO NOT record:\n"
                "- Runtime state: PIDs, current VRAM usage, specific IP addresses, uptimes\n"
                "- Information easily derived from code or git history\n"
                "- Verbatim tool output, command results, or file listings\n"
                "- Feature lists or tech stacks that are in a project's README\n"
                "- Greetings, weather, time-specific queries\n\n"
                "Other rules:\n"
                "- Prefer updating existing pages over creating new ones\n"
                "- Use lowercase-hyphen naming for new pages: topic-name.md\n"
                "- Link related pages using [Title](path.md) syntax\n"
                "- Set skip_reason to a string if no wiki changes are needed\n"
                "- When in doubt, skip — a smaller wiki of high-quality pages beats a bloated one"
            )},
            {"role": "user", "content": (
                f"Wiki index:\n{index_content}\n\n"
                f"Conversation:\n{conversation[:3000]}"
            )},
        ]
        response = await llm.chat(messages, temperature=0.2)
        return json.loads(response.content)

    async def _update_page(self, path: str, current: str, new_info: str, llm) -> str:
        """Ask LLM to merge new information into existing page."""
        messages = [
            {"role": "system", "content": (
                "You are editing a wiki page. Merge the new information into the existing "
                "content. Preserve all existing facts unless directly contradicted by newer "
                "information. Keep the page well-organized with clear headers.\n\n"
                "Focus on durable knowledge: architecture, design decisions, rationale, "
                "preferences, and relationships. Strip out any runtime state (PIDs, current "
                "memory usage, specific uptimes), verbatim command output, or information "
                "that's easily derived from the code itself.\n\n"
                "Cross-link to other wiki pages where relevant using [Title](path.md) syntax.\n\n"
                "Return ONLY the complete updated page content in markdown."
            )},
            {"role": "user", "content": (
                f"Current page ({path}):\n{current}\n\n"
                f"New information to incorporate:\n{new_info}"
            )},
        ]
        response = await llm.chat(messages, temperature=0.2)
        return response.content

    async def _create_page(self, path: str, title: str, info: str, llm) -> str:
        """Ask LLM to write a new wiki page."""
        messages = [
            {"role": "system", "content": (
                "Write a wiki page about the given topic. Use clear markdown with an H1 "
                "title, organized H2 sections, and concise factual content.\n\n"
                "Focus on WHY and HOW, not just WHAT:\n"
                "- Why was this built/chosen? What problem does it solve?\n"
                "- How does it relate to other projects or decisions?\n"
                "- What are the key design choices and their rationale?\n\n"
                "DO NOT include:\n"
                "- Runtime state (PIDs, current memory usage, uptimes)\n"
                "- Feature lists or tech stacks easily found in README/code\n"
                "- Verbatim command output or file listings\n\n"
                "Cross-link to other wiki pages where relevant using [Title](path.md) syntax.\n\n"
                "Return ONLY the page content in markdown."
            )},
            {"role": "user", "content": f"Title: {title}\nInformation:\n{info}"},
        ]
        response = await llm.chat(messages, temperature=0.2)
        return response.content

    # --- Lint ---

    async def lint(self, llm) -> str:
        """Audit wiki health. Returns a report string."""
        pages = self.list_pages()
        index_entries = self._read_index()
        index_paths = {e["path"] for e in index_entries}
        page_set = set(pages)

        issues: list[str] = []

        # Pages not in index
        meta_pages = {"schema.md", "index.md", "log.md"}
        for p in pages:
            if p not in index_paths and p not in meta_pages:
                issues.append(f"Page not in index: {p}")

        # Index entries pointing to missing files
        for e in index_entries:
            if e["path"] not in page_set:
                issues.append(f"Index entry points to missing file: {e['path']}")

        if not issues:
            return "Wiki is healthy. No issues found."

        return "Wiki lint found issues:\n" + "\n".join(f"- {i}" for i in issues)

    # --- Utilities ---

    def get_stats(self) -> dict:
        """Return wiki statistics."""
        pages = self.list_pages()
        total_size = sum(
            (self.wiki_dir / p).stat().st_size for p in pages
            if (self.wiki_dir / p).exists()
        )
        return {
            "page_count": len(pages),
            "index_entries": len(self._read_index()),
            "total_size_kb": round(total_size / 1024, 1),
            "wiki_dir": str(self.wiki_dir),
        }
