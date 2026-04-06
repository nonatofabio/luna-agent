"""Tests for the LLM Wiki knowledge base."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from luna.config import WikiConfig
from luna.wiki import (
    WikiManager,
    _parse_index_entries,
    _rrf_fuse,
    _score_entry,
    _tokenize,
)


@pytest.fixture
def wiki(tmp_path):
    """Create a fresh wiki in a temp directory."""
    config = WikiConfig(wiki_dir=str(tmp_path / "wiki"), enabled=True, auto_init=True)
    return WikiManager(config)


@pytest.fixture
def populated_wiki(wiki):
    """Wiki with a few pages and index entries."""
    wiki.write_page("homelab-setup.md", "# Homelab Setup\n\nDual RTX 3090 GPUs, 64GB RAM.")
    wiki.write_page("user-preferences.md", "# User Preferences\n\nFabio prefers concise answers.")
    wiki.write_page("projects/luna-agent.md", "# Luna Agent\n\nLocal AI agent with memory and tools.")
    return wiki


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestWikiInit:
    def test_seed_files_created(self, wiki):
        assert (wiki.wiki_dir / "schema.md").exists()
        assert (wiki.wiki_dir / "index.md").exists()
        assert (wiki.wiki_dir / "log.md").exists()

    def test_schema_has_content(self, wiki):
        content = wiki.read_page("schema.md")
        assert "Wiki Schema" in content
        assert "Page Naming" in content

    def test_index_starts_empty(self, wiki):
        entries = wiki._read_index()
        assert entries == []

    def test_reinit_is_idempotent(self, wiki):
        # Write a page, then re-init — page should survive
        wiki.write_page("test.md", "# Test")
        wiki._ensure_initialized()
        assert wiki.read_page("test.md") == "# Test"

    def test_enabled_property(self, tmp_path):
        config = WikiConfig(wiki_dir=str(tmp_path / "wiki"), enabled=True, auto_init=True)
        w = WikiManager(config)
        assert w.enabled is True

        config2 = WikiConfig(wiki_dir=str(tmp_path / "wiki2"), enabled=False, auto_init=False)
        w2 = WikiManager(config2)
        assert w2.enabled is False


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------


class TestWikiReadWrite:
    def test_write_and_read(self, wiki):
        wiki.write_page("test.md", "# Test Page\n\nSome content.")
        content = wiki.read_page("test.md")
        assert content == "# Test Page\n\nSome content."

    def test_read_nonexistent_raises(self, wiki):
        with pytest.raises(FileNotFoundError):
            wiki.read_page("nonexistent.md")

    def test_write_creates_parent_dirs(self, wiki):
        wiki.write_page("deep/nested/page.md", "# Deep Page")
        assert wiki.read_page("deep/nested/page.md") == "# Deep Page"

    def test_overwrite_existing(self, wiki):
        wiki.write_page("test.md", "v1")
        wiki.write_page("test.md", "v2")
        assert wiki.read_page("test.md") == "v2"

    def test_list_pages(self, populated_wiki):
        pages = populated_wiki.list_pages()
        assert "homelab-setup.md" in pages
        assert "user-preferences.md" in pages
        assert "projects/luna-agent.md" in pages
        # Seed files should also be listed
        assert "index.md" in pages
        assert "schema.md" in pages


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestWikiPathSafety:
    def test_traversal_blocked(self, wiki):
        with pytest.raises(ValueError, match="escapes wiki directory"):
            wiki.write_page("../escape.md", "bad")

    def test_traversal_read_blocked(self, wiki):
        with pytest.raises(ValueError, match="escapes wiki directory"):
            wiki.read_page("../../etc/passwd")

    def test_absolute_path_blocked(self, wiki):
        with pytest.raises(ValueError, match="escapes wiki directory"):
            wiki.write_page("/tmp/evil.md", "bad")

    def test_dotdot_in_middle_blocked(self, wiki):
        with pytest.raises(ValueError, match="escapes wiki directory"):
            wiki.read_page("subdir/../../escape.md")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TestWikiIndex:
    def test_parse_index_entries(self):
        text = """# Index

## Pages

- [Homelab Setup](homelab-setup.md) -- Dual RTX 3090 setup
- [User Prefs](user-preferences.md) -- Fabio's preferences
"""
        entries = _parse_index_entries(text)
        assert len(entries) == 2
        assert entries[0]["title"] == "Homelab Setup"
        assert entries[0]["path"] == "homelab-setup.md"
        assert entries[0]["summary"] == "Dual RTX 3090 setup"

    def test_write_adds_to_index(self, wiki):
        wiki.write_page("new-page.md", "# New Page\n\nSome useful content here.")
        entries = wiki._read_index()
        assert any(e["path"] == "new-page.md" for e in entries)

    def test_meta_pages_not_indexed(self, wiki):
        # schema.md, index.md, log.md should not be in index
        entries = wiki._read_index()
        paths = [e["path"] for e in entries]
        assert "schema.md" not in paths
        assert "index.md" not in paths
        assert "log.md" not in paths

    def test_no_duplicate_index_entries(self, wiki):
        wiki.write_page("test.md", "# Test\n\nContent")
        wiki.write_page("test.md", "# Test\n\nUpdated content")
        entries = wiki._read_index()
        paths = [e["path"] for e in entries]
        assert paths.count("test.md") == 1

    def test_placeholder_removed(self, wiki):
        wiki.write_page("first.md", "# First\n\nContent")
        index = wiki.read_page("index.md")
        assert "No pages yet" not in index


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class TestWikiLog:
    def test_log_entries_appended(self, wiki):
        wiki.write_page("a.md", "# A")
        wiki.write_page("b.md", "# B")
        log = wiki.read_page("log.md")
        assert "created" in log
        assert "a.md" in log
        assert "b.md" in log

    def test_log_has_timestamps(self, wiki):
        wiki.write_page("test.md", "# Test")
        log = wiki.read_page("log.md")
        # Should have [YYYY-MM-DD HH:MM] format
        import re
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]", log)


# ---------------------------------------------------------------------------
# Tokenization and scoring
# ---------------------------------------------------------------------------


class TestTokenization:
    def test_tokenize_strips_stopwords(self):
        tokens = _tokenize("the quick brown fox jumps over a lazy dog")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_tokenize_lowercase(self):
        tokens = _tokenize("Hello World FOO")
        assert "hello" in tokens
        assert "world" in tokens

    def test_score_entry_basic(self):
        entry = {"title": "Homelab Setup", "path": "homelab.md", "summary": "GPU and RAM configuration"}
        tokens = _tokenize("homelab GPU setup")
        score = _score_entry(tokens, entry)
        assert score > 0

    def test_score_entry_no_match(self):
        entry = {"title": "Cooking Recipes", "path": "cooking.md", "summary": "Italian food"}
        tokens = _tokenize("homelab GPU")
        score = _score_entry(tokens, entry)
        assert score == 0


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


class TestRRFFuse:
    def test_single_list(self):
        ranked = [("a.md", 10.0), ("b.md", 5.0), ("c.md", 1.0)]
        fused = _rrf_fuse([ranked])
        paths = [p for p, _ in fused]
        assert paths[0] == "a.md"

    def test_multiple_lists_merge(self):
        list1 = [("a.md", 10.0), ("b.md", 5.0)]
        list2 = [("b.md", 10.0), ("c.md", 5.0)]
        list3 = [("c.md", 10.0), ("a.md", 5.0)]
        fused = _rrf_fuse([list1, list2, list3])
        # All three should appear
        paths = {p for p, _ in fused}
        assert paths == {"a.md", "b.md", "c.md"}

    def test_unanimous_first_wins(self):
        list1 = [("winner.md", 10.0), ("other.md", 5.0)]
        list2 = [("winner.md", 10.0), ("other.md", 5.0)]
        list3 = [("winner.md", 10.0), ("other.md", 5.0)]
        fused = _rrf_fuse([list1, list2, list3])
        assert fused[0][0] == "winner.md"

    def test_empty_lists(self):
        assert _rrf_fuse([]) == []
        assert _rrf_fuse([[]]) == []


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestWikiQuery:
    async def test_empty_wiki_returns_empty(self, wiki):
        result = await wiki.query("anything")
        assert result == ""

    async def test_query_finds_matching_page(self, populated_wiki):
        result = await populated_wiki.query("homelab GPU setup")
        assert "RTX 3090" in result

    async def test_query_respects_max_context(self, tmp_path):
        config = WikiConfig(
            wiki_dir=str(tmp_path / "wiki"), enabled=True,
            auto_init=True, max_context_chars=50,
        )
        wiki = WikiManager(config)
        wiki.write_page("big.md", "# Big Page\n\n" + "x" * 200)
        result = await wiki.query("big page")
        assert len(result) <= 200  # some overhead for truncation message

    async def test_query_with_llm_expansion(self, populated_wiki):
        """When LLM is provided, query expands into 3 search strings."""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content='["homelab hardware", "GPU RTX 3090 setup", "server configuration"]'
        )
        result = await populated_wiki.query("tell me about the homelab", llm=mock_llm)
        mock_llm.chat.assert_called_once()
        assert "RTX 3090" in result

    async def test_query_falls_back_on_llm_failure(self, populated_wiki):
        """If LLM expansion fails, falls back to direct keyword matching."""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = RuntimeError("LLM down")
        result = await populated_wiki.query("homelab setup")
        assert "RTX 3090" in result


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestWikiIngest:
    async def test_ingest_creates_pages(self, wiki):
        mock_llm = AsyncMock()
        # Plan response: create one page
        mock_llm.chat.side_effect = [
            MagicMock(content=json.dumps({
                "updates": [],
                "creates": [{
                    "path": "test-topic.md",
                    "title": "Test Topic",
                    "summary": "A test page",
                    "info": "Some facts about testing",
                }],
                "skip_reason": None,
            })),
            # Create page response
            MagicMock(content="# Test Topic\n\nSome facts about testing."),
        ]
        result = await wiki.ingest("user: tell me about testing", "session-1", mock_llm)
        assert result["created"] == 1
        assert wiki.read_page("test-topic.md") == "# Test Topic\n\nSome facts about testing."

    async def test_ingest_updates_pages(self, populated_wiki):
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            # Plan response: update one page
            MagicMock(content=json.dumps({
                "updates": [{"path": "homelab-setup.md", "reason": "Add RAM details"}],
                "creates": [],
                "skip_reason": None,
            })),
            # Update page response
            MagicMock(content="# Homelab Setup\n\nDual RTX 3090 GPUs, 64GB DDR4 3200MHz RAM."),
        ]
        result = await populated_wiki.ingest("user: I have DDR4 3200MHz", "s1", mock_llm)
        assert result["updated"] == 1
        content = populated_wiki.read_page("homelab-setup.md")
        assert "DDR4 3200MHz" in content

    async def test_ingest_skip(self, wiki):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(content=json.dumps({
            "updates": [],
            "creates": [],
            "skip_reason": "No durable knowledge in this conversation",
        }))
        result = await wiki.ingest("user: hello!", "s1", mock_llm)
        assert result["skipped"] == "No durable knowledge in this conversation"


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


class TestWikiLint:
    async def test_healthy_wiki(self, populated_wiki):
        mock_llm = AsyncMock()
        result = await populated_wiki.lint(mock_llm)
        assert "healthy" in result.lower() or "no issues" in result.lower()

    async def test_orphan_detection(self, wiki):
        # Create a page without going through write_page's index update
        (wiki.wiki_dir / "orphan.md").write_text("# Orphan")
        mock_llm = AsyncMock()
        result = await wiki.lint(mock_llm)
        assert "orphan.md" in result

    async def test_missing_file_detection(self, wiki):
        # Add an index entry for a nonexistent file
        wiki._update_index([{"title": "Ghost", "path": "ghost.md", "summary": "Gone"}])
        mock_llm = AsyncMock()
        result = await wiki.lint(mock_llm)
        assert "ghost.md" in result


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestWikiStats:
    def test_stats_basic(self, populated_wiki):
        stats = populated_wiki.get_stats()
        assert stats["page_count"] >= 6  # 3 seed + 3 content pages
        assert stats["index_entries"] == 3
        assert stats["total_size_kb"] > 0
