"""Tests for the agent loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from luna.config import Config
from luna.llm import LLMClient, LLMResponse, ToolCall
from luna.agent import Agent, _build_system_prompt, _looks_incomplete
from luna.tools import verify_tool_result as _verify_tool_result
from luna.memory import MemoryResult, MemoryManager


class TestSystemPrompt:
    def test_basic_prompt(self):
        prompt = _build_system_prompt(None, "2026-01-01T00:00:00Z")
        assert "Luna" in prompt
        assert "2026-01-01" in prompt
        assert "recall" in prompt

    def test_with_summary(self):
        prompt = _build_system_prompt("They discussed AI.", "2026-01-01T00:00:00Z")
        assert "They discussed AI." in prompt

    def test_with_workspace(self):
        prompt = _build_system_prompt(None, "2026-01-01T00:00:00Z", workspace="/tmp/test-workspace")
        assert "/tmp/test-workspace" in prompt

    def test_with_wiki_context(self):
        prompt = _build_system_prompt(None, "2026-01-01T00:00:00Z", wiki_context="Dual RTX 3090 GPUs")
        assert "Dual RTX 3090 GPUs" in prompt
        assert "Wiki Knowledge" in prompt

    def test_with_related_intents(self):
        intents = [{"session_id": "ch-paper-channel", "interpreted_intent": "Indexing trading papers"}]
        prompt = _build_system_prompt(None, "2026-01-01T00:00:00Z", related_intents=intents)
        assert "Related Threads" in prompt
        assert "Indexing trading papers" in prompt

    def test_empty_related_intents(self):
        prompt = _build_system_prompt(None, "2026-01-01T00:00:00Z", related_intents=[])
        assert "Related Threads" not in prompt


class TestAgent:
    @pytest.fixture
    def agent(self, tmp_path):
        from luna.config import MemoryConfig
        from luna.observe import setup_logging

        setup_logging(str(tmp_path / "logs"), "DEBUG")

        config = Config()
        config.memory.db_path = str(tmp_path / "test.db")

        llm = MagicMock(spec=LLMClient)
        # First call is intent extraction (returns JSON), subsequent calls return normal text
        _intent_response = LLMResponse(content='{"intent": "test intent"}')
        _default_response = LLMResponse(content="Hello!")
        llm.chat = AsyncMock(side_effect=lambda *a, **kw: _intent_response
                             if not getattr(llm, '_intent_done', False)
                             else _default_response)

        def _intent_side_effect(*a, **kw):
            if not getattr(llm, '_intent_done', False):
                llm._intent_done = True
                return _intent_response
            return _default_response

        llm.chat = AsyncMock(side_effect=_intent_side_effect)

        memory = MemoryManager(MemoryConfig(
            db_path=str(tmp_path / "test.db"),
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            embedding_dimensions=384,
        ))
        # Patch search to avoid loading the embedding model in tests
        memory.search = MagicMock(return_value=[])

        mcp = MagicMock()
        mcp.get_all_tools.return_value = []

        return Agent(config, llm, memory, mcp)

    async def test_basic_response(self, agent):
        result = await agent.process("Hello", "test-session")
        assert result == "Hello!"
        # 2 calls: intent extraction + actual chat
        assert agent.llm.chat.call_count == 2

    async def test_saves_messages(self, agent):
        await agent.process("Hello", "test-session")
        messages = agent.memory.get_recent_messages("test-session")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    async def test_thinking_retry_after_tool_rounds(self, agent):
        """When model returns only reasoning after tool use, agent retries without tools."""
        intent_response = LLMResponse(content='{"intent": "run echo"}')
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command":"echo hi"}')],
        )
        # After tool use: only reasoning, no content
        thinking_only = LLMResponse(content=None, reasoning_content="Let me think about this...")
        # Retry response: real content
        retry_response = LLMResponse(content="Here is the answer.")

        agent.llm.chat = AsyncMock(side_effect=[intent_response, tool_response, thinking_only, retry_response])

        with patch("luna.agent.call_native_tool", new_callable=AsyncMock, return_value="hi\n"):
            result = await agent.process("run echo", "test-session")

        assert result == "Here is the answer."
        # 4 calls: intent, initial, after tool, retry
        assert agent.llm.chat.call_count == 4
        # Last call should have no tools (forces text)
        last_call_kwargs = agent.llm.chat.call_args_list[3]
        assert "tools" not in last_call_kwargs.kwargs or last_call_kwargs.kwargs.get("tools") is None

    async def test_no_retry_when_content_exists(self, agent):
        """No thinking retry when the model already returned content."""
        intent_response = LLMResponse(content='{"intent": "run echo"}')
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command":"echo hi"}')],
        )
        # After tool use: has both reasoning and content
        final_response = LLMResponse(
            content="Got it, here's the result.",
            reasoning_content="Let me think...",
        )

        agent.llm.chat = AsyncMock(side_effect=[intent_response, tool_response, final_response])

        with patch("luna.agent.call_native_tool", new_callable=AsyncMock, return_value="hi\n"):
            result = await agent.process("run echo", "test-session")

        assert result == "Got it, here's the result."
        # 3 calls: intent + initial + after tool. No retry.
        assert agent.llm.chat.call_count == 3

    async def test_narration_fragment_triggers_retry(self, agent):
        """When model returns a narration fragment after tools, agent retries."""
        intent_response = LLMResponse(content='{"intent": "fix imports"}')
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command":"ls"}')],
        )
        # After tool use: narration ending with colon — the bug pattern
        narration = LLMResponse(content="Let me fix the imports in the pipeline:")
        # Retry gives a real answer
        retry_response = LLMResponse(content="I've fixed the imports and the pipeline is working.")

        agent.llm.chat = AsyncMock(side_effect=[intent_response, tool_response, narration, retry_response])

        with patch("luna.agent.call_native_tool", new_callable=AsyncMock, return_value="file.txt"):
            result = await agent.process("fix imports", "test-session")

        assert result == "I've fixed the imports and the pipeline is working."
        # 4 calls: intent, initial, after tool (narration), retry
        assert agent.llm.chat.call_count == 4

    async def test_status_callback_called_before_tool(self, agent):
        """Status callback fires before tool execution, with correct args."""
        intent_response = LLMResponse(content='{"intent": "list files"}')
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command":"ls"}')],
        )
        final_response = LLMResponse(content="Done.")
        agent.llm.chat = AsyncMock(side_effect=[intent_response, tool_response, final_response])

        call_order = []
        status_cb = AsyncMock(side_effect=lambda n, a: call_order.append(("status", n)))

        async def mock_tool(*args, **kwargs):
            call_order.append(("tool", args[0]))
            return "file.txt"

        with patch("luna.agent.call_native_tool", side_effect=mock_tool):
            await agent.process("list files", "test-session", status_callback=status_cb)

        status_cb.assert_called_once_with("bash", '{"command":"ls"}')
        # Status fires before tool execution
        assert call_order == [("status", "bash"), ("tool", "bash")]

    async def test_status_callback_failure_ignored(self, agent):
        """If the status callback raises, the agent continues normally."""
        intent_response = LLMResponse(content='{"intent": "list files"}')
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command":"ls"}')],
        )
        final_response = LLMResponse(content="Done.")
        agent.llm.chat = AsyncMock(side_effect=[intent_response, tool_response, final_response])

        failing_cb = AsyncMock(side_effect=RuntimeError("Discord is down"))

        with patch("luna.agent.call_native_tool", new_callable=AsyncMock, return_value="file.txt"):
            result = await agent.process("list files", "test-session", status_callback=failing_cb)

        assert result == "Done."
        failing_cb.assert_called_once()


class TestVerifyToolResult:
    def test_clean_result_unchanged(self):
        result = "file1.txt\nfile2.txt"
        assert _verify_tool_result("bash", result) == result

    def test_nonzero_exit_code_flagged(self):
        result = "ls: cannot access '/nope': No such file or directory\n(exit code: 2)"
        verified = _verify_tool_result("bash", result)
        assert "[NOTE:" in verified
        assert "does not exist" in verified

    def test_nonzero_exit_code_generic(self):
        result = "some obscure error\n(exit code: 1)"
        verified = _verify_tool_result("bash", result)
        assert "[NOTE:" in verified
        assert "non-zero" in verified.lower()

    def test_empty_output_flagged(self):
        result = "(no output)"
        verified = _verify_tool_result("bash", result)
        assert "[NOTE:" in verified
        assert "empty" in verified.lower()

    def test_connection_refused_flagged(self):
        result = "curl: (7) Failed to connect to localhost port 9999: Connection refused\n(exit code: 7)"
        verified = _verify_tool_result("bash", result)
        assert "[NOTE:" in verified
        assert "down" in verified.lower()

    def test_permission_denied_flagged(self):
        result = "cat: /etc/shadow: Permission denied\n(exit code: 1)"
        verified = _verify_tool_result("bash", result)
        assert "[NOTE:" in verified
        assert "permission" in verified.lower()

    def test_successful_output_not_flagged(self):
        result = "hello world"
        assert _verify_tool_result("bash", result) == result

    def test_zero_exit_code_not_flagged(self):
        result = "output here\n(exit code: 0)"
        assert _verify_tool_result("bash", result) == result

    def test_error_pattern_without_exit_code(self):
        result = "Error fetching URL: Name or service not known"
        verified = _verify_tool_result("web_fetch", result)
        assert "[NOTE:" in verified
        assert "DNS" in verified


class TestLooksIncomplete:
    def test_trailing_colon(self):
        assert _looks_incomplete("Let me fix the imports:") is True

    def test_trailing_colon_with_context(self):
        assert _looks_incomplete("The scraper class is named RedditScraperWithAI. Let me fix the imports:") is True

    def test_empty_string(self):
        assert _looks_incomplete("") is True

    def test_whitespace_only(self):
        assert _looks_incomplete("   ") is True

    def test_short_action_phrase(self):
        assert _looks_incomplete("Let me try a different approach.") is True

    def test_short_ill_phrase(self):
        assert _looks_incomplete("I'll check the file now.") is True

    def test_short_deliberate_answer(self):
        assert _looks_incomplete("Done.") is False

    def test_short_yes(self):
        assert _looks_incomplete("Yes.") is False

    def test_long_response_not_incomplete(self):
        # 200+ chars of real content should not trigger
        content = "I've completed the integration. " * 10
        assert _looks_incomplete(content) is False

    def test_long_response_with_colon_at_end(self):
        # Even long responses ending with colon are suspect
        content = "A" * 250 + ":"
        assert _looks_incomplete(content) is True

    def test_real_bug_message_267(self):
        assert _looks_incomplete(
            "The AI Functions are taking too long. Let me test with a simpler, quicker approach first to see if the llama-server is responding:"
        ) is True

    def test_real_bug_message_274(self):
        assert _looks_incomplete(
            "Let me try a more targeted approach to remove the timeout arguments:"
        ) is True
