"""Tests for native built-in tools."""

from __future__ import annotations

import json
import os
import unittest.mock

import pytest

from unittest.mock import AsyncMock, MagicMock

from luna.tools import (
    NATIVE_TOOLS,
    _check_blocked,
    _check_write_allowed,
    _CODE_TASK_ALLOWED_TOOLS,
    _DELEGATE_ALLOWED_TOOLS,
    _get_delegate_tools,
    call_native_tool,
    init_workspace,
    init_tool_registry,
    is_native_tool,
    verify_tool_result,
)
from luna.llm import LLMResponse, ToolCall
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path):
    """Set workspace to a temp dir for all tests."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    init_workspace(str(workspace))
    return workspace


class TestToolRegistry:
    def test_all_tools_registered(self):
        names = {t["function"]["name"] for t in NATIVE_TOOLS}
        assert names == {
            "bash", "read_file", "write_file", "list_directory",
            "web_fetch", "web_search", "list_available_tools", "use_tool",
            "summarize_paper", "delegate", "code_task", "run_newsletter",
            "wiki_read", "wiki_write", "wiki_search",
        }

    def test_is_native_tool(self):
        assert is_native_tool("bash")
        assert is_native_tool("read_file")
        assert not is_native_tool("mcp__some_tool")
        assert not is_native_tool("nonexistent")

    def test_schemas_have_required_fields(self):
        for tool in NATIVE_TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


class TestBashTool:
    @pytest.mark.asyncio
    async def test_echo(self):
        result = await call_native_tool("bash", {"command": "echo hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_exit_code(self):
        result = await call_native_tool("bash", {"command": "exit 42"})
        assert "exit code: 42" in result

    @pytest.mark.asyncio
    async def test_stderr(self):
        result = await call_native_tool("bash", {"command": "echo err >&2"})
        assert "err" in result

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await call_native_tool("bash", {"command": "sleep 10", "timeout": 1})
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout_kills_children(self):
        """Child processes spawned by bash must not outlive a timeout."""
        import time
        # Spawn a child that writes its PID to a temp file, then sleeps forever
        cmd = (
            "python3 -c \""
            "import os, time, sys; "
            "sys.stdout.write(str(os.getpid())); sys.stdout.flush(); "
            "time.sleep(300)"
            "\""
        )
        result = await call_native_tool("bash", {"command": cmd, "timeout": 2})
        assert "timed out" in result.lower()
        # Give the OS a moment to reap
        time.sleep(0.5)
        # The child python3 process should be gone
        import psutil
        for proc in psutil.process_iter(["cmdline"]):
            cmdline = " ".join(proc.info.get("cmdline") or [])
            assert "time.sleep(300)" not in cmdline, "Child process survived timeout"

    @pytest.mark.asyncio
    async def test_cwd(self, tmp_path):
        result = await call_native_tool("bash", {"command": "pwd", "cwd": str(tmp_path)})
        assert str(tmp_path) in result

    @pytest.mark.asyncio
    async def test_empty_command(self):
        result = await call_native_tool("bash", {"command": ""})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_json_string_arguments(self):
        result = await call_native_tool("bash", json.dumps({"command": "echo json_test"}))
        assert "json_test" in result


class TestBlockedPatterns:
    def test_rm_rf_root_blocked(self):
        assert _check_blocked("rm -rf /") is not None
        assert _check_blocked("rm -rf / --no-preserve-root") is not None

    def test_mkfs_blocked(self):
        assert _check_blocked("mkfs.ext4 /dev/sda1") is not None

    def test_dd_blocked(self):
        assert _check_blocked("dd if=/dev/zero of=/dev/sda") is not None

    def test_shutdown_blocked(self):
        assert _check_blocked("shutdown -h now") is not None

    def test_safe_commands_allowed(self):
        assert _check_blocked("ls -la") is None
        assert _check_blocked("echo hello") is None
        assert _check_blocked("rm -rf ./build") is None
        assert _check_blocked("cat /etc/hostname") is None


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await call_native_tool("read_file", {"path": str(f)})
        assert "line1" in result
        assert "line2" in result

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line0\nline1\nline2\nline3\nline4\n")
        result = await call_native_tool("read_file", {"path": str(f), "offset": 1, "limit": 2})
        assert "line1" in result
        assert "line2" in result
        assert "line0" not in result
        assert "line3" not in result

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tmp_path):
        result = await call_native_tool("read_file", {"path": str(tmp_path / "nope.txt")})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_empty_path(self):
        result = await call_native_tool("read_file", {"path": ""})
        assert "Error" in result


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, setup_workspace):
        f = setup_workspace / "new.txt"
        result = await call_native_tool("write_file", {"path": str(f), "content": "hello"})
        assert "Wrote" in result
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self, setup_workspace):
        f = setup_workspace / "a" / "b" / "c.txt"
        result = await call_native_tool("write_file", {"path": str(f), "content": "deep"})
        assert "Wrote" in result
        assert f.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_append_mode(self, setup_workspace):
        f = setup_workspace / "append.txt"
        f.write_text("first")
        await call_native_tool("write_file", {"path": str(f), "content": " second", "mode": "append"})
        assert f.read_text() == "first second"

    @pytest.mark.asyncio
    async def test_overwrite(self, setup_workspace):
        f = setup_workspace / "over.txt"
        f.write_text("old")
        await call_native_tool("write_file", {"path": str(f), "content": "new"})
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_dict_content_serialized(self, setup_workspace):
        f = setup_workspace / "data.json"
        result = await call_native_tool("write_file", {"path": str(f), "content": {"key": "value"}})
        assert "Wrote" in result
        assert '"key"' in f.read_text()

    @pytest.mark.asyncio
    async def test_relative_path_goes_to_workspace(self, setup_workspace):
        result = await call_native_tool("write_file", {"path": "relative.txt", "content": "hi"})
        assert "Wrote" in result
        assert (setup_workspace / "relative.txt").read_text() == "hi"


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        result = await call_native_tool("list_directory", {"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.txt" in result

    @pytest.mark.asyncio
    async def test_list_with_dirs(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").touch()
        result = await call_native_tool("list_directory", {"path": str(tmp_path)})
        assert "subdir/" in result
        assert "file.txt" in result

    @pytest.mark.asyncio
    async def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").touch()
        result = await call_native_tool("list_directory", {"path": str(tmp_path), "recursive": True})
        assert "deep.txt" in result

    @pytest.mark.asyncio
    async def test_nonexistent(self, tmp_path):
        result = await call_native_tool("list_directory", {"path": str(tmp_path / "nope")})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = await call_native_tool("list_directory", {"path": str(empty)})
        assert "empty" in result.lower()


class TestWorkspaceSandbox:
    @pytest.mark.asyncio
    async def test_write_outside_workspace_blocked(self, tmp_path, setup_workspace):
        outside = tmp_path / "outside.txt"
        result = await call_native_tool("write_file", {"path": str(outside), "content": "nope"})
        assert "Blocked" in result
        assert not outside.exists()

    @pytest.mark.asyncio
    async def test_write_traversal_blocked(self, setup_workspace):
        result = await call_native_tool("write_file", {"path": "../escape.txt", "content": "nope"})
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_read_outside_workspace_allowed(self, tmp_path):
        outside = tmp_path / "readable.txt"
        outside.write_text("can read this")
        result = await call_native_tool("read_file", {"path": str(outside)})
        assert "can read this" in result

    @pytest.mark.asyncio
    async def test_list_outside_workspace_allowed(self, tmp_path):
        (tmp_path / "visible.txt").touch()
        result = await call_native_tool("list_directory", {"path": str(tmp_path)})
        assert "visible.txt" in result

    def test_check_write_inside_workspace(self, setup_workspace):
        inside = setup_workspace / "ok.txt"
        assert _check_write_allowed(inside) is None

    def test_check_write_outside_workspace(self, tmp_path, setup_workspace):
        outside = tmp_path / "nope.txt"
        result = _check_write_allowed(outside)
        assert result is not None
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_bash_cwd_defaults_to_workspace(self, setup_workspace):
        result = await call_native_tool("bash", {"command": "pwd"})
        assert str(setup_workspace) in result


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await call_native_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "Unknown" in result


def _make_mock_mcp(tools=None):
    """Create a mock MCPManager with the given tools."""
    mcp = MagicMock()
    if tools is None:
        mcp.servers = {}
        return mcp

    server = MagicMock()
    server.tools = tools
    mcp.servers = {"test_server": server}
    mcp.call_tool = AsyncMock(return_value="tool result")
    return mcp


class TestListAvailableTools:
    @pytest.mark.asyncio
    async def test_returns_tool_names_and_descriptions(self):
        mcp = _make_mock_mcp([
            {"name": "srv__foo", "description": "Does foo things"},
            {"name": "srv__bar", "description": "Does bar things"},
        ])
        init_tool_registry(mcp)
        result = await call_native_tool("list_available_tools", {})
        assert "srv__foo" in result
        assert "Does foo things" in result
        assert "srv__bar" in result
        assert "Does bar things" in result
        assert "2" in result  # count

    @pytest.mark.asyncio
    async def test_empty_mcp_manager(self):
        mcp = _make_mock_mcp()  # no servers
        init_tool_registry(mcp)
        result = await call_native_tool("list_available_tools", {})
        assert "No additional tools available" in result

    @pytest.mark.asyncio
    async def test_no_mcp_configured(self):
        init_tool_registry(None)
        result = await call_native_tool("list_available_tools", {})
        assert "No additional tools available" in result

    @pytest.mark.asyncio
    async def test_query_filter(self):
        mcp = _make_mock_mcp([
            {"name": "srv__search", "description": "Search the web"},
            {"name": "srv__calendar", "description": "Manage calendar events"},
        ])
        init_tool_registry(mcp)
        result = await call_native_tool("list_available_tools", {"query": "calendar"})
        assert "srv__calendar" in result
        assert "srv__search" not in result

    @pytest.mark.asyncio
    async def test_query_no_match(self):
        mcp = _make_mock_mcp([
            {"name": "srv__foo", "description": "Does foo"},
        ])
        init_tool_registry(mcp)
        result = await call_native_tool("list_available_tools", {"query": "zzz_nonexistent"})
        assert "No tools matching" in result


class TestUseTool:
    @pytest.mark.asyncio
    async def test_dispatches_to_mcp(self):
        mcp = _make_mock_mcp([
            {"name": "srv__foo", "description": "Does foo"},
        ])
        init_tool_registry(mcp)
        result = await call_native_tool("use_tool", {"name": "srv__foo", "arguments": {"key": "val"}})
        assert result == "tool result"
        mcp.call_tool.assert_awaited_once_with("srv__foo", {"key": "val"})

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        mcp = _make_mock_mcp()
        mcp.call_tool = AsyncMock(return_value="Error: Unknown tool 'nope'")
        init_tool_registry(mcp)
        result = await call_native_tool("use_tool", {"name": "nope", "arguments": {}})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_missing_name(self):
        mcp = _make_mock_mcp()
        init_tool_registry(mcp)
        result = await call_native_tool("use_tool", {})
        assert "Error" in result
        assert "'name' is required" in result

    @pytest.mark.asyncio
    async def test_no_mcp_configured(self):
        init_tool_registry(None)
        result = await call_native_tool("use_tool", {"name": "anything"})
        assert "Error" in result
        assert "MCP not configured" in result

    @pytest.mark.asyncio
    async def test_default_empty_arguments(self):
        mcp = _make_mock_mcp([
            {"name": "srv__ping", "description": "Ping"},
        ])
        init_tool_registry(mcp)
        await call_native_tool("use_tool", {"name": "srv__ping"})
        mcp.call_tool.assert_awaited_once_with("srv__ping", {})


class TestDelegateTool:
    def test_delegate_not_in_allowed_tools(self):
        """Delegate must not be able to call itself."""
        assert "delegate" not in _DELEGATE_ALLOWED_TOOLS

    def test_mcp_meta_tools_not_in_allowed_tools(self):
        """MCP meta-tools should not be available to sub-agent."""
        assert "list_available_tools" not in _DELEGATE_ALLOWED_TOOLS
        assert "use_tool" not in _DELEGATE_ALLOWED_TOOLS

    def test_get_delegate_tools_returns_subset(self):
        tools = _get_delegate_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == _DELEGATE_ALLOWED_TOOLS

    @pytest.mark.asyncio
    async def test_missing_task_returns_error(self):
        result = await call_native_tool("delegate", {})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_missing_llm_returns_error(self):
        result = await call_native_tool("delegate", {"task": "do something"}, llm=None)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_simple_delegation(self):
        """Test that delegate calls LLM and returns its response."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Task complete."))

        result = await call_native_tool(
            "delegate",
            {"task": "Summarize the current directory"},
            llm=mock_llm,
        )
        assert result == "Task complete."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegation_with_tool_calls(self):
        """Test that the sub-agent can use tools within its loop."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command": "echo hello"}')],
            ),
            LLMResponse(content="The command output was: hello"),
        ])

        result = await call_native_tool(
            "delegate",
            {"task": "Run echo hello and tell me what it says"},
            llm=mock_llm,
        )
        assert "hello" in result.lower()
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_delegation_blocks_disallowed_tools(self):
        """Test that sub-agent cannot use delegate or MCP meta-tools."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="delegate", arguments='{"task": "recurse"}')],
            ),
            LLMResponse(content="Done."),
        ])

        result = await call_native_tool(
            "delegate",
            {"task": "Try to recurse"},
            llm=mock_llm,
        )
        assert result == "Done."

    @pytest.mark.asyncio
    async def test_delegation_with_context(self):
        """Test that optional context is passed to the sub-agent."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Got it."))

        await call_native_tool(
            "delegate",
            {"task": "Do the thing", "context": "We are working on project X"},
            llm=mock_llm,
        )

        # Verify context appears in the system prompt
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "project X" in system_msg


class TestSummarizePaper:
    def _mock_arxiv_paper(self):
        """Create a mock arXiv paper object."""
        paper = MagicMock()
        paper.title = "TestNet: A Novel Method for Testing"
        author1 = MagicMock()
        author1.name = "Alice Smith"
        author2 = MagicMock()
        author2.name = "Bob Jones"
        paper.authors = [author1, author2]
        paper.summary = (
            "We present TestNet, a novel approach to automated testing. "
            "Our method achieves 95.2% accuracy on the TestBench dataset, "
            "outperforming the previous state-of-the-art by 3.1%."
        )
        return paper

    @pytest.mark.asyncio
    async def test_missing_arxiv_id(self):
        result = await call_native_tool("summarize_paper", {})
        assert "Error" in result
        assert "arxiv_id" in result

    @pytest.mark.asyncio
    async def test_missing_llm(self):
        result = await call_native_tool("summarize_paper", {"arxiv_id": "2601.10825"}, llm=None)
        assert "Error" in result
        assert "LLM" in result

    @pytest.mark.asyncio
    async def test_invalid_style(self):
        mock_llm = MagicMock()
        result = await call_native_tool(
            "summarize_paper",
            {"arxiv_id": "2601.10825", "style": "invalid"},
            llm=mock_llm,
        )
        assert "Error" in result
        assert "style" in result

    @pytest.mark.asyncio
    async def test_successful_summarization(self):
        """Test the full extract-then-summarize pipeline with mocks."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(content="**Method name:** TestNet\n**Key claims:** 95.2% accuracy on TestBench"),
            LLMResponse(content="TestNet achieves 95.2% accuracy on TestBench."),
        ])

        mock_paper = self._mock_arxiv_paper()

        # Patch the arxiv module that gets imported inside _tool_summarize_paper
        mock_arxiv = MagicMock()
        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])
        mock_arxiv.Client.return_value = mock_client
        mock_arxiv.Search.return_value = MagicMock()

        with unittest.mock.patch.dict("sys.modules", {"arxiv": mock_arxiv}):
            result = await call_native_tool(
                "summarize_paper",
                {"arxiv_id": "2601.10825"},
                llm=mock_llm,
            )

        # Verify output structure
        assert "TestNet" in result
        assert "Summary" in result
        assert "Extracted Facts" in result
        assert "Raw Abstract" in result

        # Verify LLM was called twice (extract + summarize)
        assert mock_llm.chat.call_count == 2

        # Verify extraction was called with low temperature
        extract_call = mock_llm.chat.call_args_list[0]
        assert extract_call[1].get("temperature") == 0.2

        # Verify summarization was called with slightly higher temperature
        summarize_call = mock_llm.chat.call_args_list[1]
        assert summarize_call[1].get("temperature") == 0.4

    @pytest.mark.asyncio
    async def test_summarize_paper_is_native(self):
        assert is_native_tool("summarize_paper")

    def test_summarize_paper_in_delegate_allowed(self):
        assert "summarize_paper" in _DELEGATE_ALLOWED_TOOLS


class TestCodeTask:
    def test_code_task_not_in_delegate_allowed(self):
        """Delegate sub-agent must not be able to spawn a code_task."""
        assert "code_task" not in _DELEGATE_ALLOWED_TOOLS

    def test_code_task_not_recursive(self):
        """code_task must not be in its own allowed tools."""
        assert "code_task" not in _CODE_TASK_ALLOWED_TOOLS

    def test_delegate_not_in_code_task_allowed(self):
        """code_task must not be able to spawn delegate."""
        assert "delegate" not in _CODE_TASK_ALLOWED_TOOLS

    def test_use_tool_not_in_code_task_allowed(self):
        """code_task must not be able to call MCP meta-tools."""
        assert "use_tool" not in _CODE_TASK_ALLOWED_TOOLS
        assert "list_available_tools" not in _CODE_TASK_ALLOWED_TOOLS

    @pytest.mark.asyncio
    async def test_missing_task_error(self):
        result = await call_native_tool("code_task", {})
        assert "Error" in result
        assert "'task' is required" in result

    @pytest.mark.asyncio
    async def test_missing_llm_error(self):
        result = await call_native_tool("code_task", {"task": "write hello"}, llm=None)
        assert "Error" in result
        assert "LLM" in result

    @pytest.mark.asyncio
    async def test_simple_code_task(self):
        """LLM returns content directly (no tool calls)."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Script written and tested."))

        result = await call_native_tool(
            "code_task",
            {"task": "Write a hello world script"},
            llm=mock_llm,
        )
        assert result == "Script written and tested."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        """Sub-agent uses tools and iterates."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="tc1", name="write_file", arguments='{"path": "hello.py", "content": "print(1)"}'),
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="tc2", name="bash", arguments='{"command": "python hello.py"}'),
                ],
            ),
            LLMResponse(content="Script works. Output: 1"),
        ])

        result = await call_native_tool(
            "code_task",
            {"task": "Write and run a script"},
            llm=mock_llm,
        )
        assert "Script works" in result
        assert mock_llm.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_blocks_disallowed_tools(self):
        """Sub-agent cannot use delegate, use_tool, or code_task."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="delegate", arguments='{"task": "recurse"}')],
            ),
            LLMResponse(content="Done."),
        ])

        result = await call_native_tool(
            "code_task",
            {"task": "Try to recurse"},
            llm=mock_llm,
        )
        assert result == "Done."

    @pytest.mark.asyncio
    async def test_working_dir_created(self, setup_workspace):
        """Working directory is created within workspace."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Done."))

        await call_native_tool(
            "code_task",
            {"task": "test task", "working_dir": "my_project"},
            llm=mock_llm,
        )
        assert (setup_workspace / "my_project").is_dir()

    @pytest.mark.asyncio
    async def test_auto_working_dir_name(self, setup_workspace):
        """Working dir name auto-derived from task when not specified."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Done."))

        await call_native_tool(
            "code_task",
            {"task": "Fetch GitHub trending repos"},
            llm=mock_llm,
        )
        # Should create a dir with a sanitized version of the task
        dirs = [d.name for d in setup_workspace.iterdir() if d.is_dir()]
        assert len(dirs) == 1
        assert "fetch" in dirs[0].lower()

    @pytest.mark.asyncio
    async def test_low_temperature(self):
        """All LLM calls use temperature=0.4."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command": "echo hi"}')],
            ),
            LLMResponse(content="Done."),
        ])

        await call_native_tool(
            "code_task",
            {"task": "test"},
            llm=mock_llm,
        )

        for call in mock_llm.chat.call_args_list:
            assert call[1].get("temperature") == 0.4

    @pytest.mark.asyncio
    async def test_verify_tool_result_called(self):
        """Error annotations are applied to inner tool results."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command": "false"}')],
            ),
            LLMResponse(content="Failed."),
        ])

        await call_native_tool(
            "code_task",
            {"task": "run failing command"},
            llm=mock_llm,
        )

        # The tool result message should have the [NOTE] annotation from verify_tool_result
        second_call_messages = mock_llm.chat.call_args_list[1][0][0]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"][0]
        assert "[NOTE:" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_context_in_system_prompt(self):
        """Optional context is included in the system prompt."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Done."))

        await call_native_tool(
            "code_task",
            {"task": "do something", "context": "We use Python 3.12"},
            llm=mock_llm,
        )

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "Python 3.12" in system_msg
