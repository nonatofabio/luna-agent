"""Tests for Claude Code integration (luna/claude_code.py and tools.py handler)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from luna.claude_code import (
    ClaudeCodeError,
    ClaudeCodeResponse,
    ClaudeCodeSession,
    ClaudeCodeSessionManager,
    get_session_manager,
    init_claude_code,
    shutdown_claude_code,
)
from luna.config import ClaudeCodeConfig
from luna.tools import (
    NATIVE_TOOLS,
    _CODE_TASK_ALLOWED_TOOLS,
    _DELEGATE_ALLOWED_TOOLS,
    call_native_tool,
    init_workspace,
    is_native_tool,
)


@pytest.fixture
def config():
    return ClaudeCodeConfig(
        enabled=True,
        max_sessions=3,
        session_timeout=600,
        turn_timeout=10,  # short for tests
        max_budget_usd=1.0,
        claude_path="claude",
    )


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    init_workspace(str(workspace))
    return workspace


def _make_init_event(session_id: str = "test-session-123") -> bytes:
    """Create a mock init event line."""
    return (json.dumps({
        "type": "system",
        "subtype": "init",
        "session_id": session_id,
        "tools": [],
        "model": "test-model",
    }) + "\n").encode()


def _make_result_event(
    result: str = "Done.",
    session_id: str = "test-session-123",
    is_error: bool = False,
    cost_usd: float = 0.01,
    duration_ms: int = 1500,
) -> bytes:
    """Create a mock result event line."""
    return (json.dumps({
        "type": "result",
        "subtype": "error" if is_error else "success",
        "is_error": is_error,
        "result": result,
        "session_id": session_id,
        "total_cost_usd": cost_usd,
        "duration_ms": duration_ms,
    }) + "\n").encode()


def _make_assistant_event(
    text: str = "",
    tool_uses: list[str] | None = None,
    session_id: str = "test-session-123",
) -> bytes:
    """Create a mock assistant event line."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tool in (tool_uses or []):
        content.append({"type": "tool_use", "name": tool, "input": {}})
    return (json.dumps({
        "type": "assistant",
        "message": {"content": content},
        "session_id": session_id,
    }) + "\n").encode()


def _mock_process(stdout_lines: list[bytes]):
    """Create a mock asyncio subprocess with canned stdout lines."""
    proc = MagicMock()
    proc.returncode = None  # still running

    # Mock stdin
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()
    proc.stdin.wait_closed = AsyncMock()
    proc.stdin.is_closing = MagicMock(return_value=False)

    # Mock stdout readline as an async iterator over lines
    line_iter = iter(stdout_lines)

    async def mock_readline():
        try:
            return next(line_iter)
        except StopIteration:
            return b""  # EOF

    proc.stdout = MagicMock()
    proc.stdout.readline = mock_readline

    # Mock stderr
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")

    # Mock wait
    async def mock_wait():
        proc.returncode = 0

    proc.wait = mock_wait
    proc.pid = 12345

    return proc


# ---------------------------------------------------------------------------
# ClaudeCodeSession tests
# ---------------------------------------------------------------------------


class TestClaudeCodeSession:
    async def test_start_spawns_process(self, config, tmp_path):
        """Start should spawn the process (session_id comes later from send)."""
        proc = _mock_process([])

        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc)):
            session = ClaudeCodeSession(config)
            await session.start(str(tmp_path))

            assert session.session_id is None  # not yet assigned
            assert session.is_alive

    async def test_start_raises_on_cli_not_found(self, config, tmp_path):
        """Start should raise ClaudeCodeError if claude binary not found."""
        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(side_effect=FileNotFoundError)):
            session = ClaudeCodeSession(config)
            with pytest.raises(ClaudeCodeError, match="not found"):
                await session.start(str(tmp_path))

    async def _start_session(self, config, tmp_path, stdout_lines):
        """Helper: start a session with mock process and prepare stdout for send."""
        proc = _mock_process([])  # start() no longer reads stdout
        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc)):
            session = ClaudeCodeSession(config)
            await session.start(str(tmp_path))

        # Set up stdout for the send() call
        line_iter = iter(stdout_lines)

        async def mock_readline():
            try:
                return next(line_iter)
            except StopIteration:
                return b""

        proc.stdout.readline = mock_readline
        return session, proc

    async def test_send_returns_result(self, config, tmp_path):
        """Send should return the result text from the result event."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            _make_assistant_event(text="Working on it..."),
            _make_result_event(result="Task completed successfully."),
        ])

        response = await session.send("Do something")
        assert response.result == "Task completed successfully."
        assert response.session_id == "test-session-123"
        assert not response.is_error

    async def test_send_captures_session_id_from_init(self, config, tmp_path):
        """First send should extract session_id from the init event."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event("my-session-42"),
            _make_result_event(result="ok", session_id="my-session-42"),
        ])

        assert session.session_id is None
        await session.send("hello")
        assert session.session_id == "my-session-42"

    async def test_send_collects_tool_calls(self, config, tmp_path):
        """Send should collect tool use info from assistant events."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            _make_assistant_event(tool_uses=["Bash", "Edit"]),
            _make_assistant_event(tool_uses=["Bash"]),
            _make_result_event(result="Done."),
        ])

        response = await session.send("Fix the bug")
        assert response.tool_calls_made == ["Bash", "Edit", "Bash"]

    async def test_send_handles_error_result(self, config, tmp_path):
        """Send should report is_error when result event indicates failure."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            _make_result_event(result="Budget exceeded", is_error=True),
        ])

        response = await session.send("Do something expensive")
        assert response.is_error

    async def test_send_on_dead_process(self, config, tmp_path):
        """Send should raise if the process has already exited."""
        proc = _mock_process([])

        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc)):
            session = ClaudeCodeSession(config)
            await session.start(str(tmp_path))

        # Kill the process
        proc.returncode = 1

        with pytest.raises(ClaudeCodeError, match="not alive"):
            await session.send("hello")

    async def test_send_skips_malformed_json(self, config, tmp_path):
        """Malformed JSON lines should be skipped without error."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            b"not json at all\n",
            b"{broken json\n",
            _make_result_event(result="It worked."),
        ])

        response = await session.send("test")
        assert response.result == "It worked."

    async def test_send_raises_on_eof(self, config, tmp_path):
        """Send should raise if process exits mid-turn (EOF on stdout)."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            _make_assistant_event(text="Starting..."),
            # EOF after this — no result event
        ])

        with pytest.raises(ClaudeCodeError, match="exited unexpectedly"):
            await session.send("do something")

    async def test_close_sends_eof(self, config, tmp_path):
        """Close should close stdin and wait for the process."""
        proc = _mock_process([])

        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc)):
            session = ClaudeCodeSession(config)
            await session.start(str(tmp_path))

        await session.close()
        proc.stdin.close.assert_called_once()

    async def test_send_writes_correct_ndjson(self, config, tmp_path):
        """First send should write NDJSON with session_id=None."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event("sess-abc"),
            _make_result_event(result="ok", session_id="sess-abc"),
        ])

        await session.send("Hello Claude")

        # Check what was written to stdin
        written = proc.stdin.write.call_args[0][0]
        parsed = json.loads(written.decode())
        assert parsed["type"] == "user"
        assert parsed["message"]["role"] == "user"
        assert parsed["message"]["content"] == "Hello Claude"
        assert parsed["session_id"] is None  # first call, no session_id yet

    async def test_cost_and_duration_extracted(self, config, tmp_path):
        """Send should extract cost and duration from result event."""
        session, proc = await self._start_session(config, tmp_path, [
            _make_init_event(),
            _make_result_event(result="Done", cost_usd=0.0523, duration_ms=4200),
        ])

        response = await session.send("test")
        assert response.cost_usd == 0.0523
        assert response.duration_ms == 4200


# ---------------------------------------------------------------------------
# ClaudeCodeSessionManager tests
# ---------------------------------------------------------------------------


class TestClaudeCodeSessionManager:
    def _create_and_register(self, manager, proc, session_id="test-session-123"):
        """Helper: create session, simulate first send, register."""
        async def _do(tmp_path):
            with patch("luna.claude_code.asyncio.create_subprocess_exec",
                       AsyncMock(return_value=proc)):
                session = await manager.create_session(str(tmp_path))
            # Simulate the init event assigning a session_id
            session._session_id = session_id
            manager.register_session(session)
            return session
        return _do

    async def test_create_session(self, config, tmp_path):
        """Create should start a session (no session_id yet)."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc)):
            session = await manager.create_session(str(tmp_path))

        assert session.session_id is None  # assigned on first send
        assert session.is_alive
        assert manager.active_count == 0  # not registered yet

    async def test_register_session(self, config, tmp_path):
        """Register should store session by session_id."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        session = await create(tmp_path)

        assert manager.active_count == 1
        assert manager.get_session("test-session-123") is session

    async def test_get_session_returns_active(self, config, tmp_path):
        """Get should return an active session."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        session = await create(tmp_path)

        found = manager.get_session("test-session-123")
        assert found is session

    async def test_get_session_returns_none_for_unknown(self, config):
        """Get should return None for unknown session_id."""
        manager = ClaudeCodeSessionManager(config)
        assert manager.get_session("nonexistent") is None

    async def test_get_session_returns_none_for_dead(self, config, tmp_path):
        """Get should return None and remove dead sessions."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        await create(tmp_path)

        # Kill the process
        proc.returncode = 1
        assert manager.get_session("test-session-123") is None
        assert manager.active_count == 0

    async def test_max_sessions_enforced(self, config, tmp_path):
        """Create should raise when max sessions reached."""
        config.max_sessions = 1
        manager = ClaudeCodeSessionManager(config)

        proc1 = _mock_process([])
        create1 = self._create_and_register(manager, proc1, "session-1")
        await create1(tmp_path)

        proc2 = _mock_process([])
        with patch("luna.claude_code.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=proc2)):
            with pytest.raises(ClaudeCodeError, match="Maximum sessions"):
                await manager.create_session(str(tmp_path))

    async def test_close_session(self, config, tmp_path):
        """Close should remove the session from the manager."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        await create(tmp_path)

        assert manager.active_count == 1
        await manager.close_session("test-session-123")
        assert manager.active_count == 0

    async def test_close_all(self, config, tmp_path):
        """Close all should remove all sessions."""
        manager = ClaudeCodeSessionManager(config)

        for i in range(2):
            proc = _mock_process([])
            create = self._create_and_register(manager, proc, f"session-{i}")
            await create(tmp_path)

        assert manager.active_count == 2
        await manager.close_all()
        assert manager.active_count == 0

    async def test_get_session_returns_none_when_expired(self, config, tmp_path):
        """Get should return None for sessions past the timeout."""
        config.session_timeout = 0  # expire immediately
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        await create(tmp_path)

        await asyncio.sleep(0.01)
        assert manager.get_session("test-session-123") is None

    async def test_expire_idle_cleans_up(self, config, tmp_path):
        """Expire idle should close sessions past timeout."""
        config.session_timeout = 0
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        create = self._create_and_register(manager, proc)
        await create(tmp_path)

        await asyncio.sleep(0.01)
        await manager._expire_idle()
        assert manager.active_count == 0


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


class TestAskClaudeCodeTool:
    def test_tool_is_registered(self):
        """ask_claude_code should be a native tool."""
        assert is_native_tool("ask_claude_code")

    def test_tool_in_native_tools_list(self):
        """ask_claude_code should be in the NATIVE_TOOLS schema list."""
        names = {t["function"]["name"] for t in NATIVE_TOOLS}
        assert "ask_claude_code" in names

    def test_not_in_delegate_allowed(self):
        """Sub-agents must not be able to call ask_claude_code."""
        assert "ask_claude_code" not in _DELEGATE_ALLOWED_TOOLS

    def test_not_in_code_task_allowed(self):
        """code_task sub-agents must not be able to call ask_claude_code."""
        assert "ask_claude_code" not in _CODE_TASK_ALLOWED_TOOLS

    async def test_missing_task_returns_error(self):
        """Empty task should return an error."""
        result = await call_native_tool("ask_claude_code", {})
        assert "Error" in result

    async def test_manager_not_initialized_returns_error(self):
        """Should return error when session manager is None."""
        from luna import claude_code
        original = claude_code._session_manager
        claude_code._session_manager = None
        try:
            result = await call_native_tool(
                "ask_claude_code", {"task": "test"}
            )
            assert "not enabled" in result
        finally:
            claude_code._session_manager = original

    async def test_new_session_response_includes_session_id(self, config, tmp_path):
        """Result should include the session_id for follow-up."""
        proc = _mock_process([])

        manager = ClaudeCodeSessionManager(config)
        from luna import claude_code
        original = claude_code._session_manager
        claude_code._session_manager = manager

        try:
            with patch("luna.claude_code.asyncio.create_subprocess_exec",
                       AsyncMock(return_value=proc)):
                # Mock send to return a response and simulate session_id assignment
                async def mock_send(self_ignored, message):
                    # Simulate what send() does: assign session_id from init event
                    return ClaudeCodeResponse(
                        result="I fixed the bug!",
                        session_id="new-sess-42",
                        tool_calls_made=["Edit", "Bash"],
                        is_error=False,
                        cost_usd=0.05,
                        duration_ms=3000,
                    )

                with patch.object(ClaudeCodeSession, "send", mock_send):
                    with patch.object(
                        ClaudeCodeSession, "session_id",
                        new_callable=lambda: property(lambda self: "new-sess-42"),
                    ):
                        result = await call_native_tool(
                            "ask_claude_code",
                            {"task": "Fix the bug in main.py"},
                        )

            assert "I fixed the bug!" in result
            assert "new-sess-42" in result
            assert "Edit" in result
            assert "$0.05" in result
        finally:
            claude_code._session_manager = original

    async def test_continue_session_with_session_id(self, config, tmp_path):
        """Passing session_id should continue an existing session."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        from luna import claude_code
        original = claude_code._session_manager
        claude_code._session_manager = manager

        try:
            with patch("luna.claude_code.asyncio.create_subprocess_exec",
                       AsyncMock(return_value=proc)):
                session = await manager.create_session(str(tmp_path))
            # Simulate session_id assignment from first send
            session._session_id = "existing-sess"
            manager.register_session(session)

            # Continue the session
            with patch.object(
                ClaudeCodeSession, "send",
                AsyncMock(return_value=ClaudeCodeResponse(
                    result="Follow-up done.",
                    session_id="existing-sess",
                ))
            ):
                result = await call_native_tool(
                    "ask_claude_code",
                    {"task": "Now run the tests", "session_id": "existing-sess"},
                )

            assert "Follow-up done." in result
        finally:
            claude_code._session_manager = original

    async def test_expired_session_returns_error(self, config):
        """Referencing an expired/unknown session should return error."""
        manager = ClaudeCodeSessionManager(config)

        from luna import claude_code
        original = claude_code._session_manager
        claude_code._session_manager = manager

        try:
            result = await call_native_tool(
                "ask_claude_code",
                {"task": "continue", "session_id": "does-not-exist"},
            )
            assert "not found or expired" in result
        finally:
            claude_code._session_manager = original

    async def test_error_result_flagged(self, config, tmp_path):
        """Error results from Claude Code should be clearly flagged."""
        proc = _mock_process([])
        manager = ClaudeCodeSessionManager(config)

        from luna import claude_code
        original = claude_code._session_manager
        claude_code._session_manager = manager

        try:
            with patch("luna.claude_code.asyncio.create_subprocess_exec",
                       AsyncMock(return_value=proc)):
                with patch.object(
                    ClaudeCodeSession, "send",
                    AsyncMock(return_value=ClaudeCodeResponse(
                        result="Budget exceeded",
                        session_id="test-session-123",
                        is_error=True,
                    ))
                ):
                    with patch.object(
                        ClaudeCodeSession, "session_id",
                        new_callable=lambda: property(lambda self: "test-session-123"),
                    ):
                        result = await call_native_tool(
                            "ask_claude_code",
                            {"task": "expensive task"},
                        )

            assert "error" in result.lower()
        finally:
            claude_code._session_manager = original


# ---------------------------------------------------------------------------
# Module-level init/shutdown tests
# ---------------------------------------------------------------------------


class TestModuleLifecycle:
    async def test_init_creates_manager(self):
        """init_claude_code should set the global session manager."""
        config = ClaudeCodeConfig()
        manager = init_claude_code(config)

        assert get_session_manager() is manager

        # Cleanup
        await shutdown_claude_code()
        assert get_session_manager() is None

    async def test_shutdown_cleans_up(self):
        """shutdown_claude_code should close all sessions."""
        config = ClaudeCodeConfig()
        init_claude_code(config)
        assert get_session_manager() is not None

        await shutdown_claude_code()
        assert get_session_manager() is None

    async def test_shutdown_when_not_initialized(self):
        """shutdown_claude_code should be safe when not initialized."""
        from luna import claude_code
        claude_code._session_manager = None
        await shutdown_claude_code()  # should not raise
