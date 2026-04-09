"""Claude Code integration: interactive multi-turn sessions via the claude CLI."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from luna.config import ClaudeCodeConfig
from luna.observe import get_logger, log_event

logger = get_logger("claude_code")

_STARTUP_TIMEOUT = 30  # seconds to wait for init event
_GRACEFUL_SHUTDOWN_TIMEOUT = 5  # seconds before SIGKILL


class ClaudeCodeError(Exception):
    """Raised when a Claude Code session encounters an error."""


@dataclass
class ClaudeCodeResponse:
    """Result from a single turn of conversation with Claude Code."""

    result: str
    session_id: str
    tool_calls_made: list[str] = field(default_factory=list)
    is_error: bool = False
    cost_usd: float | None = None
    duration_ms: int | None = None


class ClaudeCodeSession:
    """Manages a single Claude Code subprocess and its stream-json protocol."""

    def __init__(self, config: ClaudeCodeConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._last_activity: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._started = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def is_alive(self) -> bool:
        return (
            self._started
            and self._process is not None
            and self._process.returncode is None
        )

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    async def start(
        self,
        working_dir: str,
        max_budget_usd: float | None = None,
    ) -> None:
        """Spawn the claude CLI subprocess.

        Note: The claude CLI with --input-format stream-json does NOT send the
        init event until it receives the first user message on stdin. So we
        just spawn the process here and handle the init event in the first
        send() call.
        """
        if self._started:
            raise ClaudeCodeError("Session already started")

        cmd = [
            self._config.claude_path,
            "-p",
            "--verbose",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]
        budget = max_budget_usd if max_budget_usd is not None else self._config.max_budget_usd
        if budget > 0:
            cmd.extend(["--max-budget-usd", str(budget)])

        log_event(logger, "session_starting", working_dir=working_dir,
                  budget_usd=budget)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                start_new_session=True,  # own process group for cleanup
            )
        except FileNotFoundError:
            raise ClaudeCodeError(
                f"Claude Code CLI not found at '{self._config.claude_path}'. "
                "Install it or set CLAUDE_PATH."
            )

        self._started = True
        self._last_activity = time.monotonic()
        log_event(logger, "session_spawned")

    async def send(self, message: str) -> ClaudeCodeResponse:
        """Send a message and wait for the complete response."""
        if not self._started or self._process is None:
            raise ClaudeCodeError("Session is not started")
        if self._process.returncode is not None:
            raise ClaudeCodeError("Session is not alive")

        async with self._lock:
            return await self._send_inner(message)

    async def _send_inner(self, message: str) -> ClaudeCodeResponse:
        """Inner send logic, called under lock."""
        # Build and write the NDJSON user message.
        # On the first call, session_id is None — the claude CLI will assign
        # one and return it in the init event that arrives before the result.
        msg = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": message},
            "parent_tool_use_id": None,
            "session_id": self._session_id,
        })

        try:
            self._process.stdin.write((msg + "\n").encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ClaudeCodeError(f"Failed to write to Claude Code: {e}")

        log_event(logger, "session_send", session_id=self._session_id,
                  message_len=len(message))

        # Read events until we get a result
        tool_calls: list[str] = []
        cost_usd: float | None = None
        partial_result = ""

        try:
            async with asyncio.timeout(self._config.turn_timeout):
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        raise ClaudeCodeError(
                            "Claude Code process exited unexpectedly during turn. "
                            f"Partial output collected: {partial_result[:200]}"
                        )

                    try:
                        event = json.loads(line.decode(errors="replace"))
                    except json.JSONDecodeError:
                        continue  # skip malformed lines

                    event_type = event.get("type")

                    # Capture session_id from init event (first call only)
                    if (event_type == "system"
                            and event.get("subtype") == "init"
                            and self._session_id is None):
                        self._session_id = event.get("session_id")
                        log_event(logger, "session_started",
                                  session_id=self._session_id)
                        continue

                    # Collect tool use info from assistant messages
                    if event_type == "assistant":
                        msg_content = event.get("message", {}).get("content", [])
                        if isinstance(msg_content, list):
                            for block in msg_content:
                                if isinstance(block, dict):
                                    if block.get("type") == "tool_use":
                                        tool_calls.append(
                                            block.get("name", "unknown")
                                        )
                                    elif block.get("type") == "text":
                                        partial_result = block.get("text", "")

                    # Extract cost from rate limit events
                    elif event_type == "rate_limit_event":
                        # Cost info may be in the event
                        pass  # cost tracking can be added later

                    # Result event = turn complete
                    elif event_type == "result":
                        self._last_activity = time.monotonic()
                        is_error = event.get("is_error", False)
                        result_text = event.get("result", "")
                        duration = event.get("duration_ms")
                        cost = event.get("total_cost_usd")

                        log_event(
                            logger, "session_response",
                            session_id=self._session_id,
                            is_error=is_error,
                            tool_calls=len(tool_calls),
                            duration_ms=duration,
                            cost_usd=cost,
                            result_len=len(result_text),
                        )

                        return ClaudeCodeResponse(
                            result=result_text,
                            session_id=self._session_id,
                            tool_calls_made=tool_calls,
                            is_error=is_error,
                            cost_usd=cost,
                            duration_ms=duration,
                        )

        except TimeoutError:
            log_event(logger, "session_timeout", session_id=self._session_id,
                      turn_timeout=self._config.turn_timeout)
            await self._kill()
            raise ClaudeCodeError(
                f"Claude Code turn timed out after {self._config.turn_timeout}s. "
                f"Tools used so far: {tool_calls}"
            )

    async def close(self) -> None:
        """Gracefully close the session."""
        if self._process is None:
            return

        log_event(logger, "session_closing", session_id=self._session_id)

        # Close stdin to signal EOF
        if self._process.stdin and not self._process.stdin.is_closing():
            try:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
            except Exception:
                pass

        # Wait for graceful exit
        try:
            await asyncio.wait_for(
                self._process.wait(), timeout=_GRACEFUL_SHUTDOWN_TIMEOUT
            )
        except TimeoutError:
            await self._kill()

        log_event(logger, "session_closed", session_id=self._session_id)

    async def _kill(self) -> None:
        """Force-kill the process group."""
        if self._process is None or self._process.returncode is not None:
            return
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3)
            except TimeoutError:
                os.killpg(pgid, signal.SIGKILL)
                await asyncio.wait_for(self._process.wait(), timeout=3)
        except (ProcessLookupError, PermissionError):
            pass  # already dead


class ClaudeCodeSessionManager:
    """Manages a pool of active Claude Code sessions with auto-expiry."""

    def __init__(self, config: ClaudeCodeConfig) -> None:
        self._config = config
        self._sessions: dict[str, ClaudeCodeSession] = {}
        self._cleanup_task: asyncio.Task | None = None

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def start_cleanup(self) -> None:
        """Start the background cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def create_session(
        self,
        working_dir: str,
        max_budget_usd: float | None = None,
    ) -> ClaudeCodeSession:
        """Create and start a new Claude Code session.

        The session won't have a session_id until the first send() call,
        because the claude CLI only sends the init event after receiving the
        first user message. Call register_session() after the first send().
        """
        if len(self._sessions) >= self._config.max_sessions:
            active = ", ".join(
                f"{sid[:8]}... (idle {s.idle_seconds:.0f}s)"
                for sid, s in self._sessions.items()
            )
            raise ClaudeCodeError(
                f"Maximum sessions ({self._config.max_sessions}) reached. "
                f"Active: {active}. Close one first."
            )

        session = ClaudeCodeSession(self._config)
        await session.start(working_dir, max_budget_usd)
        return session

    def register_session(self, session: ClaudeCodeSession) -> None:
        """Register a session after its first send() has assigned a session_id."""
        if session.session_id:
            self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> ClaudeCodeSession | None:
        """Get an active session by ID. Returns None if expired/dead/unknown."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if not session.is_alive:
            # Dead process — remove it
            self._sessions.pop(session_id, None)
            return None
        if session.idle_seconds > self._config.session_timeout:
            # Expired — will be cleaned up by background loop
            return None
        return session

    async def close_session(self, session_id: str) -> None:
        """Explicitly close and remove a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        """Close all sessions. Called during shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.debug("Error closing session", exc_info=True)

    async def _cleanup_loop(self) -> None:
        """Periodically close idle sessions."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._expire_idle()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Cleanup loop error", exc_info=True)

    async def _expire_idle(self) -> None:
        """Close sessions that have been idle too long."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.idle_seconds > self._config.session_timeout
            or not session.is_alive
        ]
        for sid in expired:
            log_event(logger, "session_expired", session_id=sid)
            await self.close_session(sid)


# ---------------------------------------------------------------------------
# Module-level singleton (matches _mcp_manager / _wiki_manager pattern)
# ---------------------------------------------------------------------------

_session_manager: ClaudeCodeSessionManager | None = None


def init_claude_code(config: ClaudeCodeConfig) -> ClaudeCodeSessionManager:
    """Initialize the Claude Code session manager."""
    global _session_manager
    _session_manager = ClaudeCodeSessionManager(config)
    _session_manager.start_cleanup()
    log_event(logger, "claude_code_initialized",
              max_sessions=config.max_sessions,
              session_timeout=config.session_timeout)
    return _session_manager


async def shutdown_claude_code() -> None:
    """Shut down all Claude Code sessions."""
    global _session_manager
    if _session_manager:
        await _session_manager.close_all()
        _session_manager = None
        log_event(logger, "claude_code_shutdown")


def get_session_manager() -> ClaudeCodeSessionManager | None:
    """Get the global session manager (for use by tool handlers)."""
    return _session_manager
