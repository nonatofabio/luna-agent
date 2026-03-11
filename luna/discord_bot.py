"""Discord bot: relay messages to the agent loop."""

from __future__ import annotations

import json
import time

import discord

from luna.agent import Agent
from luna.observe import get_logger, log_event

logger = get_logger("discord")

MAX_MESSAGE_LENGTH = 2000  # Discord's limit
STATUS_HEADER = "**Working...**\n"
# Minimum interval between Discord edits (seconds) to avoid rate limits
_EDIT_INTERVAL = 1.5


def _session_id_for(message: discord.Message) -> str:
    """Derive a session ID from the message context.

    - Thread messages -> thread ID
    - DMs -> user ID
    - Channel messages -> channel ID + user ID
    """
    if isinstance(message.channel, discord.Thread):
        return f"thread-{message.channel.id}"
    if isinstance(message.channel, discord.DMChannel):
        return f"dm-{message.author.id}"
    return f"ch-{message.channel.id}-{message.author.id}"


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks that fit Discord's limit."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break

        # Try to split at a newline
        split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            # No newline found — split at space
            split_at = text.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            # No space either — hard split
            split_at = MAX_MESSAGE_LENGTH

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


def _format_status(tool_name: str, arguments: str) -> str:
    """Format a tool call into a short status line."""
    try:
        args = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    if tool_name == "web_search":
        return f"\U0001f50d Searching: {args.get('query', arguments)}"
    if tool_name == "web_fetch":
        return f"\U0001f4c4 Reading {args.get('url', arguments)}"
    if tool_name == "bash":
        cmd = args.get("command", arguments)
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"\u2699\ufe0f `{cmd}`"
    if tool_name == "read_file":
        return f"\U0001f4c2 Reading {args.get('path', arguments)}"
    if tool_name == "write_file":
        return f"\u270f\ufe0f Writing {args.get('path', arguments)}"
    if tool_name in ("delegate", "code_task"):
        task = args.get("task", arguments)
        if len(task) > 80:
            task = task[:77] + "..."
        return f"\U0001f916 {task}"
    if tool_name == "use_tool":
        return f"\U0001f527 Using {args.get('name', arguments)}"
    return f"\u2699\ufe0f {tool_name}..."


class _StatusMessage:
    """Manages a single Discord message that gets edited with tool call progress."""

    def __init__(self, channel: discord.abc.Messageable) -> None:
        self._channel = channel
        self._message: discord.Message | None = None
        self._lines: list[str] = []
        self._last_edit: float = 0
        self._pending: bool = False

    def _render(self) -> str:
        """Build the status message content, trimming old lines to fit."""
        # Reserve space for header
        budget = MAX_MESSAGE_LENGTH - len(STATUS_HEADER) - 50  # 50 for counter
        lines = list(self._lines)
        # Trim oldest lines if we exceed budget
        while lines and sum(len(l) + 1 for l in lines) > budget:
            lines.pop(0)
        counter = f"\n`{len(self._lines)} steps`"
        return STATUS_HEADER + "\n".join(lines) + counter

    async def update(self, tool_name: str, arguments: str) -> None:
        """Add a tool call and send/edit the status message."""
        line = _format_status(tool_name, arguments)
        self._lines.append(line)
        content = self._render()

        now = time.monotonic()
        try:
            if self._message is None:
                self._message = await self._channel.send(content)
                self._last_edit = now
            elif now - self._last_edit >= _EDIT_INTERVAL:
                await self._message.edit(content=content)
                self._last_edit = now
            else:
                self._pending = True
        except Exception:
            pass  # non-critical

    async def finish(self) -> None:
        """Final edit to show completion, then delete after a short delay."""
        if self._message is None:
            return
        try:
            if self._pending:
                # Flush the last pending update
                await self._message.edit(content=self._render())
            await self._message.delete()
        except Exception:
            pass


class LunaDiscordBot(discord.Client):
    """Discord client that relays messages to the Luna agent."""

    def __init__(self, agent: Agent) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.agent = agent

    async def on_ready(self) -> None:
        log_event(logger, "discord_ready", user=str(self.user), guilds=len(self.guilds))

    async def on_message(self, message: discord.Message) -> None:
        # Ignore own messages
        if message.author == self.user:
            return

        # Respond to DMs or when mentioned
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user in message.mentions if self.user else False
        is_thread_reply = (
            isinstance(message.channel, discord.Thread)
            and message.channel.owner_id == (self.user.id if self.user else None)
        )

        if not (is_dm or is_mentioned or is_thread_reply):
            return

        # Strip the bot mention from the message
        content = message.content
        if self.user:
            content = content.replace(f"<@{self.user.id}>", "").strip()
        if not content:
            return

        session_id = _session_id_for(message)
        log_event(logger, "discord_message", session_id=session_id,
                  author=str(message.author), channel=str(message.channel))

        status = _StatusMessage(message.channel)

        async def _send_status(tool_name: str, arguments: str) -> None:
            await status.update(tool_name, arguments)

        async with message.channel.typing():
            try:
                response = await self.agent.process(content, session_id,
                                                    status_callback=_send_status)
            except Exception:
                logger.exception("Agent processing failed")
                response = "Sorry, I encountered an error processing your message."

        # Clean up status message before sending the response
        await status.finish()

        # Send response (split if needed)
        chunks = _split_message(response)
        for chunk in chunks:
            await message.reply(chunk, mention_author=False)
