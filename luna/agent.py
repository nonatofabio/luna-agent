"""Core agent loop: receive message, retrieve memory, call LLM, execute tools, respond."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from luna.config import Config
from luna.llm import LLMClient
from luna.memory import MemoryManager
from luna.mcp_manager import MCPManager
from luna.observe import get_logger, log_event, log_duration
from luna.tools import NATIVE_TOOLS, is_native_tool, call_native_tool, verify_tool_result

logger = get_logger("agent")

SYSTEM_PROMPT_TEMPLATE = """\
You are Luna, an AI assistant running on Fabio's homelab. You are knowledgeable, precise, and concise. \
You think step by step on complex problems but keep routine answers brief.

## Capabilities
You have persistent memory (facts survive across sessions), access to a bash shell, the local filesystem, \
and the web. You can read/write files, run commands, search the internet, and fetch web pages. \
You are expected to use these tools proactively — do not describe what you could do, just do it.

## Tool Usage
- **bash**: System commands, git, scripts, process management. Check exit codes — non-zero means failure.
- **read_file / write_file**: File I/O. Relative paths resolve to the workspace. Writes outside workspace are blocked.
- **list_directory**: Browse the filesystem before reading specific files.
- **web_search**: Search the web via DuckDuckGo when you need current information, documentation, or facts you're unsure about.
- **web_fetch**: Fetch and read a specific URL. Use after web_search to get details from a result.
- **delegate**: Hand off a self-contained subtask to a sub-agent with its own tool loop. \
Use for multi-step research, complex file operations, or anything that benefits from focused context.
- **code_task**: Delegate a coding task to a sub-agent that writes code, runs it, checks results, \
and iterates on failures. Use for scripts, scrapers, automation, or any task requiring write-run-fix cycles. \
Prefer this over delegate for coding work.
- **ask_claude_code**: Collaborate with Claude Code (Anthropic's frontier coding agent) for tasks \
requiring iterative debugging, unfamiliar APIs, or complex multi-step coding. Claude Code autonomously \
edits files, runs commands, and fixes its own mistakes — it excels at the write-run-fix cycles you \
struggle with. EXPENSIVE — uses Anthropic API credits. Escalation hierarchy: try yourself first, \
then code_task, then ask_claude_code only when needed. You can continue a conversation by passing \
the session_id from a prior call. When Claude Code finishes, review its work and report results \
to the user.
- **list_available_tools / use_tool**: Discover and call additional MCP tools beyond the built-ins.
- **wiki_read / wiki_write / wiki_search**: Your persistent knowledge wiki. Read wiki pages for \
context, write pages to record important facts/preferences/project details that should persist \
across conversations. Use wiki_search to find relevant pages. Proactively update your wiki when \
you learn something durable about the user, their projects, or their preferences.

## Guidelines
- Act, don't ask. You have tools — use them. Install packages, run commands, create files, scan networks. \
Do it and report the results. Do not ask "would you like me to..." for safe, reversible operations.
- Never tell the user to run commands manually. You have bash. Run the command yourself, read the output, \
and iterate. The user should only need to intervene for physical actions (plugging in cables, rebooting hardware).
- When something fails, try a different approach. If a package install fails, try another method. \
If a scan finds nothing, try different parameters, a different tool, or debug why. Exhaust your options \
before asking the user for help.
- When there are multiple approaches, pick the best one and do it. Explain what you chose and why \
in your response — don't present a menu of options.
- Verify before destructive actions: check before deleting, overwriting, or modifying system config. \
But reading, installing, scanning, and creating are safe — just do them.
- Break complex tasks into steps. Use tools iteratively rather than guessing.
- When you don't know something, look it up (web_search, web_fetch, read docs) rather than guessing \
or asking the user.
- For file creation, use relative paths — they resolve to the workspace below.

## Approach
When given a task that requires multiple steps (e.g., "set up X", "discover devices", "install and test Y"):
1. Research first if needed (web_search, read docs)
2. Install dependencies in the workspace venv or with pip
3. Write and run code/scripts to accomplish the task
4. If something doesn't work, debug it — read errors, try alternatives, search for solutions
5. Report what you did and what the results were

## Knowledge
You have two knowledge systems:
1. **Wiki** (primary) — Markdown pages you maintain with synthesized knowledge about \
the user, projects, preferences, and past work. Relevant wiki content is included below. \
You can also read/write wiki pages directly with wiki tools.
2. **Memory** — Individual facts from past conversations. Use the `recall` tool to search \
your long-term memory when you need context. Call it whenever the topic shifts, you need \
historical context, or you want to check what you know about something. It searches across \
ALL sessions and channels.
- **diff**: Compare any file against the last version you read to see what changed.

{wiki_section}
{summary_section}
{intent_section}
{related_section}

Current time: {current_time}
Workspace: {workspace}"""


def _build_system_prompt(
    summary: str | None,
    current_time: str,
    workspace: str = "",
    thread_intent: dict[str, str] | None = None,
    wiki_context: str = "",
    related_intents: list[dict] | None = None,
) -> str:
    wiki_section = ""
    if wiki_context:
        wiki_section = f"## Wiki Knowledge\n{wiki_context}"

    summary_section = ""
    if summary:
        summary_section = f"## Previous Context\n{summary}"

    intent_section = ""
    if thread_intent:
        intent_section = (
            "## Thread Objective\n"
            f"**Original request:** {thread_intent['original_request']}\n"
            f"**Intent:** {thread_intent['interpreted_intent']}\n"
            "Keep this objective in mind throughout the conversation. All work in this thread "
            "should advance this goal."
        )

    related_section = ""
    if related_intents:
        lines = ["## Related Threads",
                 "Other active conversations that may be relevant:"]
        for ri in related_intents:
            lines.append(f"- [{ri['session_id'][:20]}] {ri['interpreted_intent']}")
        related_section = "\n".join(lines)

    return SYSTEM_PROMPT_TEMPLATE.format(
        wiki_section=wiki_section,
        summary_section=summary_section,
        intent_section=intent_section,
        related_section=related_section,
        current_time=current_time,
        workspace=workspace,
    )


_NARRATION_MIN_LEN = 200  # responses shorter than this after tool use are suspect


def _looks_incomplete(content: str) -> bool:
    """Detect narration fragments the model emits instead of a real answer.

    After tool rounds, the model sometimes returns a short preamble like
    "Let me fix the imports:" and stops — no tool calls, no real answer.
    This catches those cases so the agent can re-prompt.
    """
    stripped = content.strip()
    if not stripped:
        return True
    # Ends with colon — almost always a preamble for upcoming action
    if stripped.endswith(":"):
        return True
    # Very short response after tool work is likely a fragment
    if len(stripped) < _NARRATION_MIN_LEN:
        # Short is OK if it looks like a deliberate short answer (e.g. "Done.", "Yes.")
        # Heuristic: fragments tend to contain action verbs about future work
        lower = stripped.lower()
        action_phrases = ("let me ", "i'll ", "i will ", "let's ", "now i ")
        if any(lower.startswith(p) or f". {p}" in lower or f", {p}" in lower
               for p in action_phrases):
            return True
    return False


class Agent:
    """The orchestrator that ties LLM, memory, and MCP tools together."""

    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        memory: MemoryManager,
        mcp: MCPManager,
        wiki=None,
        tool_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.mcp = mcp
        self.wiki = wiki
        self.max_tool_rounds = 30  # safety limit on tool call loops
        self.tool_callback = tool_callback

    async def process(
        self,
        message: str,
        session_id: str,
        status_callback: Callable[[str, str], Any] | None = None,
    ) -> str:
        """Process a user message and return the assistant's response."""
        with log_duration(logger, "agent_process", session_id=session_id):
            return await self._process_inner(message, session_id, status_callback)

    async def _extract_intent(self, message: str, session_id: str) -> None:
        """Use the LLM to interpret the user's intent and persist it for this thread."""
        try:
            extract_msgs = [
                {"role": "system", "content": (
                    "You are a concise intent extractor. Given a user's message that starts a conversation, "
                    "produce a one-sentence interpretation of what they want to accomplish. "
                    "Focus on the end goal, not the specific steps. "
                    'Respond with ONLY a JSON object: {"intent": "..."}'
                )},
                {"role": "user", "content": message},
            ]
            response = await self.llm.chat(extract_msgs)
            data = json.loads(response.content)
            intent = data.get("intent", message)
            self.memory.save_thread_intent(session_id, message, intent)
        except Exception:
            # Fallback: store the raw message as the intent
            self.memory.save_thread_intent(session_id, message, message)
            logger.debug("Intent extraction failed, using raw message", exc_info=True)

    async def _process_inner(self, message: str, session_id: str,
                              status_callback: Callable[[str, str], Any] | None = None) -> str:
        # 1. Save user message
        self.memory.save_message(session_id, "user", message)

        # 2. Extract and persist thread intent on first message in a session
        if not self.memory.has_thread_intent(session_id):
            await self._extract_intent(message, session_id)

        # 3. Retrieve context (wiki, summary, intents)
        summary = self.memory.get_session_summary(session_id)
        thread_intent = self.memory.get_thread_intent(session_id)

        # Cross-session intent sharing
        related_intents = self.memory.search_related_intents(message, session_id)

        wiki_context = ""
        if self.wiki and self.wiki.enabled:
            try:
                wiki_context = await self.wiki.query(message, llm=self.llm)
            except Exception:
                logger.debug("Wiki query failed", exc_info=True)

        # 4. Build prompt
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        system = _build_system_prompt(
            summary, now, self.config.agent.workspace,
            thread_intent, wiki_context=wiki_context,
            related_intents=related_intents,
        )
        recent = self.memory.get_recent_messages(session_id, limit=self.config.agent.recent_messages)

        # 5. Get available tools (native only; MCP tools accessed via meta-tools)
        tools = NATIVE_TOOLS

        # 6. Build message list
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(recent)

        # 7. Call LLM
        response = await self.llm.chat(messages, tools=tools if tools else None)

        # 8. Tool call loop
        rounds = 0
        while response.has_tool_calls() and rounds < self.max_tool_rounds:
            rounds += 1

            # Append assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in response.tool_calls:
                log_event(logger, "tool_executing", tool=tc.name, session_id=session_id)

                if status_callback is not None:
                    try:
                        ret = status_callback(tc.name, tc.arguments)
                        if inspect.isawaitable(ret):
                            await ret
                    except Exception:
                        logger.debug("Status callback failed", exc_info=True)

                try:
                    if is_native_tool(tc.name):
                        result = await call_native_tool(
                            tc.name, tc.arguments,
                            context=message, llm=self.llm,
                        )
                    else:
                        result = await self.mcp.call_tool(tc.name, tc.arguments)
                except Exception as e:
                    result = f"Tool error: {e}"
                    logger.exception(f"Tool call failed: {tc.name}")

                result = verify_tool_result(tc.name, result)
                if self.tool_callback is not None:
                    self.tool_callback(tc.name, tc.arguments, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Call LLM again with tool results
            response = await self.llm.chat(messages, tools=tools if tools else None)

        if rounds >= self.max_tool_rounds:
            log_event(logger, "tool_loop_limit", session_id=session_id, rounds=rounds)
            # Ask the LLM to wrap up without tools
            messages.append({
                "role": "user",
                "content": "You have reached the tool call limit. Please respond to the user with what you have so far. Do not call any more tools.",
            })
            response = await self.llm.chat(messages)

        # 9. Incomplete response retry — model produced no content, only reasoning,
        #    or a narration fragment (e.g. "Let me fix the imports:") after tool use
        content = response.content
        needs_retry = False
        retry_reason = ""

        if rounds > 0:
            if not content and response.reasoning_content:
                needs_retry = True
                retry_reason = "empty_with_reasoning"
            elif content and _looks_incomplete(content):
                needs_retry = True
                retry_reason = "narration_fragment"

        if needs_retry:
            log_event(logger, "incomplete_retry", session_id=session_id, rounds=rounds,
                      reason=retry_reason, content_len=len(content or ""),
                      reasoning_len=len(response.reasoning_content or ""),
                      content_preview=(content or "")[:200])
            messages.append({
                "role": "user",
                "content": (
                    "Your last response appears incomplete — it reads like a narration of "
                    "what you were about to do, not a final answer. Please provide a complete "
                    "response summarizing what you accomplished and any results."
                ),
            })
            response = await self.llm.chat(messages)  # no tools — forces text
            content = response.content

        content = content or "(no response)"

        # 10. Save assistant response
        self.memory.save_message(session_id, "assistant", content)

        # 11. Periodic maintenance
        if self.memory.should_summarize(session_id):
            try:
                await self.memory.summarize_and_extract(session_id, self.llm)
            except Exception:
                logger.exception("Background summarization failed")

            if self.wiki and self.wiki.enabled:
                try:
                    conv_msgs = self.memory.get_recent_messages(session_id, limit=20)
                    conversation = "\n".join(
                        f"{m['role']}: {m['content']}" for m in conv_msgs
                    )
                    await self.wiki.ingest(conversation, session_id, self.llm)
                except Exception:
                    logger.exception("Wiki ingest failed")

        log_event(logger, "agent_response", session_id=session_id,
                  related_intents=len(related_intents), tool_rounds=rounds,
                  response_preview=content[:300] if content else "(empty)")
        return content
