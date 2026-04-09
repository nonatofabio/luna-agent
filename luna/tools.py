"""Native built-in tools: bash, file I/O, web fetch, web search."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

from luna.observe import get_logger, log_event
from luna.tool_output import LLMExtractor, process_large_output

logger = get_logger("tools")

BASH_MAX_OUTPUT = 50_000
BASH_DEFAULT_TIMEOUT = 30
BASH_MAX_TIMEOUT = 120
LIST_DIR_MAX_ENTRIES = 500

# Patterns that indicate tool execution problems
_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("connection refused", "The target service may be down."),
    ("permission denied", "Permission issue — may need sudo or different path."),
    ("no such file or directory", "File/path does not exist."),
    ("command not found", "Command is not installed or not in PATH."),
    ("timed out", "Operation timed out — consider a longer timeout or simpler approach."),
    ("name or service not known", "DNS resolution failed — check the hostname."),
    ("disk quota exceeded", "Out of disk space."),
    ("connection timed out", "Network timeout — host may be unreachable."),
]


def verify_tool_result(tool_name: str, result: str) -> str:
    """Check a tool result for common problems and annotate if issues found.

    Returns the result string, possibly with a [NOTE] appended.
    Pure string matching — no LLM calls.
    """
    if not result or result.strip() == "(no output)":
        return result + "\n[NOTE: Tool returned empty/no output. Verify the command was correct.]"

    result_lower = result.lower()

    # Check for non-zero exit codes in bash output
    if "(exit code:" in result_lower and "(exit code: 0)" not in result_lower:
        for pattern, hint in _ERROR_PATTERNS:
            if pattern in result_lower:
                return result + f"\n[NOTE: {hint}]"
        return result + "\n[NOTE: Command exited with non-zero status. Check the output for errors.]"

    # Check for error patterns even without exit codes (web_fetch, MCP tools, etc.)
    for pattern, hint in _ERROR_PATTERNS:
        if pattern in result_lower:
            return result + f"\n[NOTE: {hint}]"

    return result

# Workspace — set by init_workspace() at startup
_workspace: Path | None = None
_allow_read_outside: bool = True

# MCP manager — set by init_tool_registry() at startup
_mcp_manager: "MCPManager | None" = None

# Wiki manager — set by init_wiki() at startup
_wiki_manager: "WikiManager | None" = None

# Memory manager — set by init_memory() at startup
_memory_manager: "MemoryManager | None" = None


def init_tool_registry(mcp: "MCPManager") -> None:
    """Register the MCP manager so meta-tools can discover and call MCP tools."""
    global _mcp_manager
    _mcp_manager = mcp


def init_wiki(wiki: "WikiManager") -> None:
    """Register the wiki manager so wiki tools can operate."""
    global _wiki_manager
    _wiki_manager = wiki


def init_memory(memory: "MemoryManager") -> None:
    """Register the memory manager so recall/diff tools can operate."""
    global _memory_manager
    _memory_manager = memory


def init_workspace(workspace: str, allow_read_outside: bool = True) -> None:
    """Configure the workspace sandbox for file tools."""
    global _workspace, _allow_read_outside
    _workspace = Path(workspace).resolve()
    _workspace.mkdir(parents=True, exist_ok=True)
    _allow_read_outside = allow_read_outside
    log_event(logger, "workspace_initialized", workspace=str(_workspace))


def _resolve_path(path_str: str) -> Path:
    """Resolve a path: relative paths go to workspace, absolute paths stay as-is."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        base = _workspace or Path(__file__).resolve().parent.parent
        path = base / path
    return path.resolve()


def _check_write_allowed(path: Path) -> str | None:
    """Return an error message if writing to this path is not allowed."""
    if _workspace is None:
        return None  # no sandbox configured
    resolved = path.resolve()
    try:
        resolved.relative_to(_workspace)
        return None  # inside workspace
    except ValueError:
        return f"Blocked: writes are confined to workspace ({_workspace}). Path {resolved} is outside."

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/\s*$",
    r"\brm\s+-rf\s+/\s+",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r"\bsystemctl\s+(halt|poweroff|reboot)\b",
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",  # fork bomb
    r"\b>\s*/dev/sda",
]

# --- Tool schemas (OpenAI function-calling format) ---

NATIVE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command. Use for system commands, git, package management, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30, max 120).",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Supports offset/limit for large files. Relative paths resolve to the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (0-based). Default: 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to read. Default: all.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. Relative paths resolve to the workspace directory. Writes outside the workspace are not allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append"],
                        "description": "Write mode: 'write' (default, overwrite) or 'append'.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path. Relative paths resolve to the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Default: current directory.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively. Default: false.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max recursion depth. Default: 3.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page and extract content as markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to look for in the page (guides extraction).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_tools",
            "description": "List additional tools available beyond the built-in ones. Returns tool names and descriptions. Use this to discover what extra capabilities are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional keyword to filter tools by name or description.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_tool",
            "description": "Call a discovered tool by name. Use list_available_tools first to find available tools and their expected arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The tool name (as returned by list_available_tools).",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the tool.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_paper",
            "description": (
                "Summarize an arXiv paper using a two-step extract-then-summarize pipeline. "
                "Fetches the paper, extracts verbatim facts from the abstract, then generates "
                "a summary constrained to only those facts. Prevents hallucination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "The arXiv paper ID (e.g., '2601.10825').",
                    },
                    "style": {
                        "type": "string",
                        "enum": ["technical", "linkedin"],
                        "description": "Summary style: 'technical' (default) or 'linkedin'.",
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Delegate a self-contained subtask to a sub-agent with its own tool loop. "
                "The sub-agent can use bash, read_file, write_file, list_directory, web_search, "
                "and web_fetch. Use this for complex tasks that require multiple tool calls "
                "(e.g., 'research X and summarize', 'find and fix the bug in Y'). "
                "Returns the sub-agent's final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of the subtask to perform.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional background context to help the sub-agent.",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_task",
            "description": (
                "Delegate a coding task to a specialized sub-agent that writes, runs, and iterates "
                "on code. Use for scripts, scrapers, data processing, automation — anything needing "
                "code execution with a write-run-fix loop. Prefer this over delegate for coding work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of the coding task.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Background context or constraints.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Subdirectory within workspace (auto-generated if omitted).",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_newsletter",
            "description": (
                "Run the 'Last Week on Autonomous AI' multi-source newsletter pipeline. "
                "Fetches content from Reddit, arXiv, newsletters, and GitHub, ranks it with "
                "AI Functions, and generates a Markdown newsletter. Returns the pipeline "
                "summary and path to the generated newsletter file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_items_per_source": {
                        "type": "integer",
                        "description": "Maximum items to fetch per source (default 5).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_read",
            "description": (
                "Read a page from your persistent knowledge wiki. Use 'index.md' to browse "
                "available pages. The wiki contains synthesized knowledge about the user, "
                "projects, preferences, and past work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Page path relative to wiki root (e.g., 'index.md', 'homelab-setup.md').",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_write",
            "description": (
                "Create or update a page in your persistent knowledge wiki. Use this to record "
                "important facts, preferences, project details, or decisions that should persist "
                "across conversations. Always check the index first to avoid duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Page path relative to wiki root (e.g., 'homelab-setup.md').",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown content for the page.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_claude_code",
            "description": (
                "Collaborate with Claude Code (Anthropic's frontier coding agent) on a task. "
                "Claude Code can autonomously edit files, run commands, search the web, and "
                "iterate on errors — it excels at write-run-fix cycles you struggle with. "
                "Use for: complex coding requiring iterative debugging, work with unfamiliar "
                "APIs/frameworks, long multi-step autonomous tasks, or anything where code_task "
                "failed or would likely fail. "
                "COST WARNING: Uses Anthropic API credits (~$0.01-2.00 per task). Do NOT use "
                "for trivial tasks you can handle yourself. Escalation hierarchy: try yourself "
                "first, then code_task, then ask_claude_code only when needed. "
                "To continue a previous conversation, pass the session_id from a prior call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of the task or follow-up message.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Background context, constraints, or relevant information.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID from a previous ask_claude_code call to continue that conversation.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for Claude Code (default: workspace).",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": (
                "Search the wiki for pages relevant to a query. Returns matching page "
                "content from the most relevant pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the wiki.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search your long-term memory for relevant facts and context. "
                "Call this whenever you need historical context, the topic shifts, "
                "or you want to check what you know about something. "
                "Searches across ALL sessions and channels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff",
            "description": (
                "Compare a file's current content against the last version you read. "
                "Shows a unified diff of what changed. Use this to see what was modified "
                "in a file since you last looked at it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to diff.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

_NATIVE_TOOL_NAMES: set[str] = {t["function"]["name"] for t in NATIVE_TOOLS}


def is_native_tool(name: str) -> bool:
    """Check if a tool name is a native built-in tool."""
    return name in _NATIVE_TOOL_NAMES


async def call_native_tool(
    name: str,
    arguments: str | dict,
    context: str = "",
    llm: LLMExtractor | None = None,
    root: Path | None = None,
) -> str:
    """Dispatch a native tool call and return the result string."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments else {}

    handler = _TOOL_REGISTRY.get(name)
    if handler is None:
        return f"Error: Unknown native tool '{name}'"

    # Log arguments (truncate large values to keep logs manageable)
    safe_args = {}
    for k, v in arguments.items():
        s = str(v)
        safe_args[k] = s[:500] if len(s) > 500 else s
    log_event(logger, "native_tool_call", tool=name, arguments=safe_args)
    try:
        result = await handler(arguments, context=context, llm=llm, root=root)
        log_event(
            logger,
            "native_tool_result",
            tool=name,
            result_len=len(result),
            result_preview=result[:500] if len(result) > 500 else result,
            success=True,
        )
        return result
    except Exception as e:
        logger.exception(f"Native tool error: {name}")
        log_event(logger, "native_tool_result", tool=name, error=str(e), success=False)
        return f"Error executing {name}: {e}"


# --- Tool implementations ---


def _check_blocked(command: str) -> str | None:
    """Return an error message if the command matches a blocked pattern."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return f"Blocked: command matches dangerous pattern '{pattern}'"
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} total chars)"


def _run_in_pgroup(command: str, cwd: str | None, timeout: int) -> tuple:
    """Run a shell command in its own process group.

    On timeout, sends SIGTERM then SIGKILL to the entire process group
    so child processes (e.g. long-running scripts) don't become orphaned.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        cwd=cwd,
        text=True,
        start_new_session=True,  # new process group
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc, stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        pgid = os.getpgid(proc.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except ProcessLookupError:
            pass  # already dead
        raise


async def _tool_bash(args: dict, **kwargs) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: 'command' is required"

    blocked = _check_blocked(command)
    if blocked:
        return blocked

    timeout = min(args.get("timeout", BASH_DEFAULT_TIMEOUT), BASH_MAX_TIMEOUT)
    cwd = args.get("cwd") or (_workspace and str(_workspace))

    try:
        proc, stdout, stderr, returncode = await asyncio.wait_for(
            asyncio.to_thread(_run_in_pgroup, command, cwd, timeout),
            timeout=timeout + 5,  # extra margin for thread overhead
        )
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        return f"Error: Command timed out after {timeout}s"
    except FileNotFoundError:
        return f"Error: Working directory not found: {cwd}"

    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += ("\n--- stderr ---\n" if output else "") + stderr

    if returncode != 0:
        output += f"\n(exit code: {returncode})"

    return _truncate(output, BASH_MAX_OUTPUT) if output else "(no output)"


async def _tool_read_file(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' is required"

    path = _resolve_path(path_str)

    if not path.exists():
        return f"Error: File not found: {path}"
    if not path.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied: {path}"

    offset = args.get("offset", 0)
    limit = args.get("limit")

    # Track file reads for change detection
    change_note = ""
    if _memory_manager is not None:
        changed, last_seen = _memory_manager.check_file_changed(str(path), content)
        if changed and last_seen is not None:
            from datetime import datetime, timezone
            ts = datetime.fromtimestamp(last_seen, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            change_note = f"[NOTE: This file has changed since you last read it at {ts}.]\n\n"
        _memory_manager.record_file_read(str(path), content)

    if offset or limit:
        lines = content.splitlines(keepends=True)
        end = offset + limit if limit else len(lines)
        content = "".join(lines[offset:end])
        return change_note + content if change_note else content

    result = await process_large_output(
        content, context or path_str, f"read_file_{path.name}", llm, root=root
    )
    return change_note + result if change_note else result


async def _tool_write_file(args: dict, **kwargs) -> str:
    path_str = args.get("path", "")
    content = args.get("content", "")
    mode = args.get("mode", "write")

    if not path_str:
        return "Error: 'path' is required"

    # LLM sometimes passes a dict/list instead of a string
    if not isinstance(content, str):
        content = json.dumps(content, indent=2)

    path = _resolve_path(path_str)

    blocked = _check_write_allowed(path)
    if blocked:
        return blocked

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error writing file: {e}"

    return f"Wrote {len(content)} chars to {path}"


async def _tool_list_directory(args: dict, **kwargs) -> str:
    path_str = args.get("path", ".")
    recursive = args.get("recursive", False)
    max_depth = args.get("max_depth", 3)

    path = _resolve_path(path_str)

    if not path.exists():
        return f"Error: Path not found: {path}"
    if not path.is_dir():
        return f"Error: Not a directory: {path}"

    entries: list[str] = []
    count = 0

    def _walk(p: Path, depth: int) -> None:
        nonlocal count
        if count >= LIST_DIR_MAX_ENTRIES:
            return
        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            entries.append(f"  {'  ' * depth}(permission denied)")
            return
        for item in items:
            if count >= LIST_DIR_MAX_ENTRIES:
                entries.append(f"... (capped at {LIST_DIR_MAX_ENTRIES} entries)")
                return
            prefix = "  " * depth
            suffix = "/" if item.is_dir() else ""
            entries.append(f"{prefix}{item.name}{suffix}")
            count += 1
            if recursive and item.is_dir() and depth < max_depth:
                _walk(item, depth + 1)

    _walk(path, 0)
    return "\n".join(entries) if entries else "(empty directory)"


async def _tool_web_fetch(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    url = args.get("url", "")
    prompt = args.get("prompt", "")
    if not url:
        return "Error: 'url' is required"

    try:
        import httpx
    except ImportError:
        # Fallback to urllib
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Luna-Agent/0.1"})
            resp = await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
            )
            html = resp
        except urllib.error.URLError as e:
            return f"Error fetching URL: {e}"
        except Exception as e:
            return f"Error fetching URL: {e}"
    else:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(url, headers={"User-Agent": "Luna-Agent/0.1"})
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            return f"Error fetching URL: {e}"

    # Convert HTML to markdown
    try:
        import html2text
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0
        content = converter.handle(html)
    except ImportError:
        # Crude fallback: strip tags
        content = re.sub(r"<[^>]+>", "", html)
        content = re.sub(r"\s+", " ", content).strip()

    extraction_context = prompt or context or url
    source = f"web_fetch_{url.split('//')[1].split('/')[0] if '//' in url else 'unknown'}"
    return await process_large_output(content, extraction_context, source, llm, root=root)


async def _tool_web_search(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    query = args.get("query", "")
    max_results = args.get("max_results", 10)
    if not query:
        return "Error: 'query' is required"

    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search"

    try:
        results = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=max_results))
        )
    except Exception as e:
        return f"Error searching: {e}"

    if not results:
        return "No results found."

    output_parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        output_parts.append(f"{i}. **{title}**\n   {href}\n   {body}")

    output = "\n\n".join(output_parts)
    return await process_large_output(output, context or query, f"web_search_{query[:30]}", llm, root=root)


async def _tool_list_available_tools(args: dict, **kwargs) -> str:
    if _mcp_manager is None:
        return "No additional tools available (MCP not configured)."

    query = args.get("query", "").lower()
    lines: list[str] = []
    for server in _mcp_manager.servers.values():
        for tool in server.tools:
            name = tool["name"]
            desc = tool.get("description", "")
            if query and query not in name.lower() and query not in desc.lower():
                continue
            lines.append(f"- **{name}**: {desc}")

    if not lines:
        if query:
            return f"No tools matching '{query}'."
        return "No additional tools available."
    return f"Available tools ({len(lines)}):\n" + "\n".join(lines)


async def _tool_use_tool(args: dict, **kwargs) -> str:
    name = args.get("name", "")
    if not name:
        return "Error: 'name' is required"

    if _mcp_manager is None:
        return "Error: MCP not configured — no external tools available."

    arguments = args.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments else {}

    return await _mcp_manager.call_tool(name, arguments)


# --- Summarize paper (extract-then-summarize) ---

_EXTRACT_PROMPT = """\
You are a precise fact extractor. Given the abstract of a research paper, extract ONLY facts that are \
explicitly stated. Do NOT infer, interpret, or add anything.

Extract these categories:
1. **Method/Model name** — exact name as written
2. **Authors** — if mentioned in the abstract (often not)
3. **Key claims** — quote exact numbers, percentages, and comparisons verbatim
4. **Benchmarks/Datasets** — exact names as written
5. **Domains** — what field or application area

For each fact, quote the relevant text from the abstract.
If a category has no information in the abstract, write "NOT MENTIONED".

Paper title: {title}
Authors: {authors}
Abstract:
{abstract}"""

_SUMMARIZE_PROMPT = """\
You are a precise summarizer. Write a {style} summary of this paper using ONLY the extracted facts below. \
Do NOT add any information, benchmarks, numbers, or claims that are not in the extracts. \
If something is marked "NOT MENTIONED", do not guess or fill it in.

{style_instruction}

Paper title: {title}

Extracted facts:
{extracts}"""

_STYLE_INSTRUCTIONS = {
    "technical": "Write a concise technical summary (3-5 sentences). Focus on the method, key results, and significance.",
    "linkedin": "Write an engaging LinkedIn-style post (3-4 short paragraphs). Use accessible language but stay accurate to the extracts.",
}


async def _tool_summarize_paper(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    arxiv_id = args.get("arxiv_id", "")
    style = args.get("style", "technical")
    if not arxiv_id:
        return "Error: 'arxiv_id' is required"
    if llm is None:
        return "Error: LLM client not available for summarization"
    if style not in _STYLE_INSTRUCTIONS:
        return f"Error: style must be 'technical' or 'linkedin', got '{style}'"

    # Step 1: Fetch paper metadata via MCP paper_db or direct arXiv call
    paper_meta = None
    if _mcp_manager is not None:
        try:
            index_result = await _mcp_manager.call_tool(
                "paper_db__index_paper", {"arxiv_id": arxiv_id}
            )
            log_event(logger, "summarize_paper_indexed", arxiv_id=arxiv_id)
        except Exception as e:
            log_event(logger, "summarize_paper_index_failed", arxiv_id=arxiv_id, error=str(e))

    # Fetch metadata directly via arXiv API (always, to get the abstract)
    try:
        import arxiv as arxiv_lib
        client = arxiv_lib.Client()
        paper = next(client.results(arxiv_lib.Search(id_list=[arxiv_id])))
        paper_meta = {
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors),
            "abstract": paper.summary,
        }
    except Exception as e:
        return f"Error fetching paper from arXiv: {e}"

    # Step 2: Extract facts at low temperature
    extract_messages = [
        {"role": "system", "content": "You are a precise fact extractor. Follow instructions exactly."},
        {"role": "user", "content": _EXTRACT_PROMPT.format(
            title=paper_meta["title"],
            authors=paper_meta["authors"],
            abstract=paper_meta["abstract"],
        )},
    ]

    try:
        extract_response = await llm.chat(extract_messages, temperature=0.2)
        extracts = extract_response.content or "(extraction failed)"
    except Exception as e:
        return f"Error during fact extraction: {e}"

    # Step 3: Summarize from extracts at slightly higher (but still low) temperature
    summarize_messages = [
        {"role": "system", "content": "You are a precise summarizer. Use ONLY the provided extracts."},
        {"role": "user", "content": _SUMMARIZE_PROMPT.format(
            style=style,
            style_instruction=_STYLE_INSTRUCTIONS[style],
            title=paper_meta["title"],
            extracts=extracts,
        )},
    ]

    try:
        summary_response = await llm.chat(summarize_messages, temperature=0.4)
        summary = summary_response.content or "(summarization failed)"
    except Exception as e:
        return f"Error during summarization: {e}"

    # Step 4: Return combined output
    output = (
        f"# {paper_meta['title']}\n"
        f"**Authors:** {paper_meta['authors']}\n\n"
        f"## Summary ({style})\n\n{summary}\n\n"
        f"---\n\n"
        f"## Extracted Facts\n\n{extracts}\n\n"
        f"---\n\n"
        f"## Raw Abstract\n\n{paper_meta['abstract']}"
    )

    log_event(logger, "summarize_paper_complete", arxiv_id=arxiv_id, style=style)
    return output


# --- Delegate sub-agent ---

_DELEGATE_ALLOWED_TOOLS = {"bash", "read_file", "write_file", "list_directory", "web_search", "web_fetch", "summarize_paper", "wiki_read", "wiki_search", "recall"}
_DELEGATE_MAX_ROUNDS = 5

_DELEGATE_SYSTEM_PROMPT = """\
You are a focused sub-agent. Complete the assigned task using the available tools, then provide your final answer.
Be direct and thorough. Do not ask clarifying questions — work with what you have."""


def _get_delegate_tools() -> list[dict[str, Any]]:
    """Return the subset of NATIVE_TOOLS that the delegate sub-agent can use."""
    return [t for t in NATIVE_TOOLS if t["function"]["name"] in _DELEGATE_ALLOWED_TOOLS]


async def _tool_delegate(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return "Error: 'task' is required"
    if llm is None:
        return "Error: LLM client not available for delegation"

    extra_context = args.get("context", "")
    system_content = _DELEGATE_SYSTEM_PROMPT
    if extra_context:
        system_content += f"\n\nContext: {extra_context}"

    tools = _get_delegate_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task},
    ]

    log_event(logger, "delegate_start", task=task[:100])

    response = await llm.chat(messages, tools=tools)
    rounds = 0

    while response.has_tool_calls() and rounds < _DELEGATE_MAX_ROUNDS:
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
            log_event(logger, "delegate_tool", tool=tc.name)
            try:
                if tc.name not in _DELEGATE_ALLOWED_TOOLS:
                    result = f"Error: Tool '{tc.name}' is not available in sub-agent context."
                elif is_native_tool(tc.name):
                    result = await call_native_tool(
                        tc.name, tc.arguments,
                        context=task, llm=llm, root=root,
                    )
                else:
                    result = f"Error: Tool '{tc.name}' is not available in sub-agent context."
            except Exception as e:
                result = f"Tool error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        response = await llm.chat(messages, tools=tools)

    if rounds >= _DELEGATE_MAX_ROUNDS:
        messages.append({
            "role": "user",
            "content": "You have reached the tool limit. Please provide your final answer now.",
        })
        response = await llm.chat(messages)

    final = response.content or "(sub-agent produced no response)"
    log_event(logger, "delegate_complete", rounds=rounds, response_len=len(final))
    return final


# --- Code task sub-agent ---

_CODE_TASK_ALLOWED_TOOLS = {
    "bash", "read_file", "write_file", "list_directory",
    "web_search", "web_fetch",
    "wiki_read", "wiki_write", "wiki_search",
    "recall",
}
_CODE_TASK_MAX_ROUNDS = 25

_CODE_TASK_SYSTEM_PROMPT = """\
You are a coding sub-agent. Your job is to write code that WORKS, not code that looks right.

## Mandatory Workflow
For every piece of code you write, you MUST follow this cycle:
1. WRITE the code (write_file)
2. RUN the code (bash)
3. CHECK the output — did it succeed? Did it produce the expected result?
4. If it FAILED: read the error, diagnose, fix, go back to step 2
5. If it SUCCEEDED: verify the output makes sense (not empty, not error pages, not placeholder data)

NEVER skip steps 2-4. NEVER declare success without running the code.

## Anti-Hallucination Rules
- Do NOT guess API endpoints, DOM selectors, or URL patterns
- If you need to access a website or API: web_search for docs first, web_fetch to read them, then code
- If a library call fails, search for the correct approach — do NOT invent alternatives
- If you cannot verify something works, say so explicitly

## Error Handling
- Non-zero exit codes = FAILED. Read stderr, diagnose, fix, re-run.
- Empty output often = silent failure. Add print statements or assertions.
- HTTP 403/404/500 = wrong URL/API. Research the correct one.
- Import errors = missing package. Install it.

## Completion
Report: (1) what was accomplished, (2) files created/modified, (3) how verified, (4) known issues.

Working directory: {working_dir}
"""


async def _tool_code_task(args: dict, context: str = "", llm=None, root=None, **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return "Error: 'task' is required"
    if llm is None:
        return "Error: LLM client not available for code task"

    # Create working directory within workspace
    working_dir_name = args.get("working_dir", "")
    if not working_dir_name:
        safe_name = re.sub(r"[^\w\-]", "_", task[:40]).strip("_").lower()
        working_dir_name = safe_name
    working_dir = _workspace / working_dir_name if _workspace else Path(working_dir_name)
    working_dir.mkdir(parents=True, exist_ok=True)

    system = _CODE_TASK_SYSTEM_PROMPT.format(working_dir=str(working_dir))
    if args.get("context"):
        system += f"\n\nAdditional context: {args['context']}"

    tools = [t for t in NATIVE_TOOLS if t["function"]["name"] in _CODE_TASK_ALLOWED_TOOLS]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    log_event(logger, "code_task_start", task=task[:100])

    response = await llm.chat(messages, tools=tools, temperature=0.4)
    rounds = 0

    while response.has_tool_calls() and rounds < _CODE_TASK_MAX_ROUNDS:
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
            log_event(logger, "code_task_tool", tool=tc.name)
            try:
                if tc.name not in _CODE_TASK_ALLOWED_TOOLS:
                    result = f"Error: Tool '{tc.name}' is not available in code task context."
                elif is_native_tool(tc.name):
                    result = await call_native_tool(
                        tc.name, tc.arguments,
                        context=task, llm=llm, root=root,
                    )
                else:
                    result = f"Error: Tool '{tc.name}' is not available in code task context."
            except Exception as e:
                result = f"Tool error: {e}"

            result = verify_tool_result(tc.name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        response = await llm.chat(messages, tools=tools, temperature=0.4)

    if rounds >= _CODE_TASK_MAX_ROUNDS:
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool limit. Provide your final report now: "
                "(1) what was accomplished, (2) files created/modified, "
                "(3) how it was verified, (4) known issues or incomplete items."
            ),
        })
        response = await llm.chat(messages, temperature=0.4)

    final = response.content or "(code task produced no response)"
    log_event(logger, "code_task_complete", rounds=rounds, response_len=len(final))
    return final


# --- Newsletter pipeline ---

_NEWSLETTER_DIR = "newsletter"
_NEWSLETTER_TIMEOUT = 900  # 15 minutes — pipeline makes many sequential LLM calls


async def run_newsletter_pipeline(
    max_items: int = 5,
    workspace: Path | None = None,
) -> str:
    """Run the multi-source newsletter pipeline. Callable from tools or directly.

    Returns a formatted result string.
    """
    newsletter_dir = (workspace or Path("data/workspace")) / _NEWSLETTER_DIR

    if not (newsletter_dir / "multi_source_pipeline.py").exists():
        return f"Error: Newsletter pipeline not found at {newsletter_dir}"

    log_event(logger, "newsletter_start", max_items=max_items, directory=str(newsletter_dir))

    cmd = (
        f"cd {newsletter_dir} && "
        f"python3 -c \""
        f"import json; "
        f"from multi_source_pipeline import MultiSourceNewsletterPipeline; "
        f"p = MultiSourceNewsletterPipeline(); "
        f"s = p.run(output_dir='output', max_items_per_source={int(max_items)}); "
        f"print('---JSON_SUMMARY---'); "
        f"print(json.dumps(s, indent=2, default=str))"
        f"\""
    )

    try:
        proc, stdout, stderr, rc = await asyncio.to_thread(
            _run_in_pgroup, cmd, str(newsletter_dir), _NEWSLETTER_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        log_event(logger, "newsletter_timeout", timeout=_NEWSLETTER_TIMEOUT)
        return f"Error: Newsletter pipeline timed out after {_NEWSLETTER_TIMEOUT}s."

    if rc != 0:
        log_event(logger, "newsletter_error", exit_code=rc, stderr=stderr[:500])
        return f"Newsletter pipeline failed (exit code {rc}):\n{stderr[:2000]}\n{stdout[-1000:]}"

    # Extract the JSON summary from stdout
    summary_json = ""
    if "---JSON_SUMMARY---" in stdout:
        summary_json = stdout.split("---JSON_SUMMARY---", 1)[1].strip()

    if summary_json:
        try:
            summary = json.loads(summary_json)
            newsletter_path = summary.get("output_files", {}).get("newsletter", "")
            sources_path = summary.get("output_files", {}).get("sources", "")
            log_event(logger, "newsletter_complete",
                      total_sources=summary.get("total_sources", 0),
                      newsletter_items=summary.get("newsletter_items", 0),
                      duration=summary.get("duration_seconds", 0),
                      newsletter_path=newsletter_path)

            result = (
                f"Newsletter pipeline completed successfully!\n\n"
                f"**Sources fetched:** {summary.get('total_sources', 0)}\n"
                f"**High quality items:** {summary.get('high_quality_sources', 0)}\n"
                f"**Newsletter items:** {summary.get('newsletter_items', 0)}\n"
                f"**Duration:** {summary.get('duration_seconds', 0):.1f}s\n\n"
                f"**Breakdown:**\n"
            )
            breakdown = summary.get("sources_breakdown", {})
            for source, count in breakdown.items():
                result += f"  - {source}: {count}\n"

            result += f"\n**Newsletter file:** {newsletter_path}\n"
            result += f"**Sources file:** {sources_path}\n"

            return result
        except json.JSONDecodeError:
            pass

    # Fallback: return raw output
    log_event(logger, "newsletter_complete_raw", stdout_len=len(stdout))
    return f"Newsletter pipeline output:\n{stdout[-3000:]}"


async def _tool_run_newsletter(args: dict, context: str = "", root: Path | None = None, **kwargs) -> str:
    """Tool wrapper for run_newsletter_pipeline."""
    max_items = args.get("max_items_per_source", 5)
    return await run_newsletter_pipeline(max_items=max_items, workspace=root)


# --- Claude Code collaboration ---


async def _tool_ask_claude_code(args: dict, context: str = "", **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return "Error: 'task' is required"

    from luna.claude_code import get_session_manager, ClaudeCodeError

    manager = get_session_manager()
    if manager is None:
        return "Error: Claude Code integration is not enabled."

    session_id = args.get("session_id")
    session = None

    try:
        if session_id:
            # Continue an existing conversation
            session = manager.get_session(session_id)
            if session is None:
                return (
                    f"Error: Session {session_id[:8]}... not found or expired. "
                    "Start a new session by omitting session_id."
                )
            message = task
        else:
            # Start a new session
            working_dir = args.get("working_dir")
            if not working_dir:
                working_dir = str(_workspace) if _workspace else "."

            session = await manager.create_session(working_dir)

            # Build a rich initial prompt with context
            parts = [task]
            extra_context = args.get("context", "")
            if extra_context:
                parts.append(f"\n\nAdditional context:\n{extra_context}")
            message = "\n".join(parts)

        response = await session.send(message)

        # Register session after first send (that's when session_id is assigned)
        if not session_id and session.session_id:
            manager.register_session(session)

        # Format the result
        result_parts = [response.result]
        result_parts.append("\n---")
        result_parts.append(f"Session: {response.session_id} (pass as session_id to continue)")
        if response.tool_calls_made:
            # Summarize tool usage
            from collections import Counter
            counts = Counter(response.tool_calls_made)
            tool_summary = ", ".join(
                f"{name}({count})" if count > 1 else name
                for name, count in counts.items()
            )
            result_parts.append(f"Tools used: {tool_summary}")
        if response.cost_usd is not None:
            result_parts.append(f"Cost: ${response.cost_usd:.4f}")
        if response.duration_ms is not None:
            result_parts.append(f"Duration: {response.duration_ms / 1000:.1f}s")
        if response.is_error:
            result_parts.insert(0, "**[Claude Code reported an error]**\n")

        return "\n".join(result_parts)

    except ClaudeCodeError as e:
        return f"Claude Code error: {e}"
    except Exception as e:
        log_event(logger, "ask_claude_code_error", error=str(e))
        return f"Unexpected error communicating with Claude Code: {e}"


# --- Wiki tools ---


async def _tool_wiki_read(args: dict, **kwargs) -> str:
    if _wiki_manager is None:
        return "Error: Wiki is not enabled."
    page_path = args.get("path", "index.md")
    try:
        content = _wiki_manager.read_page(page_path)
        # Track wiki reads for change detection
        if _memory_manager is not None:
            full_path = str(_wiki_manager._validate_path(page_path))
            changed, last_seen = _memory_manager.check_file_changed(full_path, content)
            note = ""
            if changed and last_seen is not None:
                from datetime import datetime, timezone
                ts = datetime.fromtimestamp(last_seen, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                note = f"[NOTE: This wiki page has changed since you last read it at {ts}.]\n\n"
            _memory_manager.record_file_read(full_path, content)
            return note + content
        return content
    except FileNotFoundError:
        return f"Error: Wiki page not found: {page_path}"
    except ValueError as e:
        return f"Error: {e}"


async def _tool_wiki_write(args: dict, **kwargs) -> str:
    if _wiki_manager is None:
        return "Error: Wiki is not enabled."
    page_path = args.get("path", "")
    content = args.get("content", "")
    if not page_path:
        return "Error: 'path' is required."
    if not content:
        return "Error: 'content' is required."
    try:
        _wiki_manager.write_page(page_path, content)
        # Record snapshot so we know what we wrote
        if _memory_manager is not None:
            full_path = str(_wiki_manager._validate_path(page_path))
            _memory_manager.record_file_read(full_path, content)
        return f"Wiki page written: {page_path}"
    except ValueError as e:
        return f"Error: {e}"


async def _tool_wiki_search(args: dict, **kwargs) -> str:
    if _wiki_manager is None:
        return "Error: Wiki is not enabled."
    query = args.get("query", "")
    if not query:
        return "Error: 'query' is required."
    # wiki_search uses sync keyword matching (no LLM expansion in tool context)
    import asyncio
    result = await _wiki_manager.query(query)
    return result if result else "No matching wiki pages found."


# --- Memory tools ---


async def _tool_recall(args: dict, **kwargs) -> str:
    if _memory_manager is None:
        return "Error: Memory manager not available."
    query = args.get("query", "")
    if not query:
        return "Error: 'query' is required."
    top_k = args.get("top_k", 10)
    results = _memory_manager.search(query, top_k=top_k)
    if not results:
        return "No matching memories found."
    lines = [f"Found {len(results)} memories:"]
    for r in results:
        lines.append(f"- [{r.memory_type}] {r.content} (importance: {r.importance})")
    return "\n".join(lines)


async def _tool_diff(args: dict, **kwargs) -> str:
    import difflib
    import hashlib

    if _memory_manager is None:
        return "Error: Memory manager not available."
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' is required."

    path = _resolve_path(path_str)
    if not path.exists():
        return f"Error: File not found: {path}"

    snapshot = _memory_manager.get_file_snapshot_content(str(path))
    if snapshot is None:
        return f"No previous snapshot for this file. Read it first with read_file or wiki_read."

    old_content, last_seen = snapshot
    try:
        current_content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied: {path}"

    if hashlib.sha256(current_content.encode()).hexdigest() == hashlib.sha256(old_content.encode()).hexdigest():
        return "No changes detected."

    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(last_seen, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        current_content.splitlines(keepends=True),
        fromfile=f"{path} (last seen {ts})",
        tofile=f"{path} (current)",
    ))
    return "".join(diff_lines) if diff_lines else "No changes detected."


# --- Registry ---

_TOOL_REGISTRY: dict[str, Any] = {
    "bash": _tool_bash,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "list_directory": _tool_list_directory,
    "web_fetch": _tool_web_fetch,
    "web_search": _tool_web_search,
    "list_available_tools": _tool_list_available_tools,
    "use_tool": _tool_use_tool,
    "summarize_paper": _tool_summarize_paper,
    "delegate": _tool_delegate,
    "code_task": _tool_code_task,
    "run_newsletter": _tool_run_newsletter,
    "ask_claude_code": _tool_ask_claude_code,
    "wiki_read": _tool_wiki_read,
    "wiki_write": _tool_wiki_write,
    "wiki_search": _tool_wiki_search,
    "recall": _tool_recall,
    "diff": _tool_diff,
}
