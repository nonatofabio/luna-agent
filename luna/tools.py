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


def init_tool_registry(mcp: "MCPManager") -> None:
    """Register the MCP manager so meta-tools can discover and call MCP tools."""
    global _mcp_manager
    _mcp_manager = mcp


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

    log_event(logger, "native_tool_call", tool=name)
    try:
        return await handler(arguments, context=context, llm=llm, root=root)
    except Exception as e:
        logger.exception(f"Native tool error: {name}")
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

    if offset or limit:
        lines = content.splitlines(keepends=True)
        end = offset + limit if limit else len(lines)
        content = "".join(lines[offset:end])
        # If we sliced, return directly (user was specific)
        return content

    return await process_large_output(
        content, context or path_str, f"read_file_{path.name}", llm, root=root
    )


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

_DELEGATE_ALLOWED_TOOLS = {"bash", "read_file", "write_file", "list_directory", "web_search", "web_fetch", "summarize_paper"}
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
}
