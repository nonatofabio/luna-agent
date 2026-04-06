#!/usr/bin/env python3
"""Luna Agent TUI Dashboard -- real-time system health monitor.

Usage:
    python tools/luna_tui.py

Keys:
    s  Generate AI health summary (uses local LLM)
    r  Force immediate refresh
    q  Quit
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import select
import sqlite3
import subprocess
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "memory.db"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

REFRESH_INTERVAL = 5  # seconds
LLM_HEALTH_URL = "http://localhost:8001/health"
SERVICES = ["luna-agent", "worker-agent"]


# ---------------------------------------------------------------------------
# Data Collectors
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 5, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def _format_uptime(seconds: float) -> str:
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    mins = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _format_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def collect_services() -> dict:
    """Query systemd service status and LLM health endpoint."""
    try:
        result: dict[str, Any] = {}
        now = time.time()

        for svc in SERVICES:
            try:
                r = _run([
                    "systemctl", "show", svc,
                    "--property=ActiveState,SubState,MainPID,MemoryCurrent,"
                    "ExecMainStartTimestampMonotonic",
                ])
                props: dict[str, str] = {}
                for line in r.stdout.strip().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        props[k] = v

                state = props.get("SubState", props.get("ActiveState", "unknown"))
                pid = int(props.get("MainPID", "0"))
                mem_bytes = int(props.get("MemoryCurrent", "0"))
                # Compute uptime from monotonic timestamp
                start_mono = int(props.get("ExecMainStartTimestampMonotonic", "0"))
                if start_mono > 0 and state == "running":
                    # systemd monotonic is in microseconds
                    # Get system monotonic offset
                    with open("/proc/uptime") as f:
                        system_uptime = float(f.read().split()[0])
                    boot_time = now - system_uptime
                    start_time = boot_time + (start_mono / 1_000_000)
                    uptime_s = now - start_time
                    uptime = _format_uptime(max(0, uptime_s))
                else:
                    uptime = "-"

                result[svc] = {
                    "state": state,
                    "pid": pid if pid > 0 else None,
                    "memory": _format_bytes(mem_bytes) if mem_bytes > 0 else "-",
                    "uptime": uptime,
                }
            except Exception as e:
                result[svc] = {"state": "error", "error": str(e)}

        # LLM health check
        try:
            r = _run(["curl", "-s", "--max-time", "2", LLM_HEALTH_URL])
            if r.returncode == 0 and r.stdout.strip():
                health = json.loads(r.stdout)
                if health.get("status") == "ok":
                    result["llm_health"] = "ok"
                elif "error" in health:
                    result["llm_health"] = "loading"
                else:
                    result["llm_health"] = "unknown"
            else:
                result["llm_health"] = "unreachable"
        except Exception:
            result["llm_health"] = "unreachable"

        return result
    except Exception as e:
        return {"error": str(e)}


def collect_gpu() -> dict:
    """Query nvidia-smi for GPU stats."""
    try:
        r = _run([
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ])
        if r.returncode != 0:
            return {"error": f"nvidia-smi exit {r.returncode}: {r.stderr.strip()[:80]}"}

        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1].replace("NVIDIA GeForce ", ""),
                "temp": int(parts[2]),
                "mem_used": int(parts[3].replace(" MiB", "")),
                "mem_total": int(parts[4].replace(" MiB", "")),
                "util": int(parts[5].replace(" %", "")),
            })
        return {"gpus": gpus}
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except Exception as e:
        return {"error": str(e)}


def collect_activity() -> dict:
    """Parse JSONL logs and journalctl for activity stats."""
    try:
        messages_24h = 0
        errors_24h = 0
        tool_calls_24h = 0
        recent_errors: list[dict] = []

        cutoff = time.time() - 86400

        # Parse JSONL log files from last 2 days
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        for day in [yesterday, today]:
            log_file = LOG_DIR / f"luna-{day}.jsonl"
            if not log_file.exists():
                continue
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        ts_str = entry.get("timestamp", "")
                        try:
                            ts = datetime.fromisoformat(ts_str).timestamp()
                        except (ValueError, TypeError):
                            ts = 0
                        if ts < cutoff:
                            continue

                        msg = entry.get("message", "")
                        level = entry.get("level", "")

                        if msg == "discord_message":
                            messages_24h += 1
                        if msg == "tool_executing":
                            tool_calls_24h += 1
                        if level == "ERROR":
                            errors_24h += 1
                            ts_short = ts_str[5:16].replace("T", " ") if ts_str else "?"
                            error_msg = entry.get("error", msg)
                            recent_errors.append({"time": ts_short, "message": str(error_msg)[:120]})
            except Exception:
                continue

        # LLM performance from journalctl (worker-agent)
        prompt_speed = None
        gen_speed = None
        try:
            r = _run([
                "journalctl", "-u", "worker-agent", "--since", "24 hours ago",
                "--no-pager", "-o", "cat",
            ], timeout=10)
            speed_re = re.compile(r"(\d+\.?\d*)\s+tokens per second")
            for line in reversed(r.stdout.splitlines()):
                match = speed_re.search(line)
                if not match:
                    continue
                speed = float(match.group(1))
                if "prompt eval time" in line and prompt_speed is None:
                    prompt_speed = speed
                elif "eval time" in line and "prompt" not in line and gen_speed is None:
                    gen_speed = speed
                if prompt_speed is not None and gen_speed is not None:
                    break
        except Exception:
            pass

        return {
            "messages_24h": messages_24h,
            "errors_24h": errors_24h,
            "tool_calls_24h": tool_calls_24h,
            "recent_errors": recent_errors[-5:],  # last 5
            "prompt_speed": prompt_speed,
            "gen_speed": gen_speed,
        }
    except Exception as e:
        return {"error": str(e)}


def collect_memory_db() -> dict:
    """Query memory.db for stats (read-only)."""
    try:
        if not DB_PATH.exists():
            return {"error": "memory.db not found"}

        db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            memories = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            messages = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            sessions = db.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
            summaries = db.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        finally:
            db.close()

        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

        return {
            "memories": memories,
            "messages": messages,
            "sessions": sessions,
            "summaries": summaries,
            "db_size_mb": round(db_size_mb, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def collect_git() -> dict:
    """Get recent git commits."""
    try:
        r = _run(["git", "log", "--oneline", "-8"], cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            return {"error": r.stderr.strip()[:80]}
        commits = []
        for line in r.stdout.strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commits.append({"hash": parts[0], "message": parts[1]})
        return {"commits": commits}
    except Exception as e:
        return {"error": str(e)}


def collect_all() -> dict:
    """Run all collectors and return combined data."""
    return {
        "services": collect_services(),
        "gpu": collect_gpu(),
        "activity": collect_activity(),
        "memory_db": collect_memory_db(),
        "git": collect_git(),
        "collected_at": datetime.now(),
    }


# ---------------------------------------------------------------------------
# Panel Builders
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "running": "[green]●[/green]",
    "active": "[green]●[/green]",
    "inactive": "[red]●[/red]",
    "dead": "[red]●[/red]",
    "failed": "[red]●[/red]",
    "error": "[yellow]●[/yellow]",
}

HEALTH_STYLE = {
    "ok": "[green]ok[/green]",
    "loading": "[yellow]loading[/yellow]",
    "unreachable": "[red]unreachable[/red]",
    "unknown": "[yellow]unknown[/yellow]",
}


def build_services_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="Services", border_style="yellow")

    lines: list[str] = []
    for svc in SERVICES:
        info = data.get(svc, {})
        state = info.get("state", "unknown")
        icon = STATUS_ICONS.get(state, "[dim]●[/dim]")
        line = f"  {icon} [bold]{svc}[/bold]  {state}"
        lines.append(line)
        if state == "running":
            details = f"    {info.get('uptime', '?')} | {info.get('memory', '?')} | PID {info.get('pid', '?')}"
            lines.append(f"[dim]{details}[/dim]")
        elif "error" in info:
            lines.append(f"    [yellow]{info['error'][:60]}[/yellow]")

    health = data.get("llm_health", "unknown")
    health_str = HEALTH_STYLE.get(health, f"[dim]{health}[/dim]")
    lines.append(f"  LLM health: {health_str}")

    content = Text.from_markup("\n".join(lines))
    return Panel(content, title="Services", border_style="blue", padding=(0, 1))


def _temp_style(temp: int) -> str:
    if temp >= 75:
        return "red"
    if temp >= 50:
        return "yellow"
    return "green"


def _vram_bar(used: int, total: int, width: int = 16) -> str:
    ratio = used / total if total > 0 else 0
    filled = int(ratio * width)
    empty = width - filled
    style = "green" if ratio < 0.8 else ("yellow" if ratio < 0.95 else "red")
    return f"[{style}]{'|' * filled}[/{style}][dim]{'|' * empty}[/dim]"


def build_gpu_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="GPUs", border_style="yellow")

    gpus = data.get("gpus", [])
    if not gpus:
        return Panel(Text("No GPUs detected", style="dim"), title="GPUs", border_style="dim")

    lines: list[str] = []
    for g in gpus:
        temp_s = _temp_style(g["temp"])
        bar = _vram_bar(g["mem_used"], g["mem_total"])
        used_gb = g["mem_used"] / 1024
        total_gb = g["mem_total"] / 1024
        lines.append(
            f"  GPU {g['index']}: [bold]{g['name']}[/bold]  "
            f"[{temp_s}]{g['temp']}C[/{temp_s}]"
        )
        lines.append(
            f"    {bar} {used_gb:.0f}/{total_gb:.0f} GB  "
            f"util: {g['util']}%"
        )

    content = Text.from_markup("\n".join(lines))
    return Panel(content, title="GPUs", border_style="blue", padding=(0, 1))


def build_activity_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="Activity (24h)", border_style="yellow")

    errors = data.get("errors_24h", 0)
    err_style = "bold red" if errors > 0 else "green"

    line1 = (
        f"  Messages: [bold]{data.get('messages_24h', 0)}[/bold]  |  "
        f"Errors: [{err_style}]{errors}[/{err_style}]  |  "
        f"Tool calls: [bold]{data.get('tool_calls_24h', 0)}[/bold]"
    )

    ps = data.get("prompt_speed")
    gs = data.get("gen_speed")
    ps_str = f"{ps:.0f} tok/s" if ps else "[dim]n/a[/dim]"
    gs_str = f"{gs:.0f} tok/s" if gs else "[dim]n/a[/dim]"
    line2 = f"  Prompt: {ps_str}  |  Generation: {gs_str}"

    content = Text.from_markup(f"{line1}\n{line2}")
    return Panel(content, title="Activity (24h)", border_style="blue", padding=(0, 1))


def build_memory_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="Memory DB", border_style="yellow")

    line = (
        f"  Memories: [bold]{data.get('memories', 0)}[/bold]  |  "
        f"Messages: [bold]{data.get('messages', 0)}[/bold]  |  "
        f"Sessions: [bold]{data.get('sessions', 0)}[/bold]  |  "
        f"Summaries: [bold]{data.get('summaries', 0)}[/bold]  |  "
        f"DB: {data.get('db_size_mb', 0)} MB"
    )
    content = Text.from_markup(line)
    return Panel(content, title="Memory DB", border_style="blue", padding=(0, 1))


def build_commits_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="Recent Commits", border_style="yellow")

    commits = data.get("commits", [])
    if not commits:
        return Panel(Text("No commits", style="dim"), title="Recent Commits", border_style="dim")

    lines = []
    for c in commits[:6]:
        lines.append(f"  [dim]{c['hash']}[/dim] {c['message']}")

    content = Text.from_markup("\n".join(lines))
    return Panel(content, title="Recent Commits", border_style="blue", padding=(0, 1))


def build_errors_panel(data: dict) -> Panel:
    if "error" in data:
        return Panel(Text(data["error"], style="yellow"), title="Recent Errors", border_style="yellow")

    errors = data.get("recent_errors", [])
    if not errors:
        return Panel(
            Text.from_markup("  [green]No errors in the last 24h[/green]"),
            title="Recent Errors", border_style="green", padding=(0, 1),
        )

    lines = []
    for e in errors[-3:]:
        lines.append(f"  [dim]{e['time']}[/dim] [red]{e['message']}[/red]")

    content = Text.from_markup("\n".join(lines))
    return Panel(content, title="Recent Errors", border_style="red", padding=(0, 1))


def build_ai_panel(summary: str | None) -> Panel:
    if summary is None:
        content = Text.from_markup("  [dim]Press [bold]s[/bold] to generate AI summary...[/dim]")
        style = "dim"
    elif summary == "_generating_":
        content = Text.from_markup("  [yellow]Generating AI summary...[/yellow]")
        style = "yellow"
    else:
        content = Text(f"  {summary}", style="white")
        style = "green"
    return Panel(content, title="AI Summary", border_style=style, padding=(0, 1))


def build_dashboard(data: dict, ai_summary: str | None) -> Layout:
    collected_at = data.get("collected_at", datetime.now())
    ts = collected_at.strftime("%H:%M:%S")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="top_row", size=8),
        Layout(name="activity", size=4),
        Layout(name="memory", size=3),
        Layout(name="commits", size=min(2 + len(data.get("git", {}).get("commits", [])[:6]), 8)),
        Layout(name="errors", size=5),
        Layout(name="ai_summary", size=5),
        Layout(name="footer", size=1),
    )

    # Header
    header = Text(f" Luna Agent Dashboard {' ' * 40} {ts} ", style="bold white on blue")
    layout["header"].update(header)

    # Top row: services + GPUs side by side
    layout["top_row"].split_row(
        Layout(name="services"),
        Layout(name="gpus"),
    )
    layout["services"].update(build_services_panel(data.get("services", {})))
    layout["gpus"].update(build_gpu_panel(data.get("gpu", {})))

    # Activity
    layout["activity"].update(build_activity_panel(data.get("activity", {})))

    # Memory DB
    layout["memory"].update(build_memory_panel(data.get("memory_db", {})))

    # Commits
    layout["commits"].update(build_commits_panel(data.get("git", {})))

    # Errors
    layout["errors"].update(build_errors_panel(data.get("activity", {})))

    # AI Summary
    layout["ai_summary"].update(build_ai_panel(ai_summary))

    # Footer
    footer = Text.from_markup(
        " [bold]s[/bold] Summary  [bold]r[/bold] Refresh  [bold]q[/bold] Quit"
        f"    [dim]Auto-refresh: {REFRESH_INTERVAL}s[/dim]"
    )
    layout["footer"].update(footer)

    return layout


# ---------------------------------------------------------------------------
# AI Summary
# ---------------------------------------------------------------------------


def _format_data_for_llm(data: dict) -> str:
    """Convert collected data dict to human-readable text for the LLM."""
    lines: list[str] = []

    # Services
    svc_data = data.get("services", {})
    for svc in SERVICES:
        info = svc_data.get(svc, {})
        lines.append(f"Service {svc}: {info.get('state', 'unknown')}, "
                     f"uptime={info.get('uptime', '?')}, memory={info.get('memory', '?')}")
    lines.append(f"LLM health: {svc_data.get('llm_health', 'unknown')}")

    # GPUs
    gpu_data = data.get("gpu", {})
    for g in gpu_data.get("gpus", []):
        lines.append(f"GPU {g['index']}: {g['name']}, {g['temp']}C, "
                     f"{g['mem_used']}/{g['mem_total']} MiB VRAM, {g['util']}% util")

    # Activity
    act = data.get("activity", {})
    lines.append(f"Activity (24h): {act.get('messages_24h', 0)} messages, "
                 f"{act.get('errors_24h', 0)} errors, {act.get('tool_calls_24h', 0)} tool calls")
    if act.get("prompt_speed"):
        lines.append(f"LLM speed: prompt={act['prompt_speed']:.0f} tok/s, gen={act.get('gen_speed', 0):.0f} tok/s")
    for e in act.get("recent_errors", []):
        lines.append(f"Error [{e['time']}]: {e['message']}")

    # Memory DB
    mem = data.get("memory_db", {})
    if "error" not in mem:
        lines.append(f"Memory DB: {mem.get('memories', 0)} memories, {mem.get('messages', 0)} messages, "
                     f"{mem.get('sessions', 0)} sessions, {mem.get('db_size_mb', 0)} MB")

    # Git
    git_data = data.get("git", {})
    commits = git_data.get("commits", [])
    if commits:
        lines.append(f"Recent commits ({len(commits)}): {commits[0]['hash']} {commits[0]['message']}")

    return "\n".join(lines)


async def generate_ai_summary(data: dict) -> str:
    """Generate an AI health summary using the local LLM."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from luna.config import load_config
        from luna.llm import create_llm_client

        config = load_config(PROJECT_ROOT / "config.toml")
        config.llm.max_tokens = 512
        llm = create_llm_client(config.llm)

        data_text = _format_data_for_llm(data)

        messages = [
            {"role": "system", "content": (
                "You are a system health analyst for Luna, a local AI agent running on a homelab "
                "with dual RTX 3090 GPUs. Given the current system metrics, provide a brief 3-5 "
                "sentence health summary. Note anything concerning (high temps, errors, low VRAM, "
                "services down). Be concise and practical. No markdown formatting, no bullet points."
            )},
            {"role": "user", "content": f"Current system state:\n\n{data_text}"},
        ]

        response = await llm.chat(messages, temperature=0.3)
        return response.content or "(Empty response from LLM)"

    except ImportError as e:
        return f"Error: could not import Luna package: {e}"
    except Exception as e:
        return f"Error generating summary: {e}"


# ---------------------------------------------------------------------------
# Keyboard Handling
# ---------------------------------------------------------------------------


def keyboard_listener(events: dict, stop_event: threading.Event) -> None:
    """Read single keypresses in cbreak mode. Runs as daemon thread."""
    fd = sys.stdin.fileno()
    while not stop_event.is_set():
        # Use select to check if input is available (100ms timeout)
        readable, _, _ = select.select([fd], [], [], 0.1)
        if readable:
            try:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if ch == "q":
                events["quit"] = True
                break
            elif ch == "s":
                events["ai_summary"] = True
            elif ch == "r":
                events["refresh"] = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    console = Console()
    ai_summary: str | None = None
    events: dict[str, bool] = {}
    stop_event = threading.Event()

    # Initial data collection
    data = collect_all()

    # Terminal raw mode setup
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        with Live(
            build_dashboard(data, ai_summary),
            console=console,
            refresh_per_second=2,
            screen=True,
        ) as live:
            # Set cbreak mode (single-char input, Ctrl+C still works)
            tty.setcbreak(fd)

            # Start keyboard listener
            kb_thread = threading.Thread(
                target=keyboard_listener, args=(events, stop_event), daemon=True,
            )
            kb_thread.start()

            last_collect = time.monotonic()

            while True:
                # Check keyboard events
                if events.pop("quit", False):
                    break

                if events.pop("refresh", False):
                    data = collect_all()
                    last_collect = time.monotonic()

                if events.pop("ai_summary", False):
                    ai_summary = "_generating_"
                    live.update(build_dashboard(data, ai_summary))
                    live.refresh()
                    ai_summary = asyncio.run(generate_ai_summary(data))

                # Auto-refresh
                now = time.monotonic()
                if now - last_collect >= REFRESH_INTERVAL:
                    data = collect_all()
                    last_collect = now

                live.update(build_dashboard(data, ai_summary))
                time.sleep(0.3)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
