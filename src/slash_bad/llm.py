"""One-shot model calls through the user's own ``claude`` or ``codex`` login.

A model name is a Claude model (``opus``, ``claude-sonnet-5``) or ``codex:<model>``
(``codex:`` alone means Codex's default). Calls are isolated from the user's
settings, hooks and tools, and marked so slash-bad's own hooks ignore them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from slash_bad import REPLAY_ENV

TIMEOUT = 600
SYSTEM = "You answer with a single JSON object and nothing else."


def default_model() -> str:
    """Latest Opus through Claude Code if it's installed, else Codex's default."""
    if shutil.which("claude") or not shutil.which("codex"):
        return "opus"
    return "codex:"


def _claude(prompt: str, model: str, system: str) -> tuple[str, float | None, str]:
    with tempfile.TemporaryDirectory(prefix="slash-bad-llm-") as cwd:
        try:
            proc = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--model",
                    model,
                    "--system-prompt",
                    system,
                    "--tools",
                    "",
                    "--setting-sources",
                    "",
                    "--strict-mcp-config",
                    "--no-session-persistence",
                    "--output-format",
                    "json",
                ],
                input=prompt,
                cwd=cwd,
                env={**os.environ, REPLAY_ENV: "1"},
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return "", None, f"claude failed: {e}"
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return "", None, (proc.stderr or proc.stdout)[-300:]
    if out.get("is_error"):
        return "", out.get("total_cost_usd"), str(out.get("result"))[-300:]
    resolved = next(iter(out.get("modelUsage") or {}), model)
    return out.get("result") or "", out.get("total_cost_usd"), resolved


def _codex(prompt: str, model: str, system: str) -> tuple[str, float | None, str]:
    with tempfile.TemporaryDirectory(prefix="slash-bad-llm-") as cwd:
        out_file = Path(cwd) / "out.txt"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-C", cwd]
        cmd += ["-c", 'sandbox_mode="read-only"', "-o", str(out_file)]
        if model:
            cmd += ["-m", model]
        try:
            proc = subprocess.run(
                [*cmd, "-"],
                input=system + "\n\n" + prompt,
                cwd=cwd,
                env={**os.environ, REPLAY_ENV: "1"},
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return "", None, f"codex failed: {e}"
        if proc.returncode != 0 or not out_file.exists():
            return "", None, (proc.stderr or proc.stdout)[-300:]
        return out_file.read_text(), None, model or "codex default"


def ask(prompt: str, model: str, system: str = SYSTEM) -> tuple[str, float | None, str]:
    """``(text, cost, resolved model or error)``; text is empty on failure."""
    if model.startswith("codex:"):
        return _codex(prompt, model.removeprefix("codex:"), system)
    return _claude(prompt, model, system)


def parse_json(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def ask_json(prompt: str, model: str) -> dict | None:
    text, _, _ = ask(prompt, model)
    return parse_json(text) if text else None
