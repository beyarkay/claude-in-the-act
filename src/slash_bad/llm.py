"""One-shot model calls through the user's own ``claude`` or ``codex`` login.

A model name is a Claude model (``opus``, ``claude-sonnet-5``), ``codex:<model>``
(``codex:`` alone means Codex's default), or ``openrouter:<model>`` (``luna`` is
short for ``openrouter:openai/gpt-6-luna``), which needs ``OPENROUTER_API_KEY``.
Calls are isolated from the user's settings, hooks and tools, and marked so
slash-bad's own hooks ignore them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from slash_bad import REPLAY_ENV

TIMEOUT = 600
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALIASES = {"luna": "openrouter:openai/gpt-6-luna"}
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


def _openrouter(prompt: str, model: str, system: str) -> tuple[str, float | None, str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return "", None, "OPENROUTER_API_KEY is not set"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "usage": {"include": True},
        }
    ).encode()
    headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
    err = ""
    for attempt in range(3):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err = f"openrouter {e.code}: {e.read()[:300]!r}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            err = f"openrouter failed: {e}"
        else:
            choice = (out.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            if text:
                return text, (out.get("usage") or {}).get("cost"), out.get("model") or model
            err = f"openrouter returned no text: {json.dumps(out)[:300]}"
        time.sleep(2**attempt)
    return "", None, err


def ask(prompt: str, model: str, system: str = SYSTEM) -> tuple[str, float | None, str]:
    """``(text, cost, resolved model or error)``; text is empty on failure."""
    model = ALIASES.get(model, model)
    if model.startswith("openrouter:"):
        return _openrouter(prompt, model.removeprefix("openrouter:"), system)
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
