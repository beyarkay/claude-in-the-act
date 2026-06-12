"""Subprocess wrappers for the two external tools we drive: ``git`` and the
``claude`` CLI. Centralised so timeouts, the stdout→result decode, and worktree
locking live in exactly one place (they used to be copy-pasted and had drifted).
"""

from __future__ import annotations

import json
import subprocess
import threading

# A hung `claude` (auth stall, model loop, tool waiting on stdin) must not wedge
# a thread forever — that deadlocks the whole parallel eval. Generous, but bounded.
CLAUDE_TIMEOUT = 900  # seconds
GIT_TIMEOUT = 300  # clone over a slow/dead remote shouldn't block indefinitely

# `git worktree add` against one repo isn't concurrency-safe; serialise it so the
# parallel eval (many captures from the same repo) doesn't race on .git/worktrees.
WORKTREE_LOCK = threading.Lock()


def git(*args: str, timeout: int = GIT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a git command, capturing output. Never raises on non-zero exit; a
    timeout is reported as returncode 124 rather than propagating."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            e.cmd, 124, e.stdout or "", f"git timed out after {timeout}s"
        )


def git_out(*args: str, timeout: int = GIT_TIMEOUT) -> str | None:
    """Run git and return stripped stdout, or None on non-zero exit."""
    proc = git(*args, timeout=timeout)
    return proc.stdout.strip() if proc.returncode == 0 else None


def run_claude(
    prompt: str,
    *,
    model: str | None = None,
    tools: list[str] | None = None,
    cwd=None,
    resume: str | None = None,
    fork: bool = False,
    permission_mode: str | None = None,
    timeout: int = CLAUDE_TIMEOUT,
) -> tuple[str, subprocess.CompletedProcess]:
    """Run a headless ``claude -p`` and return ``(result_text, proc)``.

    ``tools``: ``None`` leaves the harness default; ``[]`` disables tools
    (``--allowed-tools ""``); a list allows exactly those. ``result_text`` is the
    decoded ``.result`` (empty string on timeout, non-zero exit, or bad JSON).
    """
    cmd = ["claude"]
    if resume:
        cmd += ["--resume", resume]
    if fork:
        cmd += ["--fork-session"]
    cmd += ["-p", prompt, "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if tools is not None:
        cmd += ["--allowed-tools", *(tools or [""])]

    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        proc = subprocess.CompletedProcess(
            cmd, 124, e.stdout or "", f"claude timed out after {timeout}s"
        )

    result = ""
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            result = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            pass
    return result, proc
