"""Tool record-replay for self-contained, codebase-free resume.

A captured transcript already contains every tool call and the result the model
saw. To replay a tool-using turn months later (when the codebase is gone), we
re-run the model with tools ON but intercept each tool call via Claude Code
hooks and feed back the RECORDED result — so nothing depends on the filesystem
and nothing executes for real.

Mechanism (validated):
- ``PreToolUse`` sees the model's call, matches it against the recording, and
  rewrites the input (``updatedInput``) to a trivially-succeeding no-op that
  *produces the recorded output natively*:
    * Read -> point file_path at a temp file holding the recorded content
    * Bash -> rewrite command to ``cat`` the recorded stdout (+ exit code)
  On a cache miss (the model diverged from the original) it denies the call and
  logs the divergence.

This module is both a library (``build_recording``, ``denumber``) and the hook
entry point (``python -m claude_in_the_act.replay``), dispatched on the
``hook_event_name`` + ``tool_name`` in the stdin payload. It reads the recording
from ``CITA_RECORDING`` and writes scratch files under ``CITA_STASH``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Tools we replay by hand (≈95% of real usage). Everything else -> cache miss.
REPLAYED_TOOLS = {"Read", "Bash", "Write", "Edit"}


def _scratch_for(stash: Path, file_path: str) -> Path:
    """Stable scratch path for an original file path. Read/Edit/Write of the same
    file map to the SAME scratch, so the read-before-edit invariant holds and
    Edit inherits the content a prior Read placed there."""
    h = hashlib.sha1(file_path.encode()).hexdigest()[:16]
    return stash / f"file-{h}"


_LINENO = re.compile(r"^\s*\d+\t")


def denumber(rendered: str) -> str:
    """Recover raw file content from Read's line-numbered rendering.

    Read shows ``<lineno>\\t<line>``; strip that prefix per line so the content
    can be written back to a temp file and re-rendered natively.
    """
    return "\n".join(_LINENO.sub("", line) for line in rendered.split("\n"))


def build_recording(recs: list[dict]) -> list[dict]:
    """Extract ordered tool calls + their recorded results from a transcript.

    Pairs each assistant ``tool_use`` with the following ``tool_result`` (by
    tool_use_id). Returns ``[{tool_name, input, content, is_error}]``.
    """
    uses: dict[str, dict] = {}
    order: list[str] = []
    results: dict[str, dict] = {}
    for r in recs:
        msg = r.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if r.get("type") == "assistant" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_use":
                    uses[b["id"]] = {"name": b["name"], "input": b["input"]}
                    order.append(b["id"])
        if r.get("type") == "user" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = b

    recording = []
    for tid in order:
        u = uses[tid]
        res = results.get(tid, {})
        rc = res.get("content")
        text = rc if isinstance(rc, str) else json.dumps(rc)
        recording.append(
            {
                "tool_name": u["name"],
                "input": u["input"],
                "content": text,
                "is_error": bool(res.get("is_error")),
            }
        )
    return recording


def match(recording: list[dict], tool_name: str, tool_input: dict) -> dict | None:
    """Find the recorded entry for a call. Exact (tool, input) first; then a
    tool-specific fuzzy fallback so cwd-dependent absolute paths still match
    (the model passes absolute paths that differ between capture and replay cwd).
    """
    for entry in recording:
        if entry["tool_name"] == tool_name and entry["input"] == tool_input:
            return entry
    if tool_name == "Read":
        base = os.path.basename(tool_input.get("file_path", ""))
        if base:
            for entry in recording:
                if (
                    entry["tool_name"] == "Read"
                    and os.path.basename(entry["input"].get("file_path", "")) == base
                ):
                    return entry
    return None


# --- hook entry point -------------------------------------------------------


def _allow(updated_input: dict) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def handle_pre(payload: dict, recording: list[dict], stash: Path) -> dict | None:
    """PreToolUse: rewrite a matched call into a trivially-succeeding no-op that
    yields the recorded output. Returns the hook output dict, or None to stay
    out of the way."""
    tool = payload.get("tool_name", "")
    tin = payload.get("tool_input", {})
    tid = payload.get("tool_use_id", "x")

    if tool not in REPLAYED_TOOLS:
        return _deny(f"cita-replay: tool {tool!r} not replayed (cache miss)")

    # Write / Edit are ACTIONS, not observations: the model doesn't need recorded
    # data back, only for them to succeed. Redirect at a scratch file so they
    # succeed with no real side effect and no recording lookup — the model is
    # free to "act" as part of its response without touching the filesystem.
    if tool == "Write":
        scratch = _scratch_for(stash, tin.get("file_path", tid))
        return _allow({**tin, "file_path": str(scratch)})

    if tool == "Edit":
        scratch = _scratch_for(stash, tin.get("file_path", tid))
        if not scratch.exists():
            # No prior Read seeded it: put old_string there so Edit can replace.
            scratch.write_text(tin.get("old_string") or "")
        return _allow({**tin, "file_path": str(scratch)})

    # Read / Bash are OBSERVATIONS: must be served from the recording.
    entry = match(recording, tool, tin)
    if entry is None:
        if tool == "Bash":
            # Let a divergent command run natively against the seeded files
            # (handles cwd-relative path differences vs the recorded command).
            return None
        return _deny(f"cita-replay: no recorded {tool} call matching this input")

    if tool == "Read":
        scratch = _scratch_for(stash, tin.get("file_path", tid))
        scratch.write_text(denumber(entry["content"]))
        return _allow({**tin, "file_path": str(scratch)})

    if tool == "Bash":
        scratch = stash / f"{tid}.bash"
        scratch.write_text(entry["content"])
        cmd = f"cat {scratch}"
        if entry["is_error"]:
            cmd += "; exit 1"
        return _allow({**tin, "command": cmd})

    return _deny(f"cita-replay: {tool} unhandled")


def main() -> int:
    payload = json.load(sys.stdin)
    recording = json.loads(Path(os.environ["CITA_RECORDING"]).read_text())
    stash = Path(os.environ["CITA_STASH"])
    stash.mkdir(parents=True, exist_ok=True)

    # Log every interception for divergence analysis.
    with (stash / "replay.log").open("a") as log:
        log.write(
            json.dumps(
                {
                    "event": payload.get("hook_event_name"),
                    "tool": payload.get("tool_name"),
                    "input": payload.get("tool_input"),
                }
            )
            + "\n"
        )

    if payload.get("hook_event_name") == "PreToolUse":
        out = handle_pre(payload, recording, stash)
        if out is not None:
            print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
