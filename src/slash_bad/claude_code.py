"""Claude Code: read a session transcript, capture a case, replay it.

The flagged message is the last assistant message with text. Everything before
it is the context, kept verbatim. A replay plants that context as a fresh
session, resumes it through ``proxy.SentinelProxy`` so no extra prompt reaches
the model, and denies every tool call, so the new model answers from the same
information the old one had.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from slash_bad import REPLAY_ENV
from slash_bad.proxy import SENTINEL
from slash_bad.store import Case, read_jsonl
from slash_bad.view import as_text, tool_call_text

HARNESS = "claude-code"
SYNTHETIC_MODEL = "<synthetic>"
CLAUDE_TIMEOUT = 900
DENY_REASON = (
    "Permission denied: the user has turned off tool use for this reply. Continue without tools."
)


def _content(rec: dict) -> object:
    return (rec.get("message") or {}).get("content")


def is_prompt(rec: dict) -> bool:
    """A message the user typed, as opposed to a tool result, a slash-command
    echo, or a harness injection such as a task notification."""
    if rec.get("type") != "user" or rec.get("isMeta"):
        return False
    if rec.get("promptSource") == "system":
        return False
    if (rec.get("origin") or {}).get("kind") == "task-notification":
        return False
    content = _content(rec)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return False
        text = as_text(content)
    else:
        return False
    return not text.lstrip().startswith(
        ("<local-command-stdout>", "<local-command-caveat>", "<command-name>")
    ) and bool(text.strip())


def _assistant_text(rec: dict) -> str:
    content = _content(rec)
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _is_real_assistant(rec: dict) -> bool:
    return (
        rec.get("type") == "assistant"
        and (rec.get("message") or {}).get("model") != SYNTHETIC_MODEL
    )


def steps(recs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in recs:
        content = _content(r)
        if is_prompt(r):
            out.append({"kind": "user", "text": as_text(content)})
        elif _is_real_assistant(r) and isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    out.append({"kind": "assistant", "text": b["text"]})
                elif b.get("type") == "tool_use":
                    out.append(
                        {
                            "kind": "tool_call",
                            "name": b.get("name", ""),
                            "text": tool_call_text(b.get("name", ""), b.get("input")),
                        }
                    )
        elif r.get("type") == "user" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append({"kind": "tool_result", "text": as_text(b.get("content"))})
    return out


def split(recs: list[dict]) -> dict:
    """``{cut, flagged, model}``: the context is ``recs[:cut]``; ``flagged`` is
    the text of the last assistant message. Raises ValueError without one."""
    idx = next(
        (
            i
            for i in range(len(recs) - 1, -1, -1)
            if _is_real_assistant(recs[i]) and _assistant_text(recs[i])
        ),
        None,
    )
    if idx is None:
        raise ValueError("no assistant message with text to flag")
    msg_id = (recs[idx].get("message") or {}).get("id")
    cut = idx
    # One API message is logged as several records (thinking, text, tool_use)
    # sharing a message id; the replay must regenerate all of it.
    while (
        cut > 0
        and msg_id
        and recs[cut - 1].get("type") == "assistant"
        and (recs[cut - 1].get("message") or {}).get("id") == msg_id
    ):
        cut -= 1
    flagged = "\n\n".join(
        t for r in recs[cut : idx + 1] if _is_real_assistant(r) and (t := _assistant_text(r))
    )
    return {
        "cut": cut,
        "flagged": flagged,
        "model": (recs[idx].get("message") or {}).get("model", ""),
    }


def _git(cwd: str, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def claude_version() -> str:
    try:
        proc = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip().split(" ")[0]


def build_case(
    transcript: Path,
    complaint: str,
    cwd: str = "",
    session_id: str = "",
    upto: int | None = None,
) -> tuple[dict, list[dict]]:
    """``(meta, context)`` for a new case from a session transcript. With
    ``upto``, the flagged message is the last one before record ``upto``."""
    recs = read_jsonl(transcript)
    if upto is not None:
        recs = recs[:upto]
    sp = split(recs)
    context = recs[: sp["cut"]]
    st = steps(context)
    cwd = cwd or next((r["cwd"] for r in reversed(recs) if r.get("cwd")), "")
    meta = {
        "harness": HARNESS,
        "complaint": complaint.strip(),
        "flagged": sp["flagged"],
        "model": sp["model"],
        "cwd": cwd,
        "sessionId": session_id or next((r["sessionId"] for r in recs if r.get("sessionId")), ""),
        "harnessVersion": next((r["version"] for r in reversed(context) if r.get("version")), ""),
        "git": {
            "sha": _git(cwd, "rev-parse", "HEAD") if cwd else None,
            "branch": _git(cwd, "rev-parse", "--abbrev-ref", "HEAD") if cwd else None,
        },
        "stats": {
            "userMessages": sum(s["kind"] == "user" for s in st),
            "toolCalls": sum(s["kind"] == "tool_call" for s in st),
            "contextRecords": len(context),
            "afterToolResult": bool(st) and st[-1]["kind"] == "tool_result",
        },
    }
    return meta, context


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _deny_settings(tmp: Path) -> str:
    deny = tmp / "deny.json"
    deny.write_text(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": DENY_REASON,
                }
            }
        )
    )
    return json.dumps(
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "*",
                        "hooks": [{"type": "command", "command": f"cat '{deny}'"}],
                    }
                ]
            }
        }
    )


def plant_text(context: list[dict], sid: str) -> str:
    """The context as a session file under a new id (set per record, never by
    text substitution, so the id can't collide with anything in the prose)."""
    return "".join(
        json.dumps({**r, "sessionId": sid} if "sessionId" in r else r) + "\n" for r in context
    )


def resolve_model(model: str) -> str:
    return "opus" if model in ("latest", "") else model


def replay(case: Case, model: str, proxy_url: str, max_turns: int = 4) -> dict:
    """Regenerate the flagged message under ``model``. Returns a replay record:
    ``first`` is the model's first message (text plus any tool call it tried),
    ``text`` is all the prose it produced once its tool calls were refused."""
    tmp = Path(tempfile.mkdtemp(prefix="slash-bad-"))
    orig_cwd = case.meta.get("cwd") or ""
    workdir = Path(orig_cwd) if orig_cwd and Path(orig_cwd).is_dir() else tmp / "work"
    workdir.mkdir(exist_ok=True)
    workdir = workdir.resolve()
    proj = Path.home() / ".claude" / "projects" / _slug(str(workdir))
    created_proj = not proj.exists()
    proj.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    planted = proj / f"{sid}.jsonl"
    context = case.context
    planted.write_text(plant_text(context, sid))

    cmd = [
        "claude",
        "-p",
        SENTINEL,
        "--resume",
        sid,
        "--model",
        resolve_model(model),
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
        "--settings",
        _deny_settings(tmp),
    ]
    env = {**os.environ, "ANTHROPIC_BASE_URL": proxy_url, REPLAY_ENV: "1"}
    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=CLAUDE_TIMEOUT,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", f"claude timed out after {CLAUDE_TIMEOUT}s"
        new = read_jsonl(planted)[len(context) :]
    finally:
        planted.unlink(missing_ok=True)
        if created_proj:
            with contextlib.suppress(OSError):
                proj.rmdir()
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        meta = json.loads(out)
    except json.JSONDecodeError:
        meta = {}
    result = parse_replay(new)
    result.update(
        {
            "kind": "replay",
            "harness": HARNESS,
            "requestedModel": model,
            "costUsd": meta.get("total_cost_usd"),
            "returncode": rc,
        }
    )
    if rc != 0 or meta.get("is_error"):
        result["error"] = (meta.get("result") or err or out)[-600:]
    return result


def parse_replay(new: list[dict]) -> dict:
    """Pull the regenerated message(s) out of the records a replay appended."""
    start = next(
        (
            i
            for i, r in enumerate(new)
            if r.get("type") == "user" and SENTINEL in as_text(_content(r))
        ),
        None,
    )
    tail = new[start + 1 :] if start is not None else new
    assistants = [r for r in tail if _is_real_assistant(r)]
    first_id = (assistants[0].get("message") or {}).get("id") if assistants else None
    first_text, first_tools, all_text, attempts = [], [], [], []
    for r in assistants:
        is_first = (r.get("message") or {}).get("id") == first_id
        for b in _content(r) or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                all_text.append(b["text"].strip())
                if is_first:
                    first_text.append(b["text"].strip())
            elif b.get("type") == "tool_use":
                call = tool_call_text(b.get("name", ""), b.get("input"))
                attempts.append(call)
                if is_first:
                    first_tools.append(call)
    return {
        "model": next(((r.get("message") or {}).get("model") for r in assistants), ""),
        "first": "\n\n".join(first_text),
        "firstToolCalls": first_tools,
        "text": "\n\n".join(all_text),
        "toolAttempts": attempts,
    }
