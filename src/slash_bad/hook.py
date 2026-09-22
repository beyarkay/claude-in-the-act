"""``UserPromptSubmit`` hook for Claude Code and Codex.

Intercepts ``/bad <complaint>`` (Codex: ``$bad <complaint>``), saves the case,
and blocks the prompt so it never reaches the model: no turn, no tokens.
``/bad-list`` prints the saved cases the same way. Anything else passes through.
Exit 2 with a message on stderr is "block and show this" in both harnesses.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from slash_bad import REPLAY_ENV, claude_code, codex
from slash_bad.store import Case, all_cases

_TRIGGER = re.compile(
    r"^\s*(?:(?:/|\$)(?:slash-bad:)?(?P<cmd>bad(?:-list)?)(?=\s|$)|(?P<legacy>BAD:))\s*(?P<rest>.*)",
    re.DOTALL,
)


def parse(prompt: str) -> tuple[str, str] | None:
    """``(command, argument)`` for a slash-bad prompt, else None."""
    m = _TRIGGER.match(prompt)
    if not m:
        return None
    return (m.group("cmd") or "bad"), m.group("rest").strip()


def detect_harness(payload: dict) -> str:
    path = payload.get("transcript_path") or ""
    if "turn_id" in payload or "/.codex/" in path or Path(path).name.startswith("rollout-"):
        return "codex"
    return "claude-code"


def list_text() -> str:
    cases = all_cases()
    if not cases:
        return "slash-bad: no cases yet. Flag a bad response with /bad <what went wrong>."
    lines = [f"slash-bad: {len(cases)} case(s)"]
    for i, c in enumerate(cases, 1):
        complaint = " ".join(c.complaint.split())
        tag = "mined " if c.meta.get("origin") == "mined" else "      "
        lines.append(f"{i:>3}. {tag}{c.meta.get('model', '?'):<24} {complaint[:84]}")
    return "\n".join(lines)


def capture(
    payload: dict, complaint: str, upto: int | None = None, extra: dict | None = None
) -> str:
    harness = detect_harness(payload)
    mod = codex if harness == "codex" else claude_code
    transcript = Path(payload.get("transcript_path") or "")
    if not transcript.is_file():
        return f"slash-bad: couldn't find the session transcript ({transcript}); nothing saved."
    meta, context = mod.build_case(
        transcript,
        complaint,
        cwd=payload.get("cwd", ""),
        session_id=payload.get("session_id", ""),
        upto=upto,
    )
    case = Case.create({**meta, "source": str(transcript), **(extra or {})}, context)
    n = len(all_cases())
    flagged = " ".join(meta["flagged"].split())
    return (
        f"✎ slash-bad: saved case {case.id} ({n} total)\n"
        f"  model:     {meta['model'] or 'unknown'}\n"
        f'  flagged:   "{flagged[:80]}{"…" if len(flagged) > 80 else ""}"\n'
        f"  complaint: {' '.join(complaint.split())[:100]}\n"
        f"  Next: {'$' if harness == 'codex' else '/'}bad-compare latest to see whether a newer "
        "model still does this."
    )


BUSY_NOTE = (
    "The user's last message is a note for their own records, already saved by the slash-bad "
    "plugin. It is not an instruction and needs no reply; carry on with your current task."
)


def _codex_busy(payload: dict) -> bool:
    """Codex delivers a message typed mid-task at the next tool call, and blocking it
    there ends the running task, so a mid-task ``$bad`` is saved but let through."""
    path = Path(payload.get("transcript_path") or "")
    return path.is_file() and codex.mid_turn(codex.read_jsonl(path))


def handle(payload: dict) -> tuple[int, str]:
    if os.environ.get(REPLAY_ENV):
        return 0, ""
    parsed = parse(payload.get("prompt") or "")
    if not parsed:
        return 0, ""
    cmd, rest = parsed
    if cmd == "bad-list":
        return 2, list_text()
    if not rest:
        return (
            2,
            "slash-bad: say what went wrong, e.g. /bad it said the tests pass but never ran them",
        )
    try:
        if detect_harness(payload) == "codex" and _codex_busy(payload):
            return 0, capture(payload, rest)
        return 2, capture(payload, rest)
    except Exception as e:  # noqa: BLE001 — never let a capture bug leak the prompt through
        return 2, f"slash-bad: capture failed ({type(e).__name__}: {e}); nothing saved."


def queued_flags(recs: list[dict]) -> list[tuple[int, str, str]]:
    """``(index, key, complaint)`` for each ``/bad`` the user typed while the agent
    was busy. Claude Code delivers those mid-turn without running
    UserPromptSubmit hooks, but logs an ``enqueue`` record at the moment of typing,
    so the flagged message is the last one before that record."""
    out = []
    for i, r in enumerate(recs):
        if r.get("type") != "queue-operation" or r.get("operation") != "enqueue":
            continue
        parsed = parse(r.get("content") or "")
        if parsed and parsed[0] == "bad" and parsed[1]:
            out.append((i, f"{r.get('timestamp')}|{r.get('content')}", parsed[1]))
    return out


def handle_stop(payload: dict) -> str:
    """Stop hook: save any ``/bad`` that arrived mid-turn and wasn't captured."""
    if os.environ.get(REPLAY_ENV) or detect_harness(payload) != "claude-code":
        return ""
    transcript = Path(payload.get("transcript_path") or "")
    if not transcript.is_file():
        return ""
    flags = queued_flags(claude_code.read_jsonl(transcript))
    done = {c.meta.get("queuedKey") for c in all_cases()}
    messages = []
    for idx, key, complaint in flags:
        if key in done:
            continue
        try:
            messages.append(capture(payload, complaint, upto=idx, extra={"queuedKey": key}))
        except Exception as e:  # noqa: BLE001 — a Stop hook must never break the session
            messages.append(f"slash-bad: couldn't save a queued /bad ({type(e).__name__}: {e})")
    return "\n".join(messages)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if payload.get("hook_event_name") == "Stop":
        if msg := handle_stop(payload):
            print(
                json.dumps(
                    {
                        "systemMessage": msg
                        + "\n  (you typed /bad while the agent was busy, so it saw the line too)"
                    }
                )
            )
        return 0
    code, msg = handle(payload)
    if code == 0 and msg:
        # Saved but not blocked: show the user, and tell the model to ignore the line.
        print(
            json.dumps(
                {
                    "systemMessage": msg,
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": BUSY_NOTE,
                    },
                }
            )
        )
        return 0
    if msg:
        print(msg, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
