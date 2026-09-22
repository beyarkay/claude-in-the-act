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
        lines.append(f"{i:>3}. {c.meta.get('model', '?'):<24} {complaint[:90]}")
    return "\n".join(lines)


def capture(payload: dict, complaint: str) -> str:
    harness = detect_harness(payload)
    mod = codex if harness == "codex" else claude_code
    transcript = Path(payload.get("transcript_path") or "")
    if not transcript.is_file():
        return f"slash-bad: couldn't find the session transcript ({transcript}); nothing saved."
    meta, context = mod.build_case(
        transcript, complaint, cwd=payload.get("cwd", ""), session_id=payload.get("session_id", "")
    )
    case = Case.create({**meta, "source": str(transcript)}, context)
    n = len(all_cases())
    flagged = " ".join(meta["flagged"].split())
    return (
        f"✎ slash-bad: saved case {case.id} ({n} total)\n"
        f"  model:     {meta['model'] or 'unknown'}\n"
        f'  flagged:   "{flagged[:80]}{"…" if len(flagged) > 80 else ""}"\n'
        f"  complaint: {' '.join(complaint.split())[:100]}\n"
        f"  Next: /bad-compare latest to see whether a newer model still does this."
    )


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
        return 2, capture(payload, rest)
    except Exception as e:  # noqa: BLE001 — never let a capture bug leak the prompt through
        return 2, f"slash-bad: capture failed ({type(e).__name__}: {e}); nothing saved."


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    code, msg = handle(payload)
    if msg:
        print(msg, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
