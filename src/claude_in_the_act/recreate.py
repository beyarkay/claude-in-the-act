"""Replay a stored transcript and regenerate its final assistant turn.

Runs an independent headless Claude Code session (``claude --resume
--fork-session``) to reproduce the last assistant response, so it can be compared
against the original (bad) one — e.g. to see whether a newer model does better.

Self-contained: it strips the trailing ``bad_annotation``, drops the final
assistant turn (the bad one) and the final user turn, restores the remaining
context, and re-issues the final user message as the prompt. The replay runs
from an isolated temp dir, so it never depends on the original working tree
still existing. The harness rebuilds the system prompt + tool schemas fresh at
launch, so the replay tracks the *current* harness/model.

Tools are disabled by default (pure-text regen, no side effects). Set
``ALLOW_TOOLS=1`` to keep tools on — but then the model's calls actually execute
and read the *current* filesystem, so only do that for a reconstructed tree.

``split_transcript()`` (pure) is unit-tested; ``recreate()`` adds the claude call.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def _slug(path: str) -> str:
    """Mirror Claude Code's cwd → project-dir slug (non-alphanumeric → '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _assistant_text(rec: dict) -> str | None:
    if rec.get("type") != "assistant":
        return None
    content = rec.get("message", {}).get("content")
    if not isinstance(content, list):
        return None
    return "\n".join(
        b.get("text", "") for b in content if b.get("type") == "text"
    ).strip()


def _user_text(rec: dict) -> str | None:
    if rec.get("type") != "user":
        return None
    content = rec.get("message", {}).get("content")
    return content if isinstance(content, str) else None


def split_transcript(recs: list[dict]) -> dict:
    """Split records into the replay inputs.

    Drops any ``bad_annotation``, finds the last string user turn (the prompt to
    replay), the original assistant response that followed it, the context
    before it, and the model id. Raises ValueError if there is no user turn.
    """
    recs = [r for r in recs if r.get("type") != "bad_annotation"]
    user_idxs = [i for i, r in enumerate(recs) if _user_text(r) is not None]
    if not user_idxs:
        raise ValueError("no string user turn found to replay")
    last_user = user_idxs[-1]

    # The original response is ALL assistant text after the prompt (a turn can
    # span several records: thinking-only / tool_use-only ones have no text and
    # must be skipped, not treated as the answer).
    original = "\n\n".join(
        t for r in recs[last_user + 1 :] if (t := _assistant_text(r))
    ).strip()
    model = next(
        (
            r["message"]["model"]
            for r in reversed(recs)
            if r.get("type") == "assistant" and r.get("message", {}).get("model")
        ),
        "",
    )
    return {
        "prompt": _user_text(recs[last_user]) or "",
        "original": original,
        "context": recs[:last_user],
        "model": model,
    }


def recreate(stored: Path, model_override: str | None = None) -> dict[str, str]:
    recs = [
        json.loads(line) for line in stored.read_text().splitlines() if line.strip()
    ]
    split = split_transcript(recs)
    prompt, original = split["prompt"], split["original"]
    model = model_override or split["model"]

    # Canonicalize the temp dir (macOS /var → /private/var, /tmp → /private/tmp)
    # so our slug matches the one Claude Code derives from the real cwd —
    # otherwise --resume looks in the wrong project dir and never finds the
    # planted session.
    replay_dir = Path(tempfile.mkdtemp()).resolve()
    proj = Path.home() / ".claude" / "projects" / _slug(str(replay_dir))
    proj.mkdir(parents=True, exist_ok=True)
    new_id = str(uuid.uuid4())
    planted = proj / f"{new_id}.jsonl"
    planted.write_text("".join(json.dumps(r) + "\n" for r in split["context"]))

    cmd = [
        "claude",
        "--resume",
        new_id,
        "--fork-session",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
    ]
    if os.environ.get("ALLOW_TOOLS", "0") != "1":
        cmd += ["--allowed-tools", ""]

    try:
        proc = subprocess.run(cmd, cwd=replay_dir, capture_output=True, text=True)
        recreated = ""
        if proc.stdout.strip():
            recreated = json.loads(proc.stdout).get("result", "")
    finally:
        planted.unlink(missing_ok=True)
        try:
            proj.rmdir()
        except OSError:
            pass
        shutil.rmtree(replay_dir, ignore_errors=True)

    return {
        "original": original,
        "recreated": recreated,
        "prompt": prompt,
        "model": model,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cita-recreate <stored.jsonl> [model]", file=sys.stderr)
        return 1
    stored = Path(sys.argv[1])
    model_override = sys.argv[2] if len(sys.argv) > 2 else None
    if not stored.is_file():
        print(f"no such file: {stored}", file=sys.stderr)
        return 1
    print(json.dumps(recreate(stored, model_override), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
