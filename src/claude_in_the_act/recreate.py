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

from claude_in_the_act.replay import build_recording
from claude_in_the_act.transcript import split_transcript


def _slug(path: str) -> str:
    """Mirror Claude Code's cwd → project-dir slug (non-alphanumeric → '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _cita_root() -> Path:
    return Path(__file__).resolve().parents[2]


def recreate_with_tools(stored: Path, model_override: str | None = None) -> dict:
    """Replay a tool-using turn with tools ON but every tool call served from the
    RECORDING (no codebase, no execution).

    Sets up a temp replay dir with `.claude/settings.json` PreToolUse hooks that
    intercept Read/Bash/Write/Edit and feed back the recorded results (see
    `claude_in_the_act.replay`). The model re-runs against the recording; calls
    that don't match a recorded one are denied and logged as divergences.
    """
    recs = [
        json.loads(line) for line in stored.read_text().splitlines() if line.strip()
    ]
    split = split_transcript(recs)
    recording = build_recording(recs)
    prompt, original = split["prompt"], split["original"]
    model = model_override or split["model"]

    replay_dir = Path(tempfile.mkdtemp()).resolve()
    stash = replay_dir / "_stash"
    stash.mkdir()
    (replay_dir / "recording.json").write_text(json.dumps(recording))

    # Seed recorded files into the sandbox at cwd-relative paths. Edit (unlike
    # Read) pre-checks that the file EXISTS before the hook runs, so the file
    # must be present for the hook to then redirect it. Content still comes from
    # the recording (no real codebase needed).
    orig_cwd = next((r.get("cwd") for r in reversed(recs) if r.get("cwd")), None)
    if orig_cwd:
        from claude_in_the_act.replay import denumber

        for entry in recording:
            if entry["tool_name"] != "Read":
                continue
            try:
                rel = Path(entry["input"]["file_path"]).relative_to(orig_cwd)
            except (ValueError, KeyError):
                continue
            dest = replay_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_text(denumber(entry["content"]))
            except OSError:
                pass

    venv_py = _cita_root() / ".venv" / "bin" / "python"
    hook_cmd = (
        f"CITA_RECORDING={replay_dir}/recording.json CITA_STASH={stash} "
        f"{venv_py} -m claude_in_the_act.replay"
    )
    # Only Bash is hooked: Read/Edit/Write operate natively on the seeded files
    # (so the harness's read-before-edit tracking works). Bash output that can't
    # come from a file (git log, etc.) is served from the recording.
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": hook_cmd}],
                }
            ]
        }
    }
    claude_dir = replay_dir / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    # Restore prior conversation via --resume; but a single-turn capture has no
    # messages before the prompt, and `--resume` rejects an empty session — so
    # run a fresh `claude -p` in that case.
    has_convo = any(r.get("type") in ("user", "assistant") for r in split["context"])
    tool_args = ["--allowed-tools", "Read", "Bash", "Write", "Edit"]
    base = ["-p", prompt, "--output-format", "json", "--model", model, *tool_args]
    proj = Path.home() / ".claude" / "projects" / _slug(str(replay_dir))
    planted = None
    if has_convo:
        proj.mkdir(parents=True, exist_ok=True)
        new_id = str(uuid.uuid4())
        planted = proj / f"{new_id}.jsonl"
        planted.write_text("".join(json.dumps(r) + "\n" for r in split["context"]))
        cmd = ["claude", "--resume", new_id, "--fork-session", *base]
    else:
        cmd = ["claude", *base]

    proc = subprocess.run(cmd, cwd=replay_dir, capture_output=True, text=True)
    recreated = ""
    if proc.stdout.strip():
        try:
            recreated = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            recreated = ""
    if planted is not None:
        planted.unlink(missing_ok=True)
        try:
            proj.rmdir()
        except OSError:
            pass

    log_path = stash / "replay.log"
    replay_log = log_path.read_text() if log_path.exists() else ""
    result = {
        "original": original,
        "recreated": recreated,
        "prompt": prompt,
        "model": model,
        "replay_log": replay_log,
        "returncode": proc.returncode,
        "stderr": proc.stderr[-2000:],
        "stdout_head": proc.stdout[:500],
        "replay_dir": str(replay_dir),
    }
    if recreated and replay_log:
        shutil.rmtree(replay_dir, ignore_errors=True)  # only clean on success
    return result


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
