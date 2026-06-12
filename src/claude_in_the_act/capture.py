"""Capture a flagged (``BAD:``) Claude Code response into the dataset.

Reads a Claude Code ``UserPromptSubmit`` hook payload (JSON) on stdin. If the
prompt starts with ``BAD: ``, it snapshots the session transcript verbatim,
captures best-effort git state (including a patch of tracked + untracked
changes), appends a ``bad_annotation`` record, and exits 2 to block the prompt.
Otherwise it exits 0 (passthrough).

The transcript holds the conversation (prompts, assistant turns incl. tool calls,
tool results) but NOT the system prompt or tool schemas — those are rebuilt by
the harness at replay time. The git capture lets a file/code-dependent context
be reconstructed without assuming the original working tree still exists.

Invoked by ``flag-bad-response.sh`` only after a prefix check, so the common
no-op path never boots Python. ``process()`` holds the logic and is unit-tested;
``main()`` only wires up stdin + the ``CITA_DATA_DIR`` env override.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from claude_in_the_act.transcript import GOOD_RESPONSE_PLACEHOLDER, final_response

PREFIX = "BAD: "
DEFAULT_DATA_DIR = Path.home() / "projects" / "claude-in-the-act" / "data"


def _git(cwd: str, *args: str) -> str | None:
    """Run a git command in ``cwd``; return stripped stdout, or None on failure."""
    try:
        proc = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def capture_git(cwd: str, patch_path: Path) -> dict[str, str | None]:
    """Best-effort git state for ``cwd``.

    The patch captures the full working-tree delta vs HEAD *including untracked*
    files (``.gitignore``d ones excluded), built in a throwaway index via
    ``GIT_INDEX_FILE`` so the user's real staging area is never touched.
    """
    info: dict[str, str | None] = {
        "sha": None,
        "branch": None,
        "remote": None,
        "patchFile": None,
    }
    if not cwd or not os.path.isdir(cwd):
        return info
    if _git(cwd, "rev-parse", "--git-dir") is None:
        return info

    info["sha"] = _git(cwd, "rev-parse", "HEAD")
    info["branch"] = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    info["remote"] = _git(cwd, "remote", "get-url", "origin") or _git(
        cwd, "config", "--get", "remote.origin.url"
    )
    if not info["sha"]:
        return info  # no commits yet → nothing to diff against

    fd, tmp_index = tempfile.mkstemp(suffix=".gitindex")
    os.close(fd)
    os.unlink(tmp_index)  # let git create a fresh index at this path
    env = {**os.environ, "GIT_INDEX_FILE": tmp_index}
    try:
        staged = (
            subprocess.run(
                ["git", "-C", cwd, "read-tree", "HEAD"], env=env, capture_output=True
            ).returncode
            == 0
            and subprocess.run(
                ["git", "-C", cwd, "add", "-A"], env=env, capture_output=True
            ).returncode
            == 0
        )
        if staged:
            diff = subprocess.run(
                ["git", "-C", cwd, "diff", "--cached", "--binary", "HEAD"],
                env=env,
                capture_output=True,
                text=True,
            )
            if diff.stdout:
                patch_path.write_text(diff.stdout)
                info["patchFile"] = patch_path.name
    finally:
        Path(tmp_index).unlink(missing_ok=True)
    return info


def process(payload: dict, data_dir: Path) -> tuple[int, str | None]:
    """Core capture. Returns ``(exit_code, stderr_message)``.

    exit 0 = passthrough (not a flag); exit 2 = block the prompt (captured, or
    failed to locate the transcript). ``stderr_message`` is None on passthrough.
    """
    prompt = payload.get("prompt", "")
    if not prompt.startswith(PREFIX):
        return 0, None

    transcript = payload.get("transcript_path", "")
    session = payload.get("session_id", "unknown")
    cwd = payload.get("cwd", "")
    critique = prompt[len(PREFIX) :]

    if not transcript or not Path(transcript).is_file():
        return 2, f"BAD: could not locate transcript ('{transcript}') — nothing saved."

    data_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = data_dir / f"{ts}-{session}"
    out = base.with_suffix(".jsonl")
    shutil.copyfile(transcript, out)  # verbatim snapshot

    git_info = capture_git(cwd, base.with_suffix(".patch"))

    # The flagged response is a machine-readable triple in the annotation: the
    # bad response (extracted), the user's feedback, and a goodResponse slot to
    # be filled in later (the eval's gold target — left null at capture time).
    transcript_recs = [
        json.loads(line) for line in out.read_text().splitlines() if line.strip()
    ]
    annotation = {
        "type": "bad_annotation",
        "feedback": critique,
        "badResponse": final_response(transcript_recs),
        "goodResponse": GOOD_RESPONSE_PLACEHOLDER,
        "sessionId": session,
        "cwd": cwd,
        "capturedAt": ts,
        "git": git_info,
    }
    with out.open("a") as f:
        f.write(json.dumps(annotation) + "\n")

    records = sum(1 for _ in out.open())
    patch_note = " +patch" if git_info["patchFile"] else ""
    feedback = " ".join(critique.split())
    if len(feedback) > 70:
        feedback = feedback[:67] + "..."
    project = DEFAULT_DATA_DIR.parent
    return 2, (
        f'✎ BAD response captured — feedback: "{feedback}"\n'
        f"  bundle:  {out.resolve()} ({records} records{patch_note})\n"
        f"  (fill the goodResponse field in that jsonl with your ideal answer)\n"
        f"  judge:   uv run --project {project} cita-judge {out.resolve()} [model]"
    )


def main() -> int:
    payload = json.load(sys.stdin)
    data_dir = Path(os.environ.get("CITA_DATA_DIR", str(DEFAULT_DATA_DIR)))
    code, msg = process(payload, data_dir)
    if msg:
        print(msg, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
