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

from claude_in_the_act.transcript import split_transcript


def _slug(path: str) -> str:
    """Mirror Claude Code's cwd → project-dir slug (non-alphanumeric → '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def reconstruct(annotation: dict, patch_path: Path | None, dest: Path) -> str:
    """Rebuild the working tree at capture time into ``dest``.

    Clone the remote (default) or the original local path (fallback), check out
    the captured SHA, and apply the patch (tracked + untracked). Returns the
    source cloned from. Raises if the SHA can't be reconstructed.
    """
    git = annotation.get("git") or {}
    sha = git.get("sha")
    if not sha:
        raise RuntimeError("capture has no git SHA — cannot reconstruct")
    sources = [s for s in (git.get("remote"), annotation.get("cwd")) if s]

    for src in sources:
        shutil.rmtree(dest, ignore_errors=True)
        if _git("clone", "--quiet", src, str(dest)).returncode != 0:
            continue
        if (
            _git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
            == 0
        ):
            break
        # SHA absent from the default clone — try fetching it explicitly.
        _git("-C", str(dest), "fetch", "--quiet", src, sha)
        if (
            _git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
            == 0
        ):
            break
    else:
        raise RuntimeError(f"could not reconstruct {sha} from {sources}")

    if patch_path and patch_path.exists():
        applied = _git("-C", str(dest), "apply", "--whitespace=nowarn", str(patch_path))
        if applied.returncode != 0:
            _git(
                "-C",
                str(dest),
                "apply",
                "--3way",
                "--whitespace=nowarn",
                str(patch_path),
            )
    return src


def recreate_with_tools(stored: Path, model_override: str | None = None) -> dict:
    """Replay a tool-using turn against a FULL reconstruction of the original
    repo, with real tools (no mocking).

    Rebuilds the captured tree (git clone + checkout SHA + git apply patch) into
    a /tmp dir, then runs ``claude --resume`` there — so the harness sees a real
    git repo and every tool runs for real against it. Requires git state in the
    capture (SHA + a reachable remote or local path).
    """
    recs = [
        json.loads(line) for line in stored.read_text().splitlines() if line.strip()
    ]
    split = split_transcript(recs)
    prompt, original = split["prompt"], split["original"]
    model = model_override or split["model"]
    annotation = next(
        (r for r in reversed(recs) if r.get("type") == "bad_annotation"), {}
    )
    patch_path = None
    pf = (annotation.get("git") or {}).get("patchFile")
    if pf:
        patch_path = stored.parent / pf

    replay_root = Path(tempfile.mkdtemp()).resolve()
    repo = replay_root / "repo"
    source = reconstruct(annotation, patch_path, repo)
    repo = repo.resolve()

    # Single-turn captures have no prior messages; `--resume` rejects an empty
    # session, so run a fresh `claude -p` in that case.
    has_convo = any(r.get("type") in ("user", "assistant") for r in split["context"])
    base = [
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--permission-mode",
        "acceptEdits",
        "--allowed-tools",
        "Read",
        "Edit",
        "Write",
        "Bash",
        "Grep",
        "Glob",
    ]
    proj = Path.home() / ".claude" / "projects" / _slug(str(repo))
    planted = None
    if has_convo:
        proj.mkdir(parents=True, exist_ok=True)
        new_id = str(uuid.uuid4())
        planted = proj / f"{new_id}.jsonl"
        planted.write_text("".join(json.dumps(r) + "\n" for r in split["context"]))
        cmd = ["claude", "--resume", new_id, "--fork-session", *base]
    else:
        cmd = ["claude", *base]

    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
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

    result = {
        "original": original,
        "recreated": recreated,
        "prompt": prompt,
        "model": model,
        "source": source,
        "returncode": proc.returncode,
        "stderr": proc.stderr[-2000:],
        "replay_dir": str(replay_root),
    }
    if recreated:
        shutil.rmtree(replay_root, ignore_errors=True)
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
