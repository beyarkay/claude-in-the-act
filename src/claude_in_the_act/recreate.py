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

from claude_in_the_act.transcript import denumber, split_transcript, user_text


def _slug(path: str) -> str:
    """Mirror Claude Code's cwd → project-dir slug (non-alphanumeric → '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


_MUTATOR_TOKENS = (
    "rm ",
    "rmdir",
    "mv ",
    "cp ",
    "sed -i",
    "perl -i",
    "tee ",
    "touch ",
    "mkdir",
    "dd ",
    "truncate",
    "ln ",
    "chmod",
    "chown",
    "git checkout",
    "git reset",
    "git clean",
    "git stash",
    "git apply",
    "git restore",
    "git rm",
    "git mv",
    "install",
    "cargo build",
    "make ",
    "go build",
)


def _mutating_bash(cmd: str) -> bool:
    """Heuristic: could this shell command have changed the file tree? Over-flags
    (e.g. `> /dev/null`) on purpose — the flag is informational, not a gate."""
    if re.search(r"(?<![0-9&])>>?(?!&)", cmd):  # redirection to a file
        return True
    return any(tok in cmd for tok in _MUTATOR_TOKENS)


def _resolve(repo: Path, orig_cwd: str | None, file_path: str) -> Path:
    """Map a recorded file path into the reconstructed repo."""
    p = Path(file_path)
    if orig_cwd:
        try:
            return repo / p.relative_to(orig_cwd)
        except ValueError:
            pass
    return repo / (p.name if p.is_absolute() else p)


def _relpath(orig_cwd: str | None, file_path: str) -> str | None:
    if not orig_cwd:
        return None
    try:
        return str(Path(file_path).relative_to(orig_cwd))
    except ValueError:
        return None


def undo_last_turn(recs: list[dict], repo: Path) -> dict:
    """Reverse the final assistant turn's Write/Edit on the reconstructed
    (post-turn) tree to approximate the PRE-turn state — so the replayed model
    faces the situation it originally did, not the aftermath of its own actions.

    Flags Bash commands that could have mutated the tree (those can't be reversed
    from the transcript). Returns {preTurnExact, undone, bashMutations}.
    """
    recs_na = [r for r in recs if r.get("type") != "bad_annotation"]
    orig_cwd = next((r.get("cwd") for r in reversed(recs_na) if r.get("cwd")), None)
    user_idxs = [i for i, r in enumerate(recs_na) if user_text(r) is not None]
    if not user_idxs:
        return {"preTurnExact": True, "undone": [], "bashMutations": []}
    last_user = user_idxs[-1]

    # Latest observed content of each file BEFORE the last turn (Read result /
    # Write content) — used to restore files the last turn overwrote.
    prior: dict[str, str] = {}
    read_files: dict[str, str] = {}
    for r in recs_na[: last_user + 1]:
        content = (r.get("message") or {}).get("content")
        if r.get("type") == "assistant" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_use" and b.get("name") == "Read":
                    read_files[b["id"]] = b["input"].get("file_path")
                elif b.get("type") == "tool_use" and b.get("name") == "Write":
                    prior[b["input"].get("file_path")] = b["input"].get("content", "")
        if r.get("type") == "user" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_result":
                    fp = read_files.get(b.get("tool_use_id"))
                    rc = b.get("content")
                    if fp and isinstance(rc, str):
                        prior[fp] = denumber(rc)

    calls: list[tuple[str, dict]] = []
    for r in recs_na[last_user + 1 :]:
        if r.get("type") == "assistant":
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls.append((b["name"], b["input"]))

    undone: list[str] = []
    bash_mutations: list[str] = []
    for name, inp in reversed(calls):  # undo in reverse order
        if name == "Edit":
            target = _resolve(repo, orig_cwd, inp.get("file_path", ""))
            old, new = inp.get("old_string", ""), inp.get("new_string", "")
            if target.exists() and new:
                txt = target.read_text()
                txt = (
                    txt.replace(new, old)
                    if inp.get("replace_all")
                    else txt.replace(new, old, 1)
                )
                target.write_text(txt)
                undone.append(f"Edit {inp.get('file_path')}")
        elif name == "Write":
            fp = inp.get("file_path", "")
            target = _resolve(repo, orig_cwd, fp)
            rel = _relpath(orig_cwd, fp)
            if fp in prior:  # overwrote a file we'd seen — restore prior content
                target.write_text(prior[fp])
                undone.append(f"Write(restore) {fp}")
            elif (
                rel
                and _git("-C", str(repo), "cat-file", "-e", f"HEAD:{rel}").returncode
                == 0
            ):
                show = _git("-C", str(repo), "show", f"HEAD:{rel}")
                if show.returncode == 0:
                    target.write_text(show.stdout)
                    undone.append(f"Write(HEAD) {fp}")
            else:  # new file this turn — delete it
                if target.exists():
                    target.unlink()
                undone.append(f"Write(delete) {fp}")
        elif name == "Bash" and _mutating_bash(inp.get("command", "")):
            bash_mutations.append(inp.get("command", "")[:160])

    return {
        "preTurnExact": not bash_mutations,
        "undone": undone,
        "bashMutations": bash_mutations,
    }


def reconstruct(annotation: dict, patch_path: Path | None, dest: Path) -> str:
    """Rebuild the working tree at capture time into ``dest``.

    Fast path: a detached ``git worktree`` off the original local repo (has all
    commits incl. unpushed, real config, no network). Fallback: clone the remote
    (default) or the local path. Then ``git apply`` the patch (tracked +
    untracked). Returns a source tag (``worktree:<repo>`` or the clone URL);
    raises if the SHA is unreachable. Worktree sources must be released by the
    caller with ``release_worktree``.
    """
    git = annotation.get("git") or {}
    sha = git.get("sha")
    if not sha:
        raise RuntimeError("capture has no git SHA — cannot reconstruct")
    cwd = annotation.get("cwd")
    source = None

    # Fast path: a worktree off the local repo, if it still has the SHA.
    if (
        cwd
        and Path(cwd).is_dir()
        and _git("-C", cwd, "rev-parse", "--git-dir").returncode == 0
        and _git("-C", cwd, "cat-file", "-e", sha).returncode == 0
        and _git("-C", cwd, "worktree", "add", "--detach", str(dest), sha).returncode
        == 0
    ):
        source = f"worktree:{cwd}"

    # Fallback: clone the remote (default) or the local path.
    if source is None:
        for src in [s for s in (git.get("remote"), cwd) if s]:
            shutil.rmtree(dest, ignore_errors=True)
            if _git("clone", "--quiet", src, str(dest)).returncode != 0:
                continue
            if (
                _git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
                == 0
            ):
                source = src
                break
            _git(
                "-C", str(dest), "fetch", "--quiet", src, sha
            )  # SHA not in default clone
            if (
                _git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
                == 0
            ):
                source = src
                break
    if source is None:
        raise RuntimeError(f"could not reconstruct {sha} (worktree + clone failed)")

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
    return source


def release_worktree(source: str, dest: Path) -> None:
    """Remove a worktree created by ``reconstruct`` (no-op for clone sources)."""
    if source.startswith("worktree:"):
        orig = source[len("worktree:") :]
        _git("-C", orig, "worktree", "remove", "--force", str(dest))
        _git("-C", orig, "worktree", "prune")


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

    # The patch is post-turn, so the reconstruction includes the bad turn's own
    # Write/Edit. Reverse them so the model faces the PRE-turn situation.
    undo = undo_last_turn(recs, repo)

    # Path-rewrite: the resumed context references the ORIGINAL absolute cwd, but
    # the reconstruction lives elsewhere. Repoint those paths at the worktree so
    # the model operates there, not (via leaked absolute paths) on the real repo.
    orig_cwd = annotation.get("cwd")

    def _remap(text: str) -> str:
        return text.replace(orig_cwd, str(repo)) if orig_cwd else text

    # Single-turn captures have no prior messages; `--resume` rejects an empty
    # session, so run a fresh `claude -p` in that case.
    has_convo = any(r.get("type") in ("user", "assistant") for r in split["context"])
    base = [
        "-p",
        _remap(prompt),
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
        ctx = "".join(json.dumps(r) + "\n" for r in split["context"])
        planted.write_text(_remap(ctx))
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
        "preTurnExact": undo["preTurnExact"],
        "undone": undo["undone"],
        "bashMutations": undo["bashMutations"],
    }
    # Always release a worktree (it lives in the real repo's .git); keep the
    # /tmp dir on failure only for clone sources (for debugging).
    if source.startswith("worktree:"):
        release_worktree(source, repo)
        shutil.rmtree(replay_root, ignore_errors=True)
    elif recreated:
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
    # Reconstruction (real repo + real tools) when the capture has git state;
    # otherwise the tool-free regeneration.
    recs = [
        json.loads(line) for line in stored.read_text().splitlines() if line.strip()
    ]
    has_git = any(
        (r.get("git") or {}).get("sha")
        for r in recs
        if r.get("type") == "bad_annotation"
    )
    fn = recreate_with_tools if has_git else recreate
    print(json.dumps(fn(stored, model_override), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
