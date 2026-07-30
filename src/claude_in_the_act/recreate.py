"""Regenerate a captured bad response under the current harness/model.

Two paths, both driven by ``cita-judge``:

- ``recreate`` — tool-free turns: restore the pre-prompt context in an isolated
  temp session and re-issue the final user message with tools off. Self-contained.
- ``recreate_with_tools`` — tool-using turns: reconstruct the real repo (a
  detached ``git worktree`` off the local repo, fallback to a clone), undo the
  bad turn's own Write/Edit so the model meets the genuine pre-turn state, rewrite
  the resumed context's paths into the reconstruction, and run real tools there.

The pure pieces (``split_transcript``, ``undo_last_turn``, ``reconstruct``) are
unit-tested; the ``recreate*`` entry points add the ``claude`` call.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from claude_in_the_act.harness import (
    WORKTREE_LOCK,
    claude_meta,
    git,
    git_out,
    run_claude,
)
from claude_in_the_act.transcript import (
    ANNOTATION_TYPE,
    annotation as bundle_annotation,
    denumber,
    dump_jsonl,
    load_bundle,
    split_transcript,
    user_text,
)

# Tools the reconstructed (real-repo) replay is allowed to run.
REPLAY_TOOLS = ["Read", "Edit", "Write", "Bash", "Grep", "Glob"]


def _slug(path: str) -> str:
    """Mirror Claude Code's cwd → project-dir slug (non-alphanumeric → '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


_MUTATOR_TOKENS = (
    r"\brm\b",
    r"\brmdir\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\bsed\s+-i",
    r"\bperl\s+-i",
    r"\btee\b",
    r"\btouch\b",
    r"\bmkdir\b",
    r"\bdd\b",
    r"\btruncate\b",
    r"\bln\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bgit\b[^;|&]*\b(checkout|reset|clean|stash|apply|restore|rm|mv|add|commit)\b",
    r"\bgh\s+(pr|release)\s+(create|merge|edit)\b",
    r"\b(pip|uv pip|npm|cargo|go)\s+install\b",
    r"\bcargo\s+build\b",
    r"\bmake\b",
    r"\bgo\s+build\b",
    r"\bruff\s+format\b",  # rewrites the files it is pointed at
)
_MUTATOR_RE = re.compile("|".join(_MUTATOR_TOKENS))

# A redirect whose target is scratch space tells us nothing about the repo. The
# status-line hook alone (`printf … > /tmp/claude-task-<uuid>.txt`) fires on most
# recent captures, and `>/dev/null` fires on a large share of the rest.
_SCRATCH_TARGETS = ("/dev/null", "/tmp/", "/private/tmp/", "/var/folders/")

# `>` or `>>` as a redirect, not as part of `>=`, `->`, `=>`, `2>&1` or `>&2`.
_REDIRECT = re.compile(r"(?<![0-9&<>=!\-])>>?(?![&=>])\s*(?P<target>[^\s;|&)]*)")


def _mutating_bash(cmd: str) -> bool:
    """Could this shell command have changed the *repo's* file tree?

    Used to decide whether a replay faithfully restored the pre-turn state, so
    precision matters in both directions: a missed mutation silently corrupts a
    comparison, while a spurious one discards a usable transcript. Redirects into
    scratch space and comparison operators (``NR>=2``) are therefore not
    mutations, but anything writing to an unrecognised path is.
    """
    for m in _REDIRECT.finditer(cmd):
        if not m.group("target").startswith(_SCRATCH_TARGETS):
            return True
    return bool(_MUTATOR_RE.search(cmd))


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
    recs_na = [r for r in recs if r.get("type") != ANNOTATION_TYPE]
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
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                inp = b.get("input") or {}
                if b.get("name") == "Read":
                    read_files[b.get("id")] = inp.get("file_path")
                elif b.get("name") == "Write" and inp.get("file_path"):
                    prior[inp["file_path"]] = inp.get("content", "")
        if r.get("type") == "user" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    fp = read_files.get(b.get("tool_use_id"))
                    rc = b.get("content")
                    if fp and isinstance(rc, str):
                        prior[fp] = denumber(rc)

    calls: list[tuple[str, dict]] = []
    for r in recs_na[last_user + 1 :]:
        if r.get("type") == "assistant":
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls.append((b.get("name"), b.get("input") or {}))

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
                and git("-C", str(repo), "cat-file", "-e", f"HEAD:{rel}").returncode
                == 0
            ):
                show = git("-C", str(repo), "show", f"HEAD:{rel}")
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


def local_repo_with_sha(cwd: str | None, sha: str) -> str | None:
    """The nearest existing ancestor of ``cwd`` that is a git repo holding ``sha``.

    Captures taken inside a ``git worktree`` (Claude Code puts them under
    ``<repo>/.claude/worktrees/<name>``) record a cwd that gets deleted the moment
    the branch is done — but the commits still live in the parent repo's object
    store, so walking up recovers them without retaining the worktree. ``cwd``
    itself is tried first, so an intact repo takes the same path it always did.
    """
    if not cwd or not sha:
        return None
    for cand in [Path(cwd), *Path(cwd).parents]:
        if not cand.is_dir():
            continue
        # Normalise to the repo root: an intermediate directory (e.g. the
        # surviving `.claude/worktrees/` whose child was deleted) resolves to the
        # containing repo, and reporting *that* is what callers want to see.
        top = git_out("-C", str(cand), "rev-parse", "--show-toplevel")
        if not top:
            continue
        if git("-C", top, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0:
            return top
    return None


def reconstruct(annotation: dict, patch_path: Path | None, dest: Path) -> str:
    """Rebuild the working tree at capture time into ``dest``.

    Fast path: a detached ``git worktree`` off the original local repo (has all
    commits incl. unpushed, real config, no network). Fallback: clone the remote
    (default) or the local path. Then ``git apply`` the patch (tracked +
    untracked). Returns a source tag (``worktree:<repo>`` or the clone URL);
    raises if the SHA is unreachable. Worktree sources must be released by the
    caller with ``release_worktree``.
    """
    git_info = annotation.get("git") or {}
    sha = git_info.get("sha")
    if not sha:
        raise RuntimeError("capture has no git SHA — cannot reconstruct")
    cwd = annotation.get("cwd")
    source = None

    # Fast path: a detached worktree off the local repo, if it still has the SHA.
    # Falls back to the nearest ancestor repo, so captures made in a since-deleted
    # `.claude/worktrees/<name>` still reconstruct from the parent's objects.
    local = local_repo_with_sha(cwd, sha)
    if local:
        with WORKTREE_LOCK:  # `git worktree add` on one repo isn't concurrency-safe
            if (
                git(
                    "-C", local, "worktree", "add", "--detach", str(dest), sha
                ).returncode
                == 0
            ):
                source = f"worktree:{local}"

    # Fallback: clone the remote (default) or the local path.
    if source is None:
        for src in [s for s in (git_info.get("remote"), cwd) if s]:
            shutil.rmtree(dest, ignore_errors=True)
            if git("clone", "--quiet", src, str(dest)).returncode != 0:
                continue
            if (
                git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
                == 0
            ):
                source = src
                break
            git(
                "-C", str(dest), "fetch", "--quiet", src, sha
            )  # SHA not in default clone
            if (
                git("-C", str(dest), "checkout", "--quiet", "--detach", sha).returncode
                == 0
            ):
                source = src
                break
    if source is None:
        raise RuntimeError(f"could not reconstruct {sha} (worktree + clone failed)")

    if patch_path and patch_path.exists():
        applied = git("-C", str(dest), "apply", "--whitespace=nowarn", str(patch_path))
        if applied.returncode != 0:
            applied = git(
                "-C",
                str(dest),
                "apply",
                "--3way",
                "--whitespace=nowarn",
                str(patch_path),
            )
        if applied.returncode != 0:
            print(
                f"cita: warning: patch {patch_path.name} did not apply cleanly — "
                f"reconstruction may be missing uncommitted changes",
                file=sys.stderr,
            )
    return source


def release_worktree(source: str, dest: Path) -> None:
    """Remove a worktree created by ``reconstruct`` (no-op for clone sources)."""
    if source.startswith("worktree:"):
        orig = source[len("worktree:") :]
        with WORKTREE_LOCK:
            git("-C", orig, "worktree", "remove", "--force", str(dest))
            git("-C", orig, "worktree", "prune")


def recreate_with_tools(stored: Path, model_override: str | None = None) -> dict:
    """Replay a tool-using turn against a FULL reconstruction of the original repo,
    with real tools (no mocking). Requires git state (SHA + reachable repo)."""
    stored = Path(stored)
    recs = load_bundle(stored)
    split = split_transcript(recs)
    prompt, original = split["prompt"], split["original"]
    model = model_override or split["model"]
    ann = bundle_annotation(recs)
    pf = (ann.get("git") or {}).get("patchFile")
    patch_path = stored.parent / pf if pf else None

    replay_root = Path(tempfile.mkdtemp()).resolve()
    repo = replay_root / "repo"
    source = None
    proj = None
    planted = None
    try:
        source = reconstruct(ann, patch_path, repo)
        repo = repo.resolve()

        # The patch is post-turn, so the reconstruction includes the bad turn's
        # own Write/Edit. Reverse them so the model faces the PRE-turn situation.
        undo = undo_last_turn(recs, repo)

        # The resumed context references the ORIGINAL absolute cwd; repoint those
        # paths at the reconstruction so the model never touches the real repo.
        orig_cwd = ann.get("cwd")

        def remap(text: str) -> str:
            return text.replace(orig_cwd, str(repo)) if orig_cwd else text

        # Single-turn captures have no prior messages; `--resume` rejects an empty
        # session, so run a fresh `claude -p` in that case.
        has_convo = any(
            r.get("type") in ("user", "assistant") for r in split["context"]
        )
        resume_id = None
        if has_convo:
            proj = Path.home() / ".claude" / "projects" / _slug(str(repo))
            proj.mkdir(parents=True, exist_ok=True)
            resume_id = str(uuid.uuid4())
            planted = proj / f"{resume_id}.jsonl"
            planted.write_text(remap(dump_jsonl(split["context"])))

        recreated, proc = run_claude(
            remap(prompt),
            model=model,
            tools=REPLAY_TOOLS,
            cwd=repo,
            resume=resume_id,
            fork=bool(resume_id),
            permission_mode="acceptEdits",
        )
        return {
            "model": model,
            "prompt": prompt,
            "original": original,
            "recreated": recreated,
            "source": source,
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-2000:],
            "preTurnExact": undo["preTurnExact"],
            "undone": undo["undone"],
            "bashMutations": undo["bashMutations"],
            "cost": claude_meta(proc),
        }
    finally:
        if planted is not None:
            planted.unlink(missing_ok=True)
            if proj is not None:
                try:
                    proj.rmdir()
                except OSError:
                    pass
        if source:  # release the worktree from the real repo's .git
            release_worktree(source, repo)
        shutil.rmtree(replay_root, ignore_errors=True)


def recreate(stored: Path, model_override: str | None = None) -> dict[str, str]:
    """Replay a tool-free turn in an isolated temp session (tools off by default;
    ``ALLOW_TOOLS=1`` keeps them on, but then they run against an empty dir)."""
    recs = load_bundle(stored)
    split = split_transcript(recs)
    prompt, original = split["prompt"], split["original"]
    model = model_override or split["model"]

    # Canonicalize the temp dir (macOS /var → /private/var) so our slug matches
    # the one Claude Code derives from the real cwd, or --resume can't find it.
    replay_dir = Path(tempfile.mkdtemp()).resolve()
    proj = Path.home() / ".claude" / "projects" / _slug(str(replay_dir))
    proj.mkdir(parents=True, exist_ok=True)
    new_id = str(uuid.uuid4())
    planted = proj / f"{new_id}.jsonl"
    planted.write_text(dump_jsonl(split["context"]))

    tools = None if os.environ.get("ALLOW_TOOLS", "0") == "1" else []
    try:
        recreated, proc = run_claude(
            prompt, model=model, tools=tools, cwd=replay_dir, resume=new_id, fork=True
        )
    finally:
        planted.unlink(missing_ok=True)
        try:
            proj.rmdir()
        except OSError:
            pass
        shutil.rmtree(replay_dir, ignore_errors=True)

    return {
        "model": model,
        "prompt": prompt,
        "original": original,
        "recreated": recreated,
        "returncode": proc.returncode,
        "cost": claude_meta(proc),
    }
