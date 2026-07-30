"""Pin every capture's commit so git can never garbage-collect it.

A capture records a SHA and expects to reconstruct from it later. That works
fine while the branch is alive — but captures taken inside a ``git worktree``
(Claude Code puts them under ``<repo>/.claude/worktrees/<name>``) routinely
outlive both the worktree directory and the branch. Once the branch is deleted
the commit becomes unreachable, and unreachable objects are pruned by ``git gc``
(default: two weeks). The transcript survives; the code it refers to quietly
does not.

The fix is one ref per capture, not a retained worktree. ``refs/cita/<capture>``
in whichever repo holds the object makes it reachable forever, costs nothing (no
working tree, no checkout), and stays out of ``refs/heads`` so it never shows up
in ``git branch`` or gets pushed by accident.

``cita-pin`` pins every bundle in ``DATA_DIR``; ``cita-pin --list`` shows what is
pinned and what is at risk; ``cita-pin --unpin`` removes them again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from claude_in_the_act import DATA_DIR
from claude_in_the_act.harness import git, git_out
from claude_in_the_act.recreate import local_repo_with_sha
from claude_in_the_act.transcript import annotation, load_bundle

REF_PREFIX = "refs/cita"


def _tail_annotation(bundle: Path) -> dict:
    """The bundle's bad_annotation, read from the file tail — these transcripts
    run to tens of megabytes and the annotation is always the last record."""
    size = bundle.stat().st_size
    with bundle.open("rb") as f:
        f.seek(max(0, size - 400_000))
        tail = f.read().decode("utf-8", "replace")
    # Rebuild just the tail into records; the annotation is the last one.
    lines = tail.splitlines()
    if size > 400_000 and lines:
        lines = lines[1:]  # first line is probably truncated mid-record
    return annotation(load_bundle_lines(lines))


def load_bundle_lines(lines: list[str]) -> list[dict]:
    import json

    recs = []
    for line in lines:
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def capture_ref(bundle: Path) -> str:
    """``refs/cita/<capture-stem>`` — one ref per capture, named after the file."""
    return f"{REF_PREFIX}/{bundle.stem}"


def is_reachable(repo: str, sha: str) -> bool:
    """Is ``sha`` reachable from any ref (i.e. safe from ``git gc``)?"""
    out = git_out("-C", repo, "for-each-ref", "--contains", sha, "--count=1")
    return bool(out)


def pin_bundle(bundle: Path, *, unpin: bool = False) -> dict:
    """Create (or delete) the ref pinning one capture's commit.

    Returns a status dict; never raises, so one odd bundle can't sink a sweep.
    """
    ann = _tail_annotation(bundle)
    sha = (ann.get("git") or {}).get("sha") or ""
    cwd = ann.get("cwd") or ""
    row = {"bundle": bundle.name, "sha": sha[:10], "repo": "", "status": ""}
    if not sha:
        row["status"] = "no-sha"
        return row

    repo = local_repo_with_sha(cwd, sha)
    if not repo:
        row["status"] = "sha-unreachable"  # nothing local holds this object
        return row
    row["repo"] = repo
    ref = capture_ref(bundle)

    if unpin:
        rc = git("-C", repo, "update-ref", "-d", ref).returncode
        row["status"] = "unpinned" if rc == 0 else "unpin-failed"
        return row

    if git_out("-C", repo, "rev-parse", "--verify", "--quiet", ref) == sha:
        row["status"] = "already-pinned"
        return row
    rc = git("-C", repo, "update-ref", ref, sha).returncode
    row["status"] = "pinned" if rc == 0 else "pin-failed"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list",
        action="store_true",
        help="report each capture's commit status without changing anything",
    )
    ap.add_argument(
        "--unpin", action="store_true", help="remove the refs this command created"
    )
    args = ap.parse_args()

    bundles = sorted(DATA_DIR.glob("*.jsonl"))
    if not bundles:
        print(f"no bundles in {DATA_DIR}", file=sys.stderr)
        return 1

    rows = []
    for b in bundles:
        if args.list:
            ann = _tail_annotation(b)
            sha = (ann.get("git") or {}).get("sha") or ""
            cwd = ann.get("cwd") or ""
            repo = local_repo_with_sha(cwd, sha) if sha else None
            if not sha:
                status = "no-sha"
            elif not repo:
                status = "LOST (no local repo has this commit)"
            elif is_reachable(repo, sha):
                status = "safe (reachable from a ref)"
            else:
                status = "AT RISK (unreachable — gc will prune)"
            rows.append(
                {
                    "bundle": b.name,
                    "sha": sha[:10],
                    "repo": repo or "",
                    "status": status,
                }
            )
        else:
            rows.append(pin_bundle(b, unpin=args.unpin))

    width = max(len(r["bundle"]) for r in rows)
    for r in rows:
        repo = Path(r["repo"]).name if r["repo"] else "-"
        print(f"{r['bundle']:<{width}}  {r['sha']:<10}  {repo:<22}  {r['status']}")

    counts: dict[str, int] = {}
    for r in rows:
        key = r["status"].split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
