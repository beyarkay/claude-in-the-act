"""On-disk case store.

A case is one directory under ``CASES_DIR``::

    <id>/case.json       metadata: complaint, flagged response, model, harness
    <id>/context.jsonl   the native transcript up to (not including) the
                         flagged message — exactly what a replay resumes from
    <id>/results.jsonl   append-only log of replays and judge verdicts

Nothing about the repository is copied: the replay only ever re-samples the
flagged message from the stored context, so the context is all we need.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from slash_bad import CASES_DIR


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_case_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def read_jsonl(path: Path) -> list[dict]:
    """Records of a jsonl file, skipping blank and malformed lines."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_jsonl(path: Path, recs: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r) + "\n" for r in recs))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    Path(tmp).replace(path)


class Case:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.name
        self.meta: dict = json.loads((path / "case.json").read_text())

    @classmethod
    def create(
        cls, meta: dict, context: list[dict], root: Path | None = None
    ) -> Case:
        root = root or CASES_DIR
        case_id = meta.get("id") or new_case_id()
        path = root / case_id
        path.mkdir(parents=True, exist_ok=False)
        write_jsonl(path / "context.jsonl", context)
        _atomic_write(path / "case.json", json.dumps({**meta, "id": case_id}, indent=2))
        return cls(path)

    @property
    def context(self) -> list[dict]:
        return read_jsonl(self.path / "context.jsonl")

    @property
    def complaint(self) -> str:
        return self.meta.get("complaint", "")

    @property
    def harness(self) -> str:
        return self.meta.get("harness", "claude-code")

    def save_meta(self) -> None:
        _atomic_write(self.path / "case.json", json.dumps(self.meta, indent=2))

    def results(self) -> list[dict]:
        return read_jsonl(self.path / "results.jsonl")

    def add_result(self, rec: dict) -> dict:
        rec = {"at": now_iso(), **rec}
        with (self.path / "results.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


def all_cases(root: Path | None = None) -> list[Case]:
    root = root or CASES_DIR
    if not root.exists():
        return []
    return [Case(p) for p in sorted(root.iterdir()) if (p / "case.json").exists()]


def find_case(ref: str, root: Path | None = None) -> Case:
    """A case by exact id, unique id prefix, or 1-based index from `list`."""
    cases = all_cases(root)
    if ref.isdigit() and 0 < int(ref) <= len(cases):
        return cases[int(ref) - 1]
    matches = [c for c in cases if c.id.startswith(ref)]
    if len(matches) != 1:
        raise KeyError(f"no unique case matches {ref!r} ({len(matches)} matches)")
    return matches[0]
