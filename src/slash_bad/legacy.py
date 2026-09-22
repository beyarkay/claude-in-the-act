"""Import captures made by the old ``claude-in-the-act`` plugin (a full session
transcript with a ``bad_annotation`` record appended)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from slash_bad import claude_code
from slash_bad.store import Case, all_cases, read_jsonl, write_jsonl

_APPENDED = ("bad_annotation", "judge_result")


def main(args) -> int:
    src = Path(args.dir).expanduser()
    done = {c.meta.get("legacyBundle") for c in all_cases()}
    bundles = sorted(src.glob("*.jsonl"))
    imported = skipped = failed = 0
    for i, bundle in enumerate(bundles, 1):
        print(f"\r[{i}/{len(bundles)}] {bundle.name[:60]}\x1b[K", end="", file=sys.stderr)
        if bundle.name in done:
            skipped += 1
            continue
        recs = read_jsonl(bundle)
        ann = next((r for r in reversed(recs) if r.get("type") == "bad_annotation"), None)
        if not ann:
            skipped += 1
            continue
        body = [r for r in recs if r.get("type") not in _APPENDED]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            write_jsonl(path, body)
            try:
                meta, context = claude_code.build_case(
                    path,
                    ann.get("feedback", ""),
                    cwd=ann.get("cwd", ""),
                    session_id=ann.get("sessionId", ""),
                )
            except ValueError as e:
                failed += 1
                print(f"\n  skipped {bundle.name}: {e}", file=sys.stderr)
                continue
        meta["legacyBundle"] = bundle.name
        stamp = ann.get("capturedAt") or ""
        if len(stamp) == 15:  # 20260612T164400
            meta["id"] = f"{stamp[:8]}-{stamp[9:]}-{bundle.stem[-6:]}"
        Case.create(meta, context)
        imported += 1
    print(file=sys.stderr)
    print(f"imported {imported}, skipped {skipped}, failed {failed}")
    return 0
