"""Contribute anonymised cases to the shared Hugging Face dataset.

Each case becomes two files, named by a random id that stays stable for the case:

    data/<id>.jsonl                     one row: complaint, flagged response, the
                                        conversation as harness-neutral steps,
                                        and every verdict so far
    sessions/<harness>/<id>.jsonl       the redacted native context, enough for
                                        anyone to replay the case themselves

By default this opens a pull request (any Hugging Face account with a write
token can); ``--direct`` commits straight to the repository for its owner.
The upload uses the Hub's commit endpoint directly, so there's still nothing to
install.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from slash_bad import HOME, __version__
from slash_bad.compare import HARNESSES
from slash_bad.redact import Redactor, redact_case_dir
from slash_bad.store import Case, all_cases, find_case, now_iso

HUB = "https://huggingface.co"
MAX_FILE_BYTES = 8_000_000


def hf_token() -> str | None:
    if tok := os.environ.get("HF_TOKEN"):
        return tok.strip()
    for p in (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token",
        Path.home() / ".huggingface/token",
    ):
        if p.exists():
            return p.read_text().strip()
    return None


def share_id(case: Case) -> str:
    if not case.meta.get("shareId"):
        case.meta["shareId"] = uuid.uuid4().hex
        case.save_meta()
    return case.meta["shareId"]


def _verdicts(case: Case) -> list[dict]:
    replays = {
        r["replayId"]: r for r in case.results() if r.get("kind") == "replay" and r.get("replayId")
    }
    out = []
    for v in case.results():
        if v.get("kind") != "verdict" or v.get("score") is None:
            continue
        rep = replays.get(v.get("target"))
        out.append(
            {
                "response": "flagged" if v.get("target") == "original" else "replay",
                "responseModel": v.get("responseModel"),
                "responseText": rep.get("text") if rep else None,
                "score": v.get("score"),
                "rationale": v.get("rationale"),
                "judge": v.get("judgeModel"),
                "rubric": v.get("rubric"),
                "variant": v.get("variant"),
                "at": v.get("at"),
            }
        )
    return out


def build_files(cases: list[Case], redactor: Redactor) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for case in cases:
        sid = share_id(case)
        red = redact_case_dir(case.path, redactor)
        meta = red["meta"]
        steps = HARNESSES[case.harness].steps(red["context"])
        row = {
            "id": sid,
            "harness": case.harness,
            "harness_version": meta.get("harnessVersion"),
            "model": meta.get("model"),
            "complaint": meta.get("complaint"),
            "flagged_response": meta.get("flagged"),
            "conversation": steps,
            "verdicts": redactor.value(_verdicts(case)),
            "stats": meta.get("stats"),
            "slash_bad_version": __version__,
            "contributed_at": now_iso(),
        }
        files[f"data/{sid}.jsonl"] = (json.dumps(row) + "\n").encode()
        harness_dir = "claude_code" if case.harness == "claude-code" else "codex"
        files[f"sessions/{harness_dir}/{sid}.jsonl"] = "".join(
            json.dumps(r) + "\n" for r in red["context"]
        ).encode()
    return files


def commit(repo: str, files: dict[str, bytes], message: str, token: str, pr: bool) -> dict:
    lines = [
        {
            "key": "header",
            "value": {"summary": message, "description": "Contributed with slash-bad."},
        }
    ]
    for path, data in files.items():
        lines.append(
            {
                "key": "file",
                "value": {
                    "path": path,
                    "encoding": "base64",
                    "content": base64.b64encode(data).decode(),
                },
            }
        )
    url = f"{HUB}/api/datasets/{repo}/commit/main" + ("?create_pr=1" if pr else "")
    req = urllib.request.Request(
        url,
        data=b"".join(json.dumps(line).encode() + b"\n" for line in lines),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-ndjson"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Hugging Face said {e.code}: {e.read().decode()[:400]}") from e


def main(args) -> int:
    cases = [find_case(r) for r in args.cases] if args.cases else all_cases()
    if not cases:
        print("no cases to share", file=sys.stderr)
        return 1
    redactor = Redactor([s.strip() for s in args.redact.split(",") if s.strip()])
    files = build_files(cases, redactor)
    too_big = [p for p, d in files.items() if len(d) > MAX_FILE_BYTES]
    preview = HOME / "share-preview"
    for path, data in files.items():
        out = preview / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    total = sum(len(d) for d in files.values())
    print(f"{len(cases)} case(s) → {len(files)} files, {total / 1e6:.1f} MB")
    print(
        "redacted: "
        + (", ".join(f"{k} ×{v}" for k, v in sorted(redactor.counts.items())) or "nothing")
    )
    print(f"anonymised copies written to {preview}. Read them before sharing: code, file")
    print("contents and project names are not removed.")
    if too_big:
        print(
            f"too large to upload ({MAX_FILE_BYTES / 1e6:.0f} MB limit): {', '.join(too_big)}",
            file=sys.stderr,
        )
        files = {p: d for p, d in files.items() if p not in too_big}
    if args.dry_run:
        return 0
    token = hf_token()
    if not token:
        print("no Hugging Face token: run `hf auth login` or set HF_TOKEN", file=sys.stderr)
        return 1
    if not args.yes:
        answer = input(f"Upload these to huggingface.co/datasets/{args.repo}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("not uploaded")
            return 1
    result = commit(
        args.repo,
        files,
        f"Add {len(cases)} slash-bad case(s)",
        token,
        pr=not args.direct,
    )
    for case in cases:
        case.meta.setdefault("sharedTo", []).append({"repo": args.repo, "at": now_iso()})
        case.save_meta()
    print(result.get("pullRequestUrl") or result.get("commitUrl") or json.dumps(result))
    return 0
