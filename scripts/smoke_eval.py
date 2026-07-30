#!/usr/bin/env python3
"""Run the eval over a chosen set of bundles, watching cost + reproducibility.

Two modes:
  --mode stored    judge the already-captured response directly (cheap; one
                   judge call, no replay). The baseline pass.
  --mode regen     replay the turn under the current harness, then judge the new
                   response on the SAME blind rubric (expensive; reconstructs
                   the repo). Subtract the two for the before/after.

``--save`` appends each verdict to its bundle .jsonl (a ``judge_result`` record)
and writes a combined results file, so the run is viewable later.

A note on size: ``transcript_tokens`` is the whole appended session *log* (bytes
/ 4 ≈ tokens of every turn + tool result ever logged), NOT a live context window
— Claude Code caps and compacts context at ~1M, so a replay never actually sends
a multi-million-token prompt. We use the log size only to pick cheap bundles and
to rough-extrapolate the regen-mode bill.

Usage:
    python3 scripts/smoke_eval.py                       # 5 smallest, judge-only
    python3 scripts/smoke_eval.py --smallest 41 --save  # judge all, persist
    python3 scripts/smoke_eval.py --workers 24          # 24-way parallel judges
    python3 scripts/smoke_eval.py --max-ktok 250        # all under ~250k log-tok
    python3 scripts/smoke_eval.py FILE [FILE...]        # explicit bundle(s)
    python3 scripts/smoke_eval.py --mode regen --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from claude_in_the_act import DATA_DIR, evaluate, judge
from claude_in_the_act.transcript import append_judgement

# A live context window can't exceed ~1M tokens (and Claude Code compacts well
# before that), so a regen replay never sends a whole multi-MB log — cap each
# bundle here when estimating replay input, or the bill is wildly overstated.
CONTEXT_CAP = 1_000_000


def transcript_tokens(bundle: Path) -> int:
    """Tokens-equivalent of the whole session *log* (stored transcriptBytes / 4,
    falling back to file size) — a size proxy, not a live context window. Reads
    only the file tail (the bad_annotation is the last line)."""
    size = bundle.stat().st_size
    with bundle.open("rb") as f:
        f.seek(max(0, size - 300_000))
        tail = f.read().decode("utf-8", "replace")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "bad_annotation":
            tb = (obj.get("metrics") or {}).get("transcriptBytes") or size
            return int(tb) // 4
    return size // 4


def classify(r: dict) -> str:
    """One-word reproducibility/judge outcome for a result row."""
    score = r.get("score")
    rationale = (r.get("rationale") or "").lower()
    if isinstance(score, int):
        return "ok"
    if "produced no output" in rationale or "nothing to judge" in rationale:
        return "empty-response"
    if "could not reconstruct" in rationale:
        return "reconstruct-fail"
    if "unparseable judge" in rationale:
        return "judge-unparseable"
    if rationale.startswith("error:"):
        return "error"
    return "no-score"


def _usd(cost: dict | None, key: str = "costUsd") -> float:
    v = (cost or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def select(args: argparse.Namespace) -> list[Path]:
    if args.bundles:
        out = []
        for b in args.bundles:
            p = Path(b)
            out.append(p if p.exists() else DATA_DIR / b)
        return out
    pool = sorted(DATA_DIR.glob("*.jsonl"), key=transcript_tokens)
    if args.max_ktok is not None:
        pool = [p for p in pool if transcript_tokens(p) <= args.max_ktok * 1000]
    return pool[: args.smallest]


def histogram(results: list[dict]) -> str:
    """A 0-10 score bar chart — the calibration signal. In 'stored' mode a low
    skew means the judge agrees the flagged responses were bad."""
    scores = [r["score"] for r in results if isinstance(r.get("score"), int)]
    if not scores:
        return "  (no scored bundles)"
    lines = []
    for s in range(11):
        c = scores.count(s)
        lines.append(f"  {s:>2} | {'█' * c}{' ' * (c == 0)} {c}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundles", nargs="*", help="explicit bundle paths/names")
    ap.add_argument("--smallest", type=int, default=5, help="N smallest bundles")
    ap.add_argument(
        "--max-ktok", type=int, default=None, help="cap log size (k tok-equiv)"
    )
    ap.add_argument(
        "--mode",
        choices=("stored", "regen"),
        default="stored",
        help="stored = judge the captured response (cheap, no replay; the "
        "baseline); regen = replay the turn then judge the new response",
    )
    ap.add_argument("--model", default=None, help="replay model (regen mode only)")
    ap.add_argument("--judge-model", default=None, help="override the judge model")
    ap.add_argument("--workers", type=int, default=8, help="parallel judge calls")
    ap.add_argument(
        "--save",
        action="store_true",
        help="append each verdict to its bundle .jsonl + write a combined report",
    )
    args = ap.parse_args()

    bundles = select(args)
    if not bundles:
        print(f"no bundles selected from {DATA_DIR}", file=sys.stderr)
        return 1

    scorer = (
        judge.judge_bundle_regen if args.mode == "regen" else judge.judge_bundle_stored
    )
    judge_model = args.judge_model or judge.DEFAULT_JUDGE_MODEL
    work = "replay+judge" if args.mode == "regen" else "judge stored response"
    print(
        f"smoke eval [{args.mode}]: {len(bundles)} bundle(s), {work}, "
        f"judge={judge_model}"
        + (
            f", replay-model={args.model or '(captured)'}"
            if args.mode == "regen"
            else ""
        )
        + "\n",
        file=sys.stderr,
    )
    for b in bundles:
        print(
            f"  · {b.name}  (~{transcript_tokens(b) / 1000:.0f}k tok-log)",
            file=sys.stderr,
        )
    print(f"\nparallelism: {args.workers} · save: {args.save}\n", file=sys.stderr)

    def progress(done: int, n: int, r: dict) -> None:
        cost = _usd(r.get("regenCost")) + _usd(r.get("judgeCost"))
        s = r.get("score")
        print(
            f"  [{done}/{n}] {r['bundle'][:34]:<34} {classify(r):<16} "
            f"score={s if s is not None else '-'}  ${cost:.4f}",
            file=sys.stderr,
        )

    results = evaluate.run_eval(
        [str(b) for b in bundles],
        args.model,
        judge_model=judge_model,
        max_workers=args.workers,
        on_done=progress,
        scorer=scorer,
    )

    # Persist each verdict back into its bundle (+ a combined report) so the run
    # is viewable later. One timestamp for the whole batch.
    if args.save:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        saved = 0
        for r, b in zip(results, bundles):
            if isinstance(r.get("score"), int):  # don't persist failed judge calls
                append_judgement(
                    b,
                    r,
                    mode=args.mode,
                    judge_model=judge_model,
                    judged_at=stamp,
                    rubric=judge.RUBRIC,
                )
                saved += 1
        report = DATA_DIR / f"eval-{args.mode}-{stamp}.json"
        report.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "judgeModel": judge_model,
                    "judgedAt": stamp,
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(
            f"\nsaved {saved} verdict(s) into bundles + report {report}",
            file=sys.stderr,
        )

    # --- aggregate ----------------------------------------------------------
    outcomes: dict[str, int] = {}
    total_cost = 0.0
    total_in = total_out = 0
    replay_tok_used = 0  # log size capped at the context window (regen estimate)
    for r, b in zip(results, bundles):
        outcomes[classify(r)] = outcomes.get(classify(r), 0) + 1
        for c in (r.get("regenCost") or {}, r.get("judgeCost") or {}):
            total_cost += _usd(c)
            # input-side = fresh + cache-write + cache-read (the bulk is usually
            # cached — counting only `inputTokens` wildly undercounts the work).
            total_in += (
                int(c.get("inputTokens") or 0)
                + int(c.get("cacheCreationTokens") or 0)
                + int(c.get("cacheReadTokens") or 0)
            )
            total_out += int(c.get("outputTokens") or 0)
        replay_tok_used += min(transcript_tokens(b), CONTEXT_CAP)

    n = len(results)
    ok = outcomes.get("ok", 0)
    scored = [r["score"] for r in results if isinstance(r.get("score"), int)]
    print("\n" + "=" * 72)
    print(f"SMOKE EVAL [{args.mode}] — {n} bundle(s)")
    print("=" * 72)
    print("outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    print(f"scored: {ok}/{n} ({100 * ok / n:.0f}%) · failures: {n - ok}/{n}")
    if scored:
        print(
            f"mean score: {sum(scored) / len(scored):.1f}/10  (lower = judge agrees it was bad)"
        )
        print("score distribution:")
        print(histogram(results))
    print(f"model tokens: {total_in:,} in (incl. cache) · {total_out:,} out")
    print(f"total cost: ${total_cost:.4f} · ${total_cost / n:.4f}/bundle avg")

    # reproducibility fidelity (regen path only): exact pre-turn state or not
    approxs = [
        r for r in results if r.get("preTurnExact") is False or r.get("bashMutations")
    ]
    if approxs:
        print(
            f"reconstruction was APPROXIMATE on {len(approxs)} bundle(s) "
            f"(unreversible bash in the flagged turn):"
        )
        for r in approxs:
            print(
                f"  · {r['bundle'][:40]}: "
                f"{len(r.get('bashMutations') or [])} bash mutation(s)"
            )

    # --- extrapolate to the whole dataset -----------------------------------
    all_bundles = sorted(DATA_DIR.glob("*.jsonl"))
    if total_cost > 0 and ok:
        print("-" * 72)
        if args.mode == "regen" and replay_tok_used > 0:
            all_tok = sum(min(transcript_tokens(b), CONTEXT_CAP) for b in all_bundles)
            rate = total_cost / (replay_tok_used / 1e6)  # $ per million replay-tokens
            print(
                f"measured rate: ${rate:.2f} per million replay input-tokens "
                f"(each bundle capped at ~{CONTEXT_CAP // 1_000_000}M — context compacts)"
            )
            print(
                f"EXTRAPOLATED full run of all {len(all_bundles)} bundles "
                f"(~{all_tok / 1e6:.1f}M capped replay-tokens): ~${rate * all_tok / 1e6:.0f}"
            )
            print(
                "  (rough: caps each bundle's log at the context window, then "
                "ignores prompt caching + output-token variance)"
            )
        else:  # stored mode: cost is ~flat per bundle (input is just the reply)
            per = total_cost / ok
            print(f"avg cost/bundle: ${per:.4f} (judge call only — no replay)")
            print(
                f"EXTRAPOLATED full run of all {len(all_bundles)} bundles: "
                f"~${per * len(all_bundles):.2f}"
            )
    print("=" * 72)

    # full per-bundle JSON for the curious / for saving (drop the bulky text)
    for r in results:
        for k in ("original", "response", "prompt", "badResponse"):
            r.pop(k, None)
    print("\nper-bundle detail (JSON):")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
