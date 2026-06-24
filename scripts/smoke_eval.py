#!/usr/bin/env python3
"""Smoke-test the eval on a cheap subset, watching cost + reproducibility.

The full eval re-loads each captured session and replays it against the model
(``recreate``/``recreate_with_tools``) before judging — re-sending the whole
pre-prompt context as input tokens, so cost scales with transcript size, which
spans ~1000x across the dataset. This runner deliberately samples the *cheap*
end so we can (a) confirm the pipeline works end-to-end, (b) measure real $ per
replay, and (c) see how often a session fails to reproduce — then extrapolate
the full-run cost from each bundle's stored ``transcriptBytes`` without paying
to replay the multi-million-token ones.

Usage:
    python3 scripts/smoke_eval.py                # 5 smallest bundles
    python3 scripts/smoke_eval.py --smallest 3   # N smallest
    python3 scripts/smoke_eval.py --max-ktok 250 # all under ~250k approx-tokens
    python3 scripts/smoke_eval.py FILE [FILE...] # explicit bundle(s)
    python3 scripts/smoke_eval.py --model claude-sonnet-4-6 --smallest 5

The replay model defaults to whatever produced each captured response (so this
scores the *existing* responses' own model); override with --model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claude_in_the_act import DATA_DIR, evaluate, judge


def approx_tokens(bundle: Path) -> int:
    """Cheap size proxy: stored transcriptBytes / 4, falling back to file size.
    Reads only the file tail (the bad_annotation is the last line)."""
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
    if "produced no output" in rationale:
        return "regen-empty"
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
    pool = sorted(DATA_DIR.glob("*.jsonl"), key=approx_tokens)
    if args.max_ktok is not None:
        pool = [p for p in pool if approx_tokens(p) <= args.max_ktok * 1000]
    return pool[: args.smallest]


def histogram(results: list[dict]) -> str:
    """A 0-10 score bar chart — the calibration signal. In 'existing' mode a low
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
    ap.add_argument("--max-ktok", type=int, default=None, help="cap approx-tokens (k)")
    ap.add_argument(
        "--mode",
        choices=("existing", "regen"),
        default="existing",
        help="existing = judge the stored bad response (cheap, no replay); "
        "regen = regenerate then judge the improvement (expensive)",
    )
    ap.add_argument("--model", default=None, help="replay model (regen mode only)")
    ap.add_argument("--judge-model", default=None, help="override the judge model")
    args = ap.parse_args()

    bundles = select(args)
    if not bundles:
        print(f"no bundles selected from {DATA_DIR}", file=sys.stderr)
        return 1

    scorer = judge.judge_bundle if args.mode == "regen" else judge.judge_bundle_existing
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
        print(f"  · {b.name}  (~{approx_tokens(b) / 1000:.0f}k tok)", file=sys.stderr)
    print("", file=sys.stderr)

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
        max_workers=4,
        on_done=progress,
        scorer=scorer,
    )

    # --- aggregate ----------------------------------------------------------
    outcomes: dict[str, int] = {}
    total_cost = 0.0
    total_in = total_out = 0
    approx_used = 0
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
        approx_used += approx_tokens(b)

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
        if args.mode == "regen" and approx_used > 0:
            all_ktok = sum(approx_tokens(b) for b in all_bundles) / 1000
            rate = total_cost / (approx_used / 1e6)  # $ per million approx-tokens
            print(
                f"measured rate: ${rate:.2f} per million approx-tokens (replay+judge)"
            )
            print(
                f"EXTRAPOLATED full run of all {len(all_bundles)} bundles "
                f"(~{all_ktok / 1000:.1f}M approx-tokens): ~${rate * all_ktok / 1000:.0f}"
            )
            print(
                "  (rough: ignores per-turn context resends, prompt caching, and "
                "output-token variance — treat as an order-of-magnitude floor)"
            )
        else:  # existing mode: cost is ~flat per bundle (input is just the reply)
            per = total_cost / ok
            print(f"avg cost/bundle: ${per:.4f} (judge call only — no replay)")
            print(
                f"EXTRAPOLATED full run of all {len(all_bundles)} bundles: "
                f"~${per * len(all_bundles):.2f}"
            )
    print("=" * 72)

    # full per-bundle JSON for the curious / for saving (drop the bulky text)
    for r in results:
        for k in ("original", "recreated", "prompt", "badResponse"):
            r.pop(k, None)
    print("\nper-bundle detail (JSON):")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
