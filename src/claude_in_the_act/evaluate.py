"""Aggregate eval: run the judge over every captured response (in parallel) and
emit one scored report — per-transcript score + rationale, plus an overall
average. A Boyd-specific measure of how much a model fixes what you dislike.

Two modes, both scored on the same blind rubric so their numbers subtract:

``cita-eval --mode stored`` (default, cheap) judges the responses you captured —
the baseline. ``cita-eval --mode regen [--model M]`` replays each flagged turn
under ``M`` and judges the new response. Reports land in
``data/eval-<mode>-<ts>-<model>.md``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from datetime import datetime
from pathlib import Path

from claude_in_the_act import DATA_DIR, judge
from claude_in_the_act.judge import DEFAULT_JUDGE_MODEL


def _judge_one(bundle: str, regen_model: str | None, judge_model: str, scorer) -> dict:
    try:
        return scorer(bundle, regen_model, judge_model)
    except Exception as e:  # one bad bundle shouldn't sink the whole run
        return {
            "bundle": Path(bundle).name,
            "feedback": "",
            "score": None,
            "rationale": f"ERROR: {type(e).__name__}: {e}",
            "regenModel": regen_model,
        }


def run_eval(
    bundles: list[str],
    regen_model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_workers: int = 4,
    on_done=None,
    scorer=None,
) -> list[dict]:
    """Judge every bundle concurrently; results keep the input order.

    ``scorer`` picks the eval mode: it defaults to the cheap
    ``judge.judge_bundle_stored``; pass ``judge.judge_bundle_regen`` for the
    replay-then-judge pass. Resolved here (not as a default arg) so the tests'
    ``monkeypatch`` of the judge entry points still takes effect.
    """
    scorer = scorer or judge.judge_bundle_stored
    results: list[dict | None] = [None] * len(bundles)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_judge_one, b, regen_model, judge_model, scorer): i
            for i, b in enumerate(bundles)
        }
        done = 0
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if on_done:
                on_done(done, len(bundles), results[i])
    return results  # type: ignore[return-value]


def _cost_summary(results: list[dict]) -> str | None:
    """Total $ spend (replay + judge calls) across results, or None if no result
    carried cost data (e.g. an older run, or a fully-mocked report)."""
    total = 0.0
    seen = False
    for r in results:
        for key in ("regenCost", "judgeCost"):
            usd = (r.get(key) or {}).get("costUsd")
            if isinstance(usd, (int, float)):
                total += usd
                seen = True
    if not seen:
        return None
    n = len(results) or 1
    return f"## Cost: ${total:.2f} total · ${total / n:.3f}/transcript avg"


def reproduced(r: dict) -> bool:
    """Did the replay actually put the model back in the original situation?

    ``preTurnExact`` goes False when the flagged turn ran shell commands we can't
    reverse from the transcript (``gh pr create``, ``git commit``). Those replays
    routinely meet a *different* world — the branch has since been pushed and the
    PR merged, so the model looks around, sees the work is already done, and says
    so. The blind judge correctly scores that 10 (the complaint doesn't describe
    the response) but the 10 means "never faced the situation", not "fixed it".
    Averaging those in silently converts reconstruction failures into wins.
    """
    return r.get("preTurnExact") is not False


def _delta_summary(results: list[dict]) -> str | None:
    """Baseline → new movement, over bundles that carry both scores *and* actually
    reproduced. ``None`` when nothing was baselined first."""
    usable = [r for r in results if reproduced(r)]
    pairs = [
        (r["baselineScore"], r["score"])
        for r in usable
        if isinstance(r.get("score"), int) and isinstance(r.get("baselineScore"), int)
    ]
    if not pairs:
        return None
    base = sum(b for b, _ in pairs) / len(pairs)
    new = sum(n for _, n in pairs) / len(pairs)
    better = sum(1 for b, n in pairs if n > b)
    same = sum(1 for b, n in pairs if n == b)
    worse = len(pairs) - better - same
    dropped = sum(
        1 for r in results if not reproduced(r) and isinstance(r.get("score"), int)
    )
    out = (
        f"## Change: {base:.1f} → {new:.1f} (**{new - base:+.1f}**) over {len(pairs)} "
        f"baselined · {better} better, {same} same, {worse} worse"
    )
    if dropped:
        out += (
            f"\n\n> {dropped} further bundle(s) scored but are **excluded**: the "
            f"replay could not restore the pre-turn state (unreversible shell "
            f"commands in the flagged turn), so the model did not face the "
            f"original situation and its score is not a model-quality signal."
        )
    return out


def format_report(
    results: list[dict], regen_model: str | None, judge_model: str, when: str
) -> str:
    vals = [r["score"] for r in results if isinstance(r.get("score"), int)]
    errors = len(results) - len(vals)
    avg = sum(vals) / len(vals) if vals else 0.0
    lo, hi = (min(vals), max(vals)) if vals else (0, 0)
    mode = next((r.get("mode") for r in results if r.get("mode")), "stored")
    out = [
        f"# claude-in-the-act eval [{mode}] — model: {regen_model or '(as captured)'}",
        f"_judge: {judge_model} ({judge.RUBRIC}) · {when}_",
        "",
        f"## Overall: avg {avg:.1f}/10 · min {lo} · max {hi} · "
        f"{len(vals)} scored, {errors} error(s), {len(results)} total",
        "",
    ]
    if (delta := _delta_summary(results)) is not None:
        out += [delta, ""]
    if (cost := _cost_summary(results)) is not None:
        out += [cost, ""]
    out += ["## Per-transcript (worst first)", ""]
    for r in sorted(
        results, key=lambda r: (r.get("score") is None, r.get("score") or 0)
    ):
        s = r.get("score")
        tag = f"{s}/10" if isinstance(s, int) else "?/10"
        base = r.get("baselineScore")
        if isinstance(base, int) and isinstance(s, int):
            tag += f" · was {base}/10 ({s - base:+d})"
        fb = " ".join((r.get("feedback") or "").split())
        out.append(f"### [{tag}] {fb[:80]}")
        out.append(f"- **why:** {r.get('rationale', '')}")
        out.append(f"- `{r.get('bundle', '')}`")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("stored", "regen"),
        default="stored",
        help="stored = judge the captured response (cheap, no replay; the "
        "baseline); regen = replay the turn then judge the new response",
    )
    ap.add_argument("--model", default=None, help="replay model (regen mode only)")
    ap.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="model doing the scoring"
    )
    ap.add_argument("--workers", type=int, default=4, help="parallel judge calls")
    args = ap.parse_args()

    bundles = sorted(str(b) for b in DATA_DIR.glob("*.jsonl"))
    if not bundles:
        print(f"no bundles in {DATA_DIR}", file=sys.stderr)
        return 1

    scorer = (
        judge.judge_bundle_regen if args.mode == "regen" else judge.judge_bundle_stored
    )
    print(
        f"judging {len(bundles)} transcripts [{args.mode}] "
        f"replay-model={args.model or '(as captured)'} "
        f"judge={args.judge_model} ...",
        file=sys.stderr,
    )

    def progress(done: int, n: int, r: dict) -> None:
        s = r.get("score")
        print(
            f"  [{done}/{n}] {r['bundle']} -> {s if s is not None else 'ERR'}",
            file=sys.stderr,
        )

    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = run_eval(
        bundles,
        args.model,
        judge_model=args.judge_model,
        max_workers=args.workers,
        on_done=progress,
        scorer=scorer,
    )
    report = format_report(results, args.model, args.judge_model, when)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_model = (args.model or "as-captured").replace("/", "-")
    out_path = DATA_DIR / f"eval-{args.mode}-{ts}-{safe_model}.md"
    out_path.write_text(report + "\n")
    print("\n" + report)
    print(f"\nsaved report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
