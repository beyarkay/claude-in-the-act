"""Aggregate eval: run the judge over every captured bad response (in parallel)
and emit one scored report — per-transcript score + rationale, plus an overall
XX/(10·N). A Boyd-specific measure of how much a model fixes what you dislike.

``cita-eval [model]`` regenerates each flagged response with ``[model]`` (the
harness default if omitted), judges each against your feedback, and writes a
markdown report to ``data/eval-<ts>-<model>.md``.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
from datetime import datetime
from pathlib import Path

from claude_in_the_act import judge
from claude_in_the_act.judge import DEFAULT_JUDGE_MODEL

DATA_DIR = Path.home() / "projects" / "claude-in-the-act" / "data"


def _judge_one(bundle: str, regen_model: str | None, judge_model: str) -> dict:
    try:
        return judge.judge_bundle(bundle, regen_model, judge_model)
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
) -> list[dict]:
    """Judge every bundle concurrently; results keep the input order."""
    results: list[dict | None] = [None] * len(bundles)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_judge_one, b, regen_model, judge_model): i
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


def format_report(
    results: list[dict], regen_model: str | None, judge_model: str, when: str
) -> str:
    scored = [r for r in results if isinstance(r.get("score"), int)]
    total = sum(r["score"] for r in scored)
    maximum = 10 * len(results)
    avg = total / len(scored) if scored else 0.0
    out = [
        f"# claude-in-the-act eval — model: {regen_model or '(harness default)'}",
        f"_judge: {judge_model} · {when} · {len(results)} transcripts_",
        "",
        f"## Overall: {total}/{maximum}   (avg {avg:.1f}/10 over {len(scored)} scored)",
        "",
        "## Per-transcript (worst first)",
        "",
    ]
    for r in sorted(
        results, key=lambda r: (r.get("score") is None, r.get("score") or 0)
    ):
        s = r.get("score")
        tag = f"{s}/10" if isinstance(s, int) else "?/10"
        fb = " ".join((r.get("feedback") or "").split())
        out.append(f"### [{tag}] {fb[:80]}")
        out.append(f"- **why:** {r.get('rationale', '')}")
        out.append(f"- `{r.get('bundle', '')}`")
        out.append("")
    return "\n".join(out)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    regen_model = args[0] if args else None
    bundles = sorted(str(b) for b in DATA_DIR.glob("*.jsonl"))
    if not bundles:
        print(f"no bundles in {DATA_DIR}", file=sys.stderr)
        return 1

    print(
        f"judging {len(bundles)} transcripts with model="
        f"{regen_model or '(default)'} (judge {DEFAULT_JUDGE_MODEL}) ...",
        file=sys.stderr,
    )

    def progress(done: int, n: int, r: dict) -> None:
        s = r.get("score")
        print(
            f"  [{done}/{n}] {r['bundle']} -> {s if s is not None else 'ERR'}",
            file=sys.stderr,
        )

    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = run_eval(bundles, regen_model, on_done=progress)
    report = format_report(results, regen_model, DEFAULT_JUDGE_MODEL, when)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_model = (regen_model or "default").replace("/", "-")
    out_path = DATA_DIR / f"eval-{ts}-{safe_model}.md"
    out_path.write_text(report + "\n")
    print("\n" + report)
    print(f"\nsaved report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
