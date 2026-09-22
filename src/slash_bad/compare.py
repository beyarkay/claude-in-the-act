"""Run the personal benchmark: judge every stored bad response, re-sample it on
a new model, judge that blind on the same rubric, and tally the movement.

Everything is cached in each case's ``results.jsonl``: a baseline is judged
once per judge, a replay runs once per model, so re-running a comparison, or
changing the judge, only pays for what is missing.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time
import uuid
from contextlib import ExitStack

from slash_bad import claude_code, codex, judge
from slash_bad.proxy import SentinelProxy
from slash_bad.store import Case
from slash_bad.view import judge_context

BETTER_BY = 2  # points on the 0-10 scale before a change counts as a change

HARNESSES = {"claude-code": claude_code, "codex": codex}


def harness_for_model(model: str) -> str | None:
    """Which harness can run ``model``: None means "whichever the case came from"."""
    if model in ("latest", ""):
        return None
    if model.startswith(("gpt", "o3", "o4", "codex")):
        return "codex"
    return "claude-code"


def _latest(results: list[dict], **match) -> dict | None:
    for r in reversed(results):
        if all(r.get(k) == v for k, v in match.items()):
            return r
    return None


def response_for(replay: dict, variant: str) -> str:
    """What the judge sees for a replay.

    ``deny``: all the prose the model wrote once its tool calls were refused.
    ``stop``: its first message only, with any tool call it tried spelled out.
    """
    if variant == "deny" and replay.get("text"):
        return replay["text"]
    parts = [replay.get("first") or ""]
    parts += [f"[calls tool {c}]" for c in replay.get("firstToolCalls") or []]
    return "\n\n".join(p for p in parts if p).strip()


def baseline(case: Case, judge_model: str) -> dict:
    """The stored flagged response's verdict, judged once per judge + rubric."""
    cached = _latest(
        case.results(), kind="verdict", target="original", judge=judge_model, rubric=judge.RUBRIC
    )
    if cached and cached.get("score") is not None:
        return cached
    earlier, turn = judge_context(HARNESSES[case.harness].steps(case.context))
    verdict = judge.score(case.complaint, earlier, turn, case.meta.get("flagged", ""), judge_model)
    return case.add_result(
        {
            "kind": "verdict",
            "target": "original",
            "judge": judge_model,
            "rubric": judge.RUBRIC,
            "responseModel": case.meta.get("model"),
            **verdict,
        }
    )


def replayed(case: Case, model: str, proxy_url: str | None, fresh: bool) -> dict:
    if not fresh:
        cached = _latest(case.results(), kind="replay", requestedModel=model)
        if cached and not cached.get("error"):
            return cached
    mod = HARNESSES[case.harness]
    rec = mod.replay(case, model, proxy_url) if case.harness == "claude-code" else mod.replay(case, model)
    return case.add_result({"replayId": uuid.uuid4().hex[:12], **rec})


def judged_replay(case: Case, replay: dict, judge_model: str, variant: str) -> dict:
    cached = _latest(
        case.results(),
        kind="verdict",
        target=replay["replayId"],
        judge=judge_model,
        rubric=judge.RUBRIC,
        variant=variant,
    )
    if cached and cached.get("score") is not None:
        return cached
    response = response_for(replay, variant)
    if not response:
        return {"score": None, "rationale": "replay produced no response"}
    earlier, turn = judge_context(HARNESSES[case.harness].steps(case.context))
    verdict = judge.score(case.complaint, earlier, turn, response, judge_model)
    return case.add_result(
        {
            "kind": "verdict",
            "target": replay["replayId"],
            "judge": judge_model,
            "rubric": judge.RUBRIC,
            "variant": variant,
            "responseModel": replay.get("model"),
            **verdict,
        }
    )


def outcome(old: int | None, new: int | None) -> str:
    if old is None or new is None:
        return "error"
    if new - old >= BETTER_BY:
        return "better"
    if old - new >= BETTER_BY:
        return "worse"
    return "same"


def compare_case(
    case: Case, model: str, judge_model: str, variant: str, proxy_url: str | None, fresh: bool
) -> dict:
    row = {
        "case": case.id,
        "complaint": case.complaint,
        "harness": case.harness,
        "oldModel": case.meta.get("model", ""),
        "requestedModel": model,
    }
    try:
        base = baseline(case, judge_model)
        rep = replayed(case, model, proxy_url, fresh)
        if rep.get("error"):
            return {**row, "outcome": "error", "error": rep["error"], "old": base.get("score")}
        verdict = judged_replay(case, rep, judge_model, variant)
    except Exception as e:  # noqa: BLE001 — one broken case must not sink the run
        return {**row, "outcome": "error", "error": f"{type(e).__name__}: {e}"}
    old, new = base.get("score"), verdict.get("score")
    return {
        **row,
        "newModel": rep.get("model") or model,
        "old": old,
        "new": new,
        "outcome": outcome(old, new),
        "oldResponse": case.meta.get("flagged", ""),
        "oldWhy": base.get("rationale", ""),
        "newWhy": verdict.get("rationale", ""),
        "newResponse": response_for(rep, variant),
        "toolAttempts": rep.get("toolAttempts") or [],
        "costUsd": sum(
            c for c in (base.get("costUsd"), rep.get("costUsd"), verdict.get("costUsd")) if isinstance(c, (int, float))
        ),
    }


class Progress:
    """A small tqdm-style bar on stderr (the runtime is stdlib-only)."""

    def __init__(self, total: int, desc: str):
        self.total, self.desc, self.done, self.start = total, desc, 0, time.time()
        self.counts: dict[str, int] = {}
        self._draw()

    def update(self, row: dict) -> None:
        self.done += 1
        self.counts[row["outcome"]] = self.counts.get(row["outcome"], 0) + 1
        self._draw()

    def _draw(self) -> None:
        if not sys.stderr.isatty():
            if self.done:
                print(f"{self.desc}: {self.done}/{self.total}", file=sys.stderr)
            return
        width = 24
        filled = int(width * self.done / max(self.total, 1))
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0
        tally = " ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
        sys.stderr.write(
            f"\r{self.desc} [{'█' * filled}{'·' * (width - filled)}] {self.done}/{self.total}"
            f" · {elapsed:.0f}s elapsed · ~{eta:.0f}s left · {tally}\x1b[K"
        )
        if self.done == self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def run(
    cases: list[Case],
    model: str,
    judge_model: str = judge.DEFAULT_JUDGE,
    variant: str = "deny",
    workers: int = 8,
    fresh: bool = False,
) -> list[dict]:
    target = harness_for_model(model)
    eligible = [c for c in cases if target is None or c.harness == target]
    rows: list[dict | None] = [None] * len(eligible)
    bar = Progress(len(eligible), f"comparing against {model}")
    with ExitStack() as stack:
        proxy_url = None
        if any(c.harness == "claude-code" for c in eligible):
            proxy_url = stack.enter_context(SentinelProxy())
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(compare_case, c, model, judge_model, variant, proxy_url, fresh): i
                for i, c in enumerate(eligible)
            }
            for fut in cf.as_completed(futs):
                rows[futs[fut]] = fut.result()
                bar.update(rows[futs[fut]])
    return [r for r in rows if r is not None]
