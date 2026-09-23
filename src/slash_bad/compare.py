"""Run the personal benchmark.

For each saved case, three responses at the same point in the same conversation
are judged blind against the user's complaint:

* ``flagged``: the response that annoyed the user.
* ``control``: the *old* model re-sampled on the same context.
* ``new``: the new model re-sampled on the same context.

Better / same / worse compares ``new`` with ``control``, not with ``flagged``.
Every case was saved *because* one sample went wrong, so a fresh sample of the
same old model usually scores higher too (regression to the mean); measuring
against the flagged response alone would credit the new model with that.
When the old model can't be run any more, the flagged response stands in and
the row says so.

Everything is cached in each case's ``results.jsonl``: a replay runs once per
model and sample, a verdict once per judge and rubric, so re-running a
comparison, or switching judge, only pays for what is missing.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time
import uuid
from contextlib import ExitStack

from slash_bad import claude_code, codex, judge, locate
from slash_bad.proxy import SentinelProxy
from slash_bad.store import Case
from slash_bad.view import judge_context

BETTER_BY = 2  # points on the 0-10 scale before one transcript's change counts
GROUP_BETTER_BY = 0.5  # the same, for a behaviour's average over its transcripts

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


def _judge(case: Case, target: str, response: str, judge_model: str, extra: dict) -> dict:
    match = {"kind": "verdict", "target": target, "judge": judge_model, "rubric": judge.RUBRIC}
    cached = _latest(case.results(), **match, **{k: v for k, v in extra.items() if k == "variant"})
    if cached and cached.get("score") is not None:
        return cached
    earlier, turn = judge_context(HARNESSES[case.harness].steps(case.context))
    verdict = judge.score(case.complaint, earlier, turn, response, judge_model)
    return case.add_result({**match, **extra, **verdict})


def baseline(case: Case, judge_model: str) -> dict:
    """The flagged response's verdict."""
    return _judge(
        case,
        "original",
        case.meta.get("flagged", ""),
        judge_model,
        {"responseModel": case.meta.get("model")},
    )


def replays(case: Case, model: str, n: int, proxy_url: str | None, fresh: bool) -> list[dict]:
    """``n`` replays of the case under ``model``, reusing cached ones."""
    have = (
        []
        if fresh
        else [
            r
            for r in case.results()
            if r.get("kind") == "replay"
            and r.get("requestedModel") == model
            and not r.get("error")
            and r.get("replayVersion") == HARNESSES[case.harness].REPLAY_VERSION
        ]
    )
    out = have[:n]
    mod = HARNESSES[case.harness]
    while len(out) < n:
        rec = (
            mod.replay(case, model, proxy_url)
            if case.harness == "claude-code"
            else mod.replay(case, model)
        )
        rec = case.add_result({"replayId": uuid.uuid4().hex[:12], **rec})
        if rec.get("error"):
            return out + [rec]
        out.append(rec)
    return out


def scored_replays(
    case: Case,
    model: str,
    n: int,
    judge_model: str,
    variant: str,
    proxy_url: str | None,
    fresh: bool,
) -> dict:
    """Mean score over ``n`` judged replays, or an ``error``."""
    reps = replays(case, model, n, proxy_url, fresh)
    if errs := [r for r in reps if r.get("error")]:
        return {"error": errs[0]["error"], "model": model}
    verdicts = []
    for rep in reps:
        response = response_for(rep, variant)
        if not response:
            return {"error": "replay produced no response", "model": model}
        verdicts.append(
            _judge(
                case,
                rep["replayId"],
                response,
                judge_model,
                {"variant": variant, "responseModel": rep.get("model")},
            )
        )
    scores = [v["score"] for v in verdicts if v.get("score") is not None]
    if len(scores) < len(verdicts):
        bad = next(v for v in verdicts if v.get("score") is None)
        return {"error": bad.get("rationale", "judge failed"), "model": model}
    return {
        "score": sum(scores) / len(scores),
        "scores": scores,
        "model": reps[0].get("model") or model,
        "response": response_for(reps[0], variant),
        "why": verdicts[0].get("rationale", ""),
        "judgeModel": verdicts[0].get("judgeModel"),
        "toolAttempts": [a for r in reps for a in r.get("toolAttempts") or []],
        "costUsd": sum(
            c
            for c in [r.get("costUsd") for r in reps] + [v.get("costUsd") for v in verdicts]
            if isinstance(c, (int, float))
        ),
    }


def outcome(reference: float | None, new: float | None, by: float = BETTER_BY) -> str:
    if reference is None or new is None:
        return "error"
    if new - reference >= by:
        return "better"
    if reference - new >= by:
        return "worse"
    return "same"


def compare_case(
    case: Case,
    model: str,
    judge_model: str,
    variant: str,
    proxy_url: str | None,
    fresh: bool,
    samples: int = 1,
    control: bool = True,
    locate_: bool = True,
) -> dict:
    old_model = case.meta.get("model", "")
    row = {
        "case": case.id,
        "complaint": case.complaint,
        "harness": case.harness,
        "oldModel": old_model,
        "requestedModel": model,
        "oldResponse": case.meta.get("flagged", ""),
        "behaviour": case.meta.get("behaviour", ""),
    }
    if old_model and old_model == model:
        return {**row, "outcome": "same-model"}
    try:
        base = baseline(case, judge_model)
        if locate_:
            loc = locate.relocate(case, judge_model, base.get("score"))
            if loc["status"] == "unmatched":
                return {**row, "outcome": "unmatched", "flagged": base.get("score")}
            if loc["status"] == "moved":
                base = baseline(case, judge_model)
                row["oldResponse"] = case.meta.get("flagged", "")
                row["relocated"] = True
        new = scored_replays(case, model, samples, judge_model, variant, proxy_url, fresh)
        if new.get("error"):
            return {**row, "outcome": "error", "error": new["error"], "flagged": base.get("score")}
        if old_model and new.get("model") == old_model:
            return {**row, "outcome": "same-model"}
        ctl = (
            scored_replays(case, old_model, samples, judge_model, variant, proxy_url, fresh)
            if control and old_model
            else {"error": "control disabled"}
        )
    except Exception as e:  # noqa: BLE001 — one broken case must not sink the run
        return {**row, "outcome": "error", "error": f"{type(e).__name__}: {e}"}
    flagged = base.get("score")
    has_control = not ctl.get("error")
    reference = ctl["score"] if has_control else flagged
    return {
        **row,
        "newModel": new["model"],
        "flagged": flagged,
        "control": ctl.get("score") if has_control else None,
        "controlError": None if has_control else ctl.get("error"),
        "new": new["score"],
        "newScores": new["scores"],
        "reference": reference,
        "vs": "control" if has_control else "flagged",
        "outcome": outcome(reference, new["score"]),
        "oldWhy": base.get("rationale", ""),
        "controlResponse": ctl.get("response", ""),
        "controlWhy": ctl.get("why", ""),
        "newResponse": new["response"],
        "newWhy": new["why"],
        "judgeModel": new.get("judgeModel") or base.get("judgeModel") or judge_model,
        "toolAttempts": new["toolAttempts"],
        "costUsd": sum(
            c
            for c in (base.get("costUsd"), new.get("costUsd"), ctl.get("costUsd"))
            if isinstance(c, (int, float))
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
    judge_model: str = "opus",
    variant: str = "deny",
    workers: int = 8,
    fresh: bool = False,
    samples: int = 1,
    control: bool = True,
    locate_: bool = True,
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
                ex.submit(
                    compare_case,
                    c,
                    model,
                    judge_model,
                    variant,
                    proxy_url,
                    fresh,
                    samples,
                    control,
                    locate_,
                ): i
                for i, c in enumerate(eligible)
            }
            for fut in cf.as_completed(futs):
                rows[futs[fut]] = fut.result()
                bar.update(rows[futs[fut]])
    return [r for r in rows if r is not None]
