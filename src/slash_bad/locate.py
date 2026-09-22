"""Find the message a complaint is actually about.

``/bad`` flags the agent's last message, but people often complain about
something said earlier in the same turn. When the judge can't find the
complaint in the flagged message, this checks the turn's earlier messages,
newest first, and moves the case to the one the complaint fits. If none fits,
the case is marked unmatched and left out of comparisons, because a replay of
the wrong message measures nothing.
"""

from __future__ import annotations

import json
import shutil

from slash_bad import claude_code, codex, judge
from slash_bad.store import Case, write_jsonl
from slash_bad.view import judge_context

UNMATCHED_AT = 7  # a flagged message scoring this or more doesn't show the complaint
MATCHED_AT = 3  # an earlier message scoring this or less is the one to use
MAX_CANDIDATES = 6

_MODS = {"claude-code": claude_code, "codex": codex}


def _is_prompt(harness: str, rec: dict) -> bool:
    if harness == "codex":
        return codex._is_prompt(codex._item(rec))
    return claude_code.is_prompt(rec)


def candidates(case: Case) -> list[dict]:
    """Earlier agent messages in the flagged turn, newest first:
    ``[{cut, flagged}]`` where the context would be ``case.context[:cut]``."""
    mod = _MODS[case.harness]
    recs = case.context
    last_prompt = max((i for i, r in enumerate(recs) if _is_prompt(case.harness, r)), default=-1)
    out = []
    while len(out) < MAX_CANDIDATES:
        try:
            sp = mod.split(recs)
        except ValueError:
            break
        if sp["cut"] <= last_prompt:
            break
        out.append({"cut": sp["cut"], "flagged": sp["flagged"]})
        recs = recs[: sp["cut"]]
    return out


def relocate(case: Case, judge_model: str, flagged_score: int | None) -> dict:
    """Move the case to a better-matching earlier message if there is one.
    Returns ``{"status": "kept" | "moved" | "unmatched", ...}``."""
    if flagged_score is None or flagged_score < UNMATCHED_AT:
        return {"status": "kept"}
    if case.meta.get("unmatched"):
        return {"status": "unmatched"}
    mod = _MODS[case.harness]
    best = None
    for cand in candidates(case):
        earlier, turn = judge_context(mod.steps(case.context[: cand["cut"]]))
        verdict = judge.score(case.complaint, earlier, turn, cand["flagged"], judge_model)
        if verdict.get("score") is not None and (
            best is None or verdict["score"] < best[1]["score"]
        ):
            best = (cand, verdict)
        if best and best[1]["score"] <= 1:
            break
    if not best or best[1]["score"] > MATCHED_AT:
        case.meta["unmatched"] = {"judge": judge_model, "flaggedScore": flagged_score}
        case.save_meta()
        return {"status": "unmatched"}
    cand, verdict = best
    context = case.context
    for name in ("context.jsonl", "results.jsonl"):
        src = case.path / name
        if src.exists():
            shutil.copy(src, case.path / name.replace(".jsonl", ".before-relocate.jsonl"))
    write_jsonl(case.path / "context.jsonl", context[: cand["cut"]])
    (case.path / "results.jsonl").unlink(missing_ok=True)
    case.meta["relocated"] = {
        "fromFlagged": case.meta.get("flagged", "")[:500],
        "fromContextRecords": len(context),
        "flaggedScore": flagged_score,
    }
    case.meta["flagged"] = cand["flagged"]
    case.save_meta()
    case.add_result(
        {
            "kind": "verdict",
            "target": "original",
            "judge": judge_model,
            "rubric": judge.RUBRIC,
            "responseModel": case.meta.get("model"),
            **verdict,
        }
    )
    return {
        "status": "moved",
        "score": verdict["score"],
        "flagged": json.dumps(cand["flagged"][:80]),
    }
