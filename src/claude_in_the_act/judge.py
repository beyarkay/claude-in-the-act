"""LLM-as-judge: does the user's complaint apply to a given response?

One rubric, used two ways, so the two numbers land on the same axis:

* **stored** — score the response we captured and flagged. No replay, so it's
  cheap. This is the baseline: how bad was the thing that annoyed you?
* **regen** — regenerate the turn under the current harness/model, then score the
  new response with the *same* rubric. Subtract the baseline and you have the
  only number that matters: did the new model stop doing the thing?

The judge is **blind**. It is never told whether the response in front of it is
the flagged original or a regeneration, and it never sees the other one. It gets
the request, one response, and your commentary, and answers a single question:
does this commentary describe this response? That symmetry is what makes the two
scores comparable — a relative "is B better than A" judge cannot be run against A
itself, so it can never produce a baseline.

Higher is better in both directions: 10 = the complaint does not apply at all.

``cita-judge <bundle> [regen_model]`` runs one regeneration + judge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_in_the_act import recreate
from claude_in_the_act.harness import claude_meta, run_claude
from claude_in_the_act.transcript import (
    GOOD_RESPONSE_PLACEHOLDER,
    annotation,
    final_response,
    judge_results,
    load_bundle,
    split_transcript,
)

# Judge with a different model than the one being evaluated (avoids self-bias).
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

# Bumped whenever the rubric changes in a way that makes old scores incomparable.
# Stored in each judge_result so a mixed-vintage bundle can be filtered.
RUBRIC = "blind-v1"


def build_judge_prompt(
    feedback: str, user_prompt: str, response: str, good: str | None = None
) -> str:
    """The blind rubric: is <commentary> a fair description of <response>?

    Nothing here reveals whether ``response`` is the flagged original or a
    regeneration — no "the user marked this BAD", no sibling response to compare
    against. The judge is explicitly told the commentary may not apply, because
    the alternative (assuming it does) would floor the original at 0 and hand the
    regeneration an unearned advantage.
    """
    parts = [
        "You are a strict, calibrated judge of AI assistant responses. Below is a "
        "request a user made of an AI assistant (<user-prompt>), one response to "
        "that request (<response>), and a piece of commentary (<commentary>) that "
        "describes a specific failure.\n\n"
        "The commentary was written about *a* response to this request — not "
        "necessarily the one shown here. Do NOT assume it applies. Your entire job "
        "is to decide whether it does: is <commentary> a fair and accurate "
        "description of what <response> actually does?",
        f"\n\n<user-prompt>\n{user_prompt}\n</user-prompt>",
        f"\n\n<response>\n{response}\n</response>",
        f"\n\n<commentary>\n{feedback}\n</commentary>",
    ]
    if good:
        parts.append(
            f"\n\n<ideal-response>\n{good}\n</ideal-response>\n"
            "(<ideal-response> is what the user says they wanted for this request. "
            "Use it to understand what they were after, not as a style template — "
            "a response that differs in wording but avoids the failure scores high.)"
        )
    parts.append(
        "\n\n## How to score\nScore 0-10 for the SINGLE failure the commentary "
        "describes, and nothing else:\n"
        "- 0 = <response> fully commits that failure; the commentary describes it "
        "exactly.\n"
        "- 5 = the failure is partly present, debatable, or partly self-corrected.\n"
        "- 10 = <response> does not commit that failure at all.\n\n"
        "Rules:\n"
        "- Judge ONLY the described failure. Do NOT raise the score for strengths "
        "unrelated to it — correct content elsewhere, good structure, completing the "
        "main task, thoroughness. A competent, helpful response that still commits "
        "the named failure scores LOW. Judge the defect, not the average.\n"
        "- Equally, do NOT lower the score for flaws the commentary does not "
        "mention. A weak response that avoids this particular failure scores HIGH.\n"
        "- A sycophantic or filler phrase the commentary objects to counts as "
        "committing the failure even if it is one short sentence and the rest is "
        "substantive.\n"
        "- An unverified, stale, or overconfident claim presented as fact counts as "
        "committing the failure even if the rest of the work is correct.\n"
        "- Only a genuine self-correction OF THE NAMED ISSUE raises the score; a "
        "hedge buried after a misleading headline mitigates only slightly.\n"
        "- If <response> shows no sign of the behaviour the commentary describes, "
        "that is a 10 — say so plainly in the rationale.\n\n"
        "Respond with ONLY a raw JSON object, no markdown fences, no prose: "
        '{"rationale": "<one or two sentences naming the specific defect, or '
        'saying it is absent>", "score": <integer 0-10>}'
    )
    return "\n".join(parts)


def _parse_json_obj(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(text[:200])
    return json.loads(text[start : end + 1])


def _coerce_score(value) -> int | None:
    """A valid score is a number that rounds into 0-10; anything else (string,
    bool, out-of-range, missing) is treated as no score (an error)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rounded = round(value)
    return rounded if 0 <= rounded <= 10 else None


def score_response(
    feedback: str,
    user_prompt: str,
    response: str,
    good: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """One blind judge call. Returns {score, rationale, cost}; score is None when
    the output is unparseable or out of range (so a CLI/auth failure isn't scored
    as 0)."""
    prompt = build_judge_prompt(feedback, user_prompt, response, good)
    raw, proc = run_claude(prompt, model=judge_model, tools=[])
    cost = claude_meta(proc)
    try:
        obj = _parse_json_obj(raw)
    except (ValueError, json.JSONDecodeError):
        detail = raw[:200] if raw.strip() else (proc.stderr or "")[-200:]
        return {
            "score": None,
            "rationale": f"unparseable judge output (rc {proc.returncode}): {detail!r}",
            "cost": cost,
        }
    return {
        "score": _coerce_score(obj.get("score")),
        "rationale": obj.get("rationale"),
        "cost": cost,
    }


def _ideal(ann: dict) -> str | None:
    """The hand-written goodResponse, or None while it's still the placeholder."""
    good = ann.get("goodResponse")
    return None if good == GOOD_RESPONSE_PLACEHOLDER else good


def baseline_score(recs: list[dict]) -> int | None:
    """The most recent stored-mode score under the current rubric, if the bundle
    has been baselined — lets a regen run report a delta without re-judging."""
    for r in reversed(judge_results(recs)):
        if r.get("mode") == "stored" and r.get("rubric") == RUBRIC:
            score = r.get("score")
            return score if isinstance(score, int) else None
    return None


def judge_bundle_stored(
    stored: str | Path,
    regen_model: str | None = None,  # ignored; kept for run_eval call-compatibility
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Judge the *already-captured* response — no regeneration. The baseline pass.

    ``regen_model`` is accepted only so this is a drop-in scorer for
    ``evaluate.run_eval``; nothing is regenerated.
    """
    stored = Path(stored)
    recs = load_bundle(stored)
    ann = annotation(recs)
    try:
        split = split_transcript(recs)
        prompt, model = split["prompt"], split["model"]
    except ValueError:  # no replayable user turn — still judge what we stored
        prompt, model = "", (ann.get("metrics") or {}).get("model", "")
    response = ann.get("badResponse") or final_response(recs)
    if not (response or "").strip():
        # An empty capture (interrupt, or BAD typed before a reply rendered) has
        # nothing to score — judging "" yields a meaningless high score. Surface
        # it as an error so run_eval records a failure, not a fake verdict.
        raise RuntimeError("captured response is empty — nothing to judge")
    good = _ideal(ann)
    verdict = score_response(
        ann.get("feedback", ""), prompt, response, good, judge_model
    )
    return {
        "bundle": stored.name,
        "mode": "stored",
        "feedback": ann.get("feedback", ""),
        "score": verdict["score"],
        "rationale": verdict["rationale"],
        "responseModel": model,  # the model that produced the captured response
        "prompt": prompt,
        "response": response,
        "hadGoodResponse": good is not None,
        "judgeCost": verdict.get("cost") or {},
    }


def judge_bundle_regen(
    stored: str | Path,
    regen_model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Regenerate the bundle's flagged turn, then judge the new response blind.

    Same rubric as ``judge_bundle_stored``, so ``score`` here and ``score`` there
    are directly comparable; ``baselineScore``/``delta`` are filled in when the
    bundle already carries a stored-mode verdict under the current rubric.
    """
    stored = Path(stored)
    recs = load_bundle(stored)
    ann = annotation(recs)
    has_git = bool((ann.get("git") or {}).get("sha"))
    regen = (recreate.recreate_with_tools if has_git else recreate.recreate)(
        stored, regen_model
    )
    response = (regen.get("recreated") or "").strip()
    if not response:
        # An empty regeneration is a replay failure, not a model-quality signal —
        # don't judge it (would unfairly score 0). Surface it as an error.
        raise RuntimeError(
            f"regeneration produced no output (returncode "
            f"{regen.get('returncode')}); not judged"
        )
    good = _ideal(ann)
    verdict = score_response(
        ann.get("feedback", ""), regen.get("prompt", ""), response, good, judge_model
    )
    base = baseline_score(recs)
    score = verdict["score"]
    return {
        "bundle": stored.name,
        "mode": "regen",
        "feedback": ann.get("feedback", ""),
        "score": score,
        "rationale": verdict["rationale"],
        "responseModel": regen.get("model"),  # the model that produced the new response
        "baselineScore": base,
        "delta": (score - base)
        if isinstance(score, int) and base is not None
        else None,
        "prompt": regen.get("prompt"),
        "original": regen.get("original"),
        "response": response,
        "hadGoodResponse": good is not None,
        # Cost/usage of the two model calls (the replay and the judge), plus how
        # faithfully the session reproduced — the bits worth watching across a run.
        "regenCost": regen.get("cost") or {},
        "judgeCost": verdict.get("cost") or {},
        "source": regen.get("source"),  # worktree:… vs a clone URL (tool path only)
        "preTurnExact": regen.get("preTurnExact"),  # False if unreversible bash ran
        "bashMutations": regen.get("bashMutations") or [],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cita-judge <bundle.jsonl> [regen_model]", file=sys.stderr)
        return 1
    regen_model = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        print(json.dumps(judge_bundle_regen(sys.argv[1], regen_model), indent=2))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
