"""LLM-as-judge: score a flagged response, two ways.

* **Score the known-bad response directly** (cheap): hand the judge model the
  user's request, the captured (bad) response, and the user's feedback, and ask
  how good that response actually is, taking the complaint seriously. No replay —
  just the stored bundle. This is the baseline/calibration pass: does the judge
  *agree* the responses we flagged were bad? (``score_existing`` /
  ``judge_bundle_existing``.)
* **Score a regenerated response** (expensive): regenerate the response under
  the current harness/model (``recreate``), then score 0-10 how much *better* the
  new response is than the original at addressing the feedback. (``score_response``
  / ``judge_bundle``.)

Both judges use higher = better, and use the hand-written ``goodResponse`` as the
target if it's been filled in. Run over every bundle and you get a Boyd-specific
eval of the things a model does that you dislike. ``cita-judge <bundle>
[regen_model]`` runs one regeneration+judge.
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
    load_bundle,
    split_transcript,
)

# Judge with a different model than the one being evaluated (avoids self-bias).
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def build_prompt(
    feedback: str, user_prompt: str, bad: str, good: str | None, recreated: str
) -> str:
    parts = [
        "You are a strict, calibrated judge of LLM responses. The user marked a "
        "previous LLM response as BAD and gave feedback about why it was bad; they "
        "may also have rewritten it into the response they wanted. Read the "
        "original (bad) response (in <bad-response> tags), the user's feedback (in "
        "<user-feedback> tags), and the user's rewritten response if available (in "
        "<ideal-response> tags). Then read the assistant's NEW response (in "
        "<new-response> tags) and judge (0-10) whether the new response is better "
        "in the ways the user described.",
        f"\n\n<user-prompt>\n{user_prompt}\n</user-prompt>",
        f"\n\n<bad-response>\n{bad}\n</bad-response>",
        f"\n\n<user-feedback>\n{feedback}\n</user-feedback>",
    ]
    if good:
        parts.append(f"\n\n<ideal-response>\n{good}\n</ideal-response>")
    parts.append(f"\n\n<new-response>\n{recreated}\n</new-response>")
    parts.append(
        "\n\n## Task\nScore 0-10 how much better the NEW response (<new-response>) "
        "is than the BAD one (<bad-response>), **specifically at addressing the "
        "user's feedback**"
        + (" and matching their ideal response" if good else "")
        + ". 0 = no better, or worse; 5 = partially addresses the feedback; "
        "10 = fully fixes exactly what the user disliked, with no regressions.\n"
        "Respond with ONLY a raw JSON object, no markdown fences, no prose: "
        '{"score": <integer 0-10>, "rationale": "<one or two sentences>"}'
    )
    return "\n".join(parts)


def build_score_prompt(
    feedback: str, user_prompt: str, bad: str, good: str | None
) -> str:
    """Judge the captured response *on its own* against the user's complaint —
    no regenerated response to compare to. Higher = better, so a response that
    fully earns the complaint scores low; one the judge thinks is actually fine
    scores high."""
    parts = [
        "You are a strict, calibrated judge of AI assistant responses. A user made "
        "a request, an assistant replied, and the user then marked that reply as BAD "
        "and explained why. Read the user's request (<user-prompt>), the assistant's "
        "response (<response>), and the user's complaint about it (<user-feedback>)"
        + (
            ", plus the response the user actually wanted (<ideal-response>)"
            if good
            else ""
        )
        + ". Judge how good the assistant's response actually is, taking the user's "
        "complaint seriously.",
        f"\n\n<user-prompt>\n{user_prompt}\n</user-prompt>",
        f"\n\n<response>\n{bad}\n</response>",
        f"\n\n<user-feedback>\n{feedback}\n</user-feedback>",
    ]
    if good:
        parts.append(f"\n\n<ideal-response>\n{good}\n</ideal-response>")
    parts.append(
        "\n\n## Task\nScore 0-10 how good the assistant's <response> is at serving "
        "the user's request, given their complaint"
        + (" and ideal response" if good else "")
        + ". 0 = the complaint is fully justified; the response fails the user in "
        "exactly that way. 5 = mixed; the complaint has real merit but the response "
        "is partly fine. 10 = the response is actually good and the complaint is "
        "minor or unjustified.\n"
        "Respond with ONLY a raw JSON object, no markdown fences, no prose: "
        '{"score": <integer 0-10>, "rationale": "<one or two sentences>"}'
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


def _run_judge(prompt: str, judge_model: str) -> dict:
    """One judge call from a built prompt. Returns {score, rationale, cost};
    score is None when the output is unparseable or out of range (so a CLI/auth
    failure isn't scored as 0)."""
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


def score_response(
    feedback: str,
    user_prompt: str,
    bad: str,
    good: str | None,
    recreated: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """One judge call scoring a regenerated response against the original."""
    return _run_judge(
        build_prompt(feedback, user_prompt, bad, good, recreated), judge_model
    )


def score_existing(
    feedback: str,
    user_prompt: str,
    bad: str,
    good: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """One judge call scoring the already-captured response directly (no replay)."""
    return _run_judge(build_score_prompt(feedback, user_prompt, bad, good), judge_model)


def judge_bundle(
    stored: str | Path,
    regen_model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Regenerate the bundle's bad response, then judge the improvement."""
    stored = Path(stored)
    ann = annotation(load_bundle(stored))
    has_git = bool((ann.get("git") or {}).get("sha"))
    regen = (recreate.recreate_with_tools if has_git else recreate.recreate)(
        stored, regen_model
    )
    if not (regen.get("recreated") or "").strip():
        # An empty regeneration is a replay failure, not a model-quality signal —
        # don't judge it (would unfairly score 0). Surface it as an error.
        raise RuntimeError(
            f"regeneration produced no output (returncode "
            f"{regen.get('returncode')}); not judged"
        )
    good = ann.get("goodResponse")
    if good == GOOD_RESPONSE_PLACEHOLDER:
        good = None  # placeholder not yet filled in → judge on feedback alone
    verdict = score_response(
        ann.get("feedback", ""),
        regen.get("prompt", ""),
        regen.get("original", ""),
        good,
        regen.get("recreated", ""),
        judge_model,
    )
    return {
        "bundle": stored.name,
        "feedback": ann.get("feedback", ""),
        "score": verdict["score"],
        "rationale": verdict["rationale"],
        "regenModel": regen.get("model"),
        "prompt": regen.get("prompt"),
        "original": regen.get("original"),
        "recreated": regen.get("recreated"),
        "hadGoodResponse": good is not None,
        # Cost/usage of the two model calls (the replay and the judge), plus how
        # faithfully the session reproduced — the bits worth watching across a run.
        "regenCost": regen.get("cost") or {},
        "judgeCost": verdict.get("cost") or {},
        "source": regen.get("source"),  # worktree:… vs a clone URL (tool path only)
        "preTurnExact": regen.get("preTurnExact"),  # False if unreversible bash ran
        "bashMutations": regen.get("bashMutations") or [],
    }


def judge_bundle_existing(
    stored: str | Path,
    regen_model: str | None = None,  # ignored; kept for run_eval call-compatibility
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Judge the *already-captured* bad response directly — no regeneration.

    The cheap baseline pass: send the judge only the user's prompt, the stored
    bad response, and the feedback (plus a hand-written ideal response if present),
    none of the expensive session replay. Tells you whether the judge agrees the
    responses you flagged were actually bad. ``regen_model`` is accepted only so
    this is a drop-in scorer for ``evaluate.run_eval``; nothing is regenerated.
    """
    stored = Path(stored)
    recs = load_bundle(stored)
    ann = annotation(recs)
    try:
        split = split_transcript(recs)
        prompt, model = split["prompt"], split["model"]
    except ValueError:  # no replayable user turn — still judge what we stored
        prompt, model = "", (ann.get("metrics") or {}).get("model", "")
    bad = ann.get("badResponse") or final_response(recs)
    good = ann.get("goodResponse")
    if good == GOOD_RESPONSE_PLACEHOLDER:
        good = None  # placeholder not yet filled in → judge on feedback alone
    verdict = score_existing(ann.get("feedback", ""), prompt, bad, good, judge_model)
    return {
        "bundle": stored.name,
        "feedback": ann.get("feedback", ""),
        "score": verdict["score"],
        "rationale": verdict["rationale"],
        "judgedModel": model,  # the model that produced the bad response
        "prompt": prompt,
        "badResponse": bad,
        "hadGoodResponse": good is not None,
        "judgeCost": verdict.get("cost") or {},
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cita-judge <bundle.jsonl> [regen_model]", file=sys.stderr)
        return 1
    regen_model = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        print(json.dumps(judge_bundle(sys.argv[1], regen_model), indent=2))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
