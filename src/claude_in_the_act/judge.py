"""LLM-as-judge: score how much a regenerated response improves on the original.

For a captured bad response, regenerate it (``recreate``), then ask a judge model
to score 0-10 how much better the regenerated response is than the original *at
addressing the user's feedback* (and matching the hand-written ``goodResponse``
if present). Run over every bundle and you get a Boyd-specific eval of the things
a model does that you dislike.

``score_response`` is the judge call alone (cheap to iterate on); ``judge_bundle``
adds the regeneration. ``cita-judge <bundle> [regen_model]`` runs one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from claude_in_the_act import recreate

# Judge with a different model than the one being evaluated (avoids self-bias).
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def _annotation(recs: list[dict]) -> dict:
    return next((r for r in reversed(recs) if r.get("type") == "bad_annotation"), {})


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


def _parse_json_obj(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(text[:200])
    return json.loads(text[start : end + 1])


def score_response(
    feedback: str,
    user_prompt: str,
    bad: str,
    good: str | None,
    recreated: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """One judge call. Returns {score, rationale}."""
    prompt = build_prompt(feedback, user_prompt, bad, good, recreated)
    proc = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            judge_model,
            "--allowed-tools",
            "",
        ],
        capture_output=True,
        text=True,
    )
    raw = ""
    if proc.stdout.strip():
        try:
            raw = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            pass
    try:
        obj = _parse_json_obj(raw)
        return {"score": obj.get("score"), "rationale": obj.get("rationale")}
    except (ValueError, json.JSONDecodeError):
        return {"score": None, "rationale": f"unparseable judge output: {raw[:200]!r}"}


def judge_bundle(
    stored: str | Path,
    regen_model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Regenerate the bundle's bad response, then judge the improvement."""
    stored = Path(stored)
    recs = [json.loads(l) for l in stored.read_text().splitlines() if l.strip()]
    ann = _annotation(recs)
    has_git = bool((ann.get("git") or {}).get("sha"))
    regen = (recreate.recreate_with_tools if has_git else recreate.recreate)(
        stored, regen_model
    )
    verdict = score_response(
        ann.get("feedback", ""),
        regen.get("prompt", ""),
        regen.get("original", ""),
        ann.get("goodResponse"),
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
        "hadGoodResponse": bool(ann.get("goodResponse")),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cita-judge <bundle.jsonl> [regen_model]", file=sys.stderr)
        return 1
    regen_model = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(judge_bundle(sys.argv[1], regen_model), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
