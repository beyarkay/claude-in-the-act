"""Group cases by the behaviour they complain about.

A benchmark result reads better per behaviour than per transcript: "the new
model stopped padding answers with praise, but still states unchecked facts".
One model call proposes a small set of behaviours from all the complaints and
assigns each case to one; the labels are stored on the cases so they stay put.
Later cases are slotted into the existing behaviours (or a new one when none
fits), so adding a case never reshuffles the rest. ``regroup`` starts over.
"""

from __future__ import annotations

import json

from slash_bad import HOME, llm
from slash_bad.store import Case

FILE_NAME = "behaviours.json"

_PROMPT = """Below are complaints a user wrote about responses from their AI coding agent, one per line, each with an id.

Group them by the underlying behaviour the user dislikes. Use {target} behaviours or fewer; merge small groups into the closest larger one rather than leaving singletons, unless a complaint is truly unlike the rest. Name each behaviour in a few plain words from the user's point of view (e.g. "praises the user", "claims things it didn't check", "unexplained jargon", "too long before the point"). A complaint that mentions several problems goes under its main one.
{existing}
<complaints>
{lines}
</complaints>

Reply with ONLY raw JSON:
{{"behaviours": [{{"name": "...", "description": "<one sentence>", "goal": "<the behaviour the user wants instead>"}}], "assign": {{"<id>": "<behaviour name>"}}}}
Every id must be assigned, and every assigned name must appear in "behaviours".
{goal_rule}"""

_GOAL_RULE = """"goal" names what the user wants, phrased positively as something the model does, in a few plain words, so that "the new model improved at <goal>" reads naturally. Examples: "praises the user" → "answers without flattery"; "doesn't explain jargon" → "explains jargon, abbreviations and numbers"; "avoids pushing back" → "pushes back and flags real problems"."""

_GOALS_PROMPT = """Below are behaviours a user dislikes in their AI coding agent.

<behaviours>
{lines}
</behaviours>

{goal_rule}

Reply with ONLY raw JSON: {{"goals": {{"<behaviour name, exactly as given>": "<goal>"}}}}"""


def _path():
    return HOME / FILE_NAME


def load() -> list[dict]:
    try:
        return json.loads(_path().read_text())["behaviours"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []


def _save(behaviours: list[dict]) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps({"behaviours": behaviours}, indent=2))


def goals() -> dict[str, str]:
    """``{behaviour name: what the user wants instead}``."""
    return {b["name"]: b["goal"] for b in load() if b.get("goal")}


def _fill_goals(behaviours: list[dict], model: str) -> None:
    todo = [b for b in behaviours if not b.get("goal")]
    if not todo:
        return
    lines = "\n".join(f"- {b['name']}: {b.get('description', '')}" for b in todo)
    out = llm.ask_json(_GOALS_PROMPT.format(lines=lines, goal_rule=_GOAL_RULE), model) or {}
    for b in todo:
        goal = (out.get("goals") or {}).get(b["name"])
        if goal:
            b["goal"] = goal


def _ask(prompt: str, model: str) -> dict:
    out = llm.ask_json(prompt, model)
    if out is None:
        raise RuntimeError(f"grouping with {model} returned nothing usable")
    return out


def assign(cases: list[Case], model: str, regroup: bool = False) -> dict[str, str]:
    """``{case id: behaviour}``, asking the model only about unlabelled cases."""
    known = [] if regroup else load()
    todo = [c for c in cases if regroup or not c.meta.get("behaviour")]
    if todo:
        lines = "\n".join(f"{c.id}: {' '.join(c.complaint.split())[:400]}" for c in todo)
        existing = ""
        if known:
            names = "\n".join(f"- {b['name']}: {b.get('description', '')}" for b in known)
            existing = (
                "\nThese behaviours already exist. Put each complaint in one of them if it fits; "
                f"only add a new behaviour when none does.\n<existing-behaviours>\n{names}\n"
                "</existing-behaviours>\n"
            )
        target = max(3, min(12, round(len(cases) ** 0.5) + 2))
        prompt = _PROMPT.format(target=target, existing=existing, lines=lines, goal_rule=_GOAL_RULE)
        out = _ask(prompt, model)
        by_name = {b["name"]: b for b in known}
        for b in out.get("behaviours", []):
            by_name.setdefault(b["name"], b)
        for c in todo:
            name = (out.get("assign") or {}).get(c.id)
            if name:
                by_name.setdefault(name, {"name": name, "description": ""})
                c.meta["behaviour"] = name
                c.save_meta()
        used = {c.meta.get("behaviour") for c in cases} | {b["name"] for b in known}
        known = [b for n, b in by_name.items() if n in used]
    if any(not b.get("goal") for b in known):
        _fill_goals(known, model)
    if todo or known != load():
        _save(known)
    return {c.id: c.meta.get("behaviour", "") for c in cases}
