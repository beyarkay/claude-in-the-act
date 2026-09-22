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
import os
import subprocess
import tempfile

from slash_bad import HOME, REPLAY_ENV
from slash_bad.store import Case

DEFAULT_MODEL = "sonnet"
FILE_NAME = "behaviours.json"

_PROMPT = """Below are complaints a user wrote about responses from their AI coding agent, one per line, each with an id.

Group them by the underlying behaviour the user dislikes. Use {target} behaviours or fewer; merge small groups into the closest larger one rather than leaving singletons, unless a complaint is truly unlike the rest. Name each behaviour in a few plain words from the user's point of view (e.g. "praises the user", "claims things it didn't check", "unexplained jargon", "too long before the point"). A complaint that mentions several problems goes under its main one.
{existing}
<complaints>
{lines}
</complaints>

Reply with ONLY raw JSON:
{{"behaviours": [{{"name": "...", "description": "<one sentence>"}}], "assign": {{"<id>": "<behaviour name>"}}}}
Every id must be assigned, and every assigned name must appear in "behaviours"."""


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


def _ask(prompt: str, model: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="slash-bad-group-") as cwd:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                model,
                "--system-prompt",
                "You answer with a single JSON object.",
                "--tools",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--output-format",
                "json",
            ],
            input=prompt,
            cwd=cwd,
            env={**os.environ, REPLAY_ENV: "1"},
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    try:
        raw = json.loads(proc.stdout).get("result", "")
        return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"grouping failed: {(proc.stderr or proc.stdout)[-300:]}") from e


def assign(cases: list[Case], model: str = DEFAULT_MODEL, regroup: bool = False) -> dict[str, str]:
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
        out = _ask(_PROMPT.format(target=target, existing=existing, lines=lines), model)
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
        _save([b for n, b in by_name.items() if n in used])
    return {c.id: c.meta.get("behaviour", "") for c in cases}
