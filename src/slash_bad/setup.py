"""Install the unprefixed ``/bad`` commands as user-level skills.

Plugin commands are namespaced (``/slash-bad:bad``), and Claude Code rejects an
unknown ``/bad`` before any hook sees it, so the short names live in
``~/.claude/skills`` (and ``~/.codex/skills``, typed as ``$bad`` there). The
plugin's SessionStart hook runs this whenever the plugin's location changes.
Only files carrying ``MARKER`` are ever overwritten or removed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MARKER = "<!-- installed by slash-bad; `slash-bad uninstall` removes it -->"

HOOK_MISSING = (
    "The slash-bad hook should have caught this before it reached you, so nothing was "
    "saved. Don't act on the complaint. Tell the user in one sentence that the slash-bad "
    "capture hook isn't running: in Claude Code, restart it or check `/plugin`; in Codex, "
    "approve the slash-bad hook in `/hooks`."
)

SKILLS = {
    "bad": {
        "description": "Flag the previous response as bad and save it to your slash-bad "
        "personal benchmark: /bad <what went wrong>.",
        "hint": "<what went wrong>",
        "body": HOOK_MISSING,
    },
    "bad-list": {
        "description": "List the cases in your slash-bad personal benchmark.",
        "hint": "",
        "body": "Run `{cli} list` and show the user its output verbatim, with no commentary.",
    },
    "bad-compare": {
        "description": "Re-run your saved /bad cases on a newer model and score whether it "
        "does better: /bad-compare <model|latest>.",
        "hint": "<model|latest> [--judge MODEL] [--samples N] [case ...]",
        "body": (
            "Run the user's personal benchmark with the shell tool:\n\n"
            "    {cli} compare $ARGUMENTS\n\n"
            "If no model was given, use `latest`. It re-samples every saved case on the new "
            "model and on the old one and has a blind judge score them, so it takes several "
            "minutes and spends tokens: run it in the background if you can, with a timeout "
            "of at least 30 minutes. When it finishes, show the scoresheet it printed "
            "verbatim in a code block, then the paths of the PNG card and the HTML report. "
            "Don't add commentary on the results unless asked."
        ),
    },
    "bad-share": {
        "description": "Anonymise your /bad cases and contribute them to the public "
        "slash-bad dataset on Hugging Face.",
        "hint": "[case ...]",
        "body": (
            "First run a dry run with the shell tool:\n\n"
            "    {cli} share --dry-run $ARGUMENTS\n\n"
            "Show the user what it reports: how many cases, what was redacted, and where the "
            "anonymised copies were written. Tell them to read those files, because code, "
            "file contents and project names are not removed automatically. Only if the user "
            "then explicitly says to go ahead, run the same command with `--yes` instead of "
            "`--dry-run`, and show them the pull request link it prints."
        ),
    },
}


def skill_text(name: str, cli: str) -> str:
    spec = SKILLS[name]
    front = [
        "---",
        f"name: {name}",
        f"description: {spec['description']}",
        "disable-model-invocation: true",
    ]
    if spec["hint"]:
        front.append(f"argument-hint: {spec['hint']}")
    front.append("---")
    return "\n".join(front) + f"\n{MARKER}\n\n" + spec["body"].format(cli=f'"{cli}"') + "\n"


def skill_dirs() -> list[Path]:
    dirs = [Path.home() / ".claude" / "skills"]
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    if codex_home.is_dir():
        dirs.append(codex_home / "skills")
    return dirs


def install(root: Path) -> list[Path]:
    cli = str((root / "bin" / "slash-bad").resolve())
    written = []
    for base in skill_dirs():
        for name in SKILLS:
            path = base / name / "SKILL.md"
            if path.exists() and MARKER not in path.read_text():
                continue  # the user's own skill with the same name; leave it alone
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(skill_text(name, cli))
            written.append(path)
    return written


def uninstall() -> list[Path]:
    removed = []
    for base in skill_dirs():
        for name in SKILLS:
            path = base / name / "SKILL.md"
            if path.exists() and MARKER in path.read_text():
                shutil.rmtree(path.parent)
                removed.append(path)
    return removed
