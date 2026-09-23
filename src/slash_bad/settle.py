"""Re-run the old model and judge a case in the background as soon as it's saved.

The old model is re-run while the files, date and environment still match the
conversation, and for Claude Code that first replay's request is saved and sent
unchanged to every later model (see ``claude_code.replay``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from slash_bad import HOME, REPLAY_ENV, compare, llm
from slash_bad.proxy import SentinelProxy
from slash_bad.store import Case, now_iso

LOG = "settle.log"
OFF_ENV = "SLASH_BAD_NO_SETTLE"


def enabled(cfg: dict) -> bool:
    return not os.environ.get(OFF_ENV) and cfg.get("settle", "on") != "off"


def spawn(case: Case) -> None:
    """Start ``slash-bad settle <case>`` detached, so the hook returns at once."""
    HOME.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[1]
    env = {**os.environ, REPLAY_ENV: "1", "PYTHONPATH": str(src)}
    with open(HOME / LOG, "a") as log:
        subprocess.Popen(
            [sys.executable, "-m", "slash_bad.cli", "settle", case.id],
            cwd=HOME,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def settle(case: Case, judge_model: str | None = None) -> dict:
    judge_model = judge_model or llm.default_model()
    old = case.meta.get("model") or ""
    print(f"{now_iso()} settling {case.id} (old model {old or 'unknown'}, judge {judge_model})")
    flagged = compare.baseline(case, judge_model)
    control: dict = {"error": "no old model recorded"}
    if old:
        with ExitStack() as stack:
            url = stack.enter_context(SentinelProxy()) if case.harness == "claude-code" else None
            control = compare.scored_replays(case, old, 1, judge_model, "deny", url, False)
    print(
        f"{now_iso()} settled {case.id}: flagged {flagged.get('score')}, "
        f"old model again {control.get('score')} {control.get('error') or ''}".rstrip()
    )
    return {"flagged": flagged, "control": control}
