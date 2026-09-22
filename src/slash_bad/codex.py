"""Codex: read a rollout, capture a case, replay it.

Same idea as ``claude_code``: the context is the rollout up to the last
assistant message, and a replay resumes it under the new model with every tool
call denied. Two Codex-specific wrinkles:

* ``codex exec resume`` always appends a user message; we send an empty one,
  the closest Codex allows to "just continue".
* The rollout records the model per turn, and a mismatch makes Codex compact
  the history and announce a model switch, so the replayed copy is rewritten to
  claim the target model all along. Encrypted reasoning is dropped: only the
  model that wrote it can read it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from slash_bad import HOME, REPLAY_ENV
from slash_bad.store import Case, read_jsonl
from slash_bad.view import as_text, tool_call_text

HARNESS = "codex"
REPLAY_VERSION = 2
CODEX_TIMEOUT = 900
DENY_REASON = (
    "Permission denied: the user has turned off tool use for this reply. Continue without tools."
)
# Harness-injected "user" messages that the user never typed.
_INJECTED = (
    "<environment_context>",
    "<recommended_plugins>",
    "<user_instructions>",
    "<skills_instructions>",
    "<permissions",
    "<apps_instructions>",
    "<model_switch>",
    "<turn_aborted>",
    "# AGENTS.md",
)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _item(rec: dict) -> dict:
    return rec.get("payload") or {} if rec.get("type") == "response_item" else {}


def _message_text(item: dict) -> str:
    return "\n".join(
        b.get("text", "")
        for b in item.get("content") or []
        if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text")
    ).strip()


def _is_prompt(item: dict) -> bool:
    if item.get("type") != "message" or item.get("role") != "user":
        return False
    text = _message_text(item)
    return bool(text) and not text.lstrip().startswith(_INJECTED)


def _call_args(item: dict) -> object:
    args = item.get("arguments", item.get("input"))
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def steps(recs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in recs:
        item = _item(r)
        kind = item.get("type")
        if _is_prompt(item):
            out.append({"kind": "user", "text": _message_text(item)})
        elif kind == "message" and item.get("role") == "assistant":
            if text := _message_text(item):
                out.append({"kind": "assistant", "text": text})
        elif kind in ("function_call", "custom_tool_call", "local_shell_call"):
            name = item.get("name") or kind
            out.append(
                {"kind": "tool_call", "name": name, "text": tool_call_text(name, _call_args(item))}
            )
        elif kind in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            if isinstance(output, dict):
                output = output.get("content") or output.get("output") or json.dumps(output)
            out.append({"kind": "tool_result", "text": as_text(output)})
        elif kind == "web_search_call":
            out.append(
                {
                    "kind": "tool_call",
                    "name": "web_search",
                    "text": tool_call_text("web_search", item.get("action")),
                }
            )
    return out


def _is_answer(rec: dict, final_only: bool) -> bool:
    item = _item(rec)
    if item.get("type") != "message" or item.get("role") != "assistant":
        return False
    if final_only and item.get("phase") != "final_answer":
        return False
    return bool(_message_text(item))


def split(recs: list[dict]) -> dict:
    """The flagged message is the last final answer: Codex also writes
    ``commentary`` progress notes mid-task, and a ``$bad`` delivered mid-task
    is about the answer the user read, not the latest note."""
    idx = None
    for final_only in (True, False):
        idx = next(
            (i for i in range(len(recs) - 1, -1, -1) if _is_answer(recs[i], final_only)), None
        )
        if idx is not None:
            break
    if idx is None:
        raise ValueError("no assistant message with text to flag")
    cut = idx
    # Reasoning (and UI events) emitted for the same response sit just before it.
    while cut > 0 and (
        _item(recs[cut - 1]).get("type") == "reasoning" or recs[cut - 1].get("type") == "event_msg"
    ):
        cut -= 1
    model = next(
        (
            (r.get("payload") or {}).get("model")
            for r in reversed(recs[:idx])
            if r.get("type") == "turn_context"
        ),
        "",
    )
    return {"cut": cut, "flagged": _message_text(_item(recs[idx])), "model": model or ""}


def mid_turn(recs: list[dict]) -> bool:
    """Is the agent still working (a task started and hasn't completed)?"""
    started = completed = -1
    for i, r in enumerate(recs):
        if r.get("type") == "event_msg":
            kind = (r.get("payload") or {}).get("type")
            if kind == "task_started":
                started = i
            elif kind in ("task_complete", "turn_aborted"):
                completed = i
    return started > completed


def build_case(
    transcript: Path,
    complaint: str,
    cwd: str = "",
    session_id: str = "",
    upto: int | None = None,
) -> tuple[dict, list[dict]]:
    recs = read_jsonl(transcript)
    if upto is not None:
        recs = recs[:upto]
    sp = split(recs)
    context = recs[: sp["cut"]]
    st = steps(context)
    meta_rec = next((r.get("payload") or {} for r in recs if r.get("type") == "session_meta"), {})
    meta = {
        "harness": HARNESS,
        "complaint": complaint.strip(),
        "flagged": sp["flagged"],
        "model": sp["model"],
        "cwd": cwd or meta_rec.get("cwd", ""),
        "sessionId": session_id or meta_rec.get("id", ""),
        "harnessVersion": meta_rec.get("cli_version", ""),
        "git": {
            "sha": (meta_rec.get("git") or {}).get("commit_hash"),
            "branch": (meta_rec.get("git") or {}).get("branch"),
        },
        "stats": {
            "userMessages": sum(s["kind"] == "user" for s in st),
            "toolCalls": sum(s["kind"] == "tool_call" for s in st),
            "contextRecords": len(context),
            "afterToolResult": bool(st) and st[-1]["kind"] == "tool_result",
        },
    }
    return meta, context


def _user_config() -> dict:
    """Top-level string settings from config.toml (enough for the model and
    effort, and it keeps the hook working on Python 3.9, which lacks tomllib)."""
    try:
        text = (codex_home() / "config.toml").read_text()
    except OSError:
        return {}
    head = re.split(r"(?m)^\s*\[", text, maxsplit=1)[0]
    return dict(re.findall(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', head))


def default_model() -> str:
    return _user_config().get("model", "")


def _probe_comp_hash(model: str) -> str | None:
    """Ask Codex itself: one tiny ``codex exec`` on ``model`` in a throwaway home,
    then read the compaction hash it recorded for that model."""
    with tempfile.TemporaryDirectory(prefix="slash-bad-probe-") as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        for name in ("auth.json", "models_cache.json"):
            if (codex_home() / name).exists():
                shutil.copy(codex_home() / name, home / name)
        (home / "config.toml").write_text(f'model = "{model}"\nmodel_reasoning_effort = "low"\n')
        try:
            subprocess.run(
                ["codex", "exec", "--skip-git-repo-check", "-m", model, "Reply with the word ok."],
                cwd=tmp,
                env={**os.environ, "CODEX_HOME": str(home), REPLAY_ENV: "1"},
                capture_output=True,
                timeout=300,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        for rollout in (home / "sessions").rglob("*.jsonl"):
            for r in read_jsonl(rollout):
                if r.get("type") == "turn_context" and (r.get("payload") or {}).get("comp_hash"):
                    return r["payload"]["comp_hash"]
    return None


def _comp_hash(model: str) -> str | None:
    """The compaction hash Codex expects for ``model``. A mismatch with the one
    in the rollout makes Codex compact the whole history before replying, so the
    new model would never see the real conversation."""
    try:
        cache = json.loads((codex_home() / "models_cache.json").read_text())
        found = next(
            (m.get("comp_hash") for m in cache.get("models", []) if m.get("slug") == model), None
        )
        if found:
            return found
    except (OSError, json.JSONDecodeError):
        pass
    known_path = HOME / "codex-comp-hash.json"
    try:
        known = json.loads(known_path.read_text())
    except (OSError, json.JSONDecodeError):
        known = {}
    if model not in known:
        probed = _probe_comp_hash(model)
        if probed is None:
            return None
        known[model] = probed
        known_path.parent.mkdir(parents=True, exist_ok=True)
        known_path.write_text(json.dumps(known, indent=2))
    return known[model]


def prepare_rollout(context: list[dict], old_id: str, new_id: str, model: str) -> list[dict]:
    """The context, re-identified and rewritten to claim ``model`` throughout."""
    comp = _comp_hash(model)
    out = []
    for r in context:
        if r.get("type") == "token_usage_record" or _item(r).get("type") == "reasoning":
            continue
        text = json.dumps(r)
        # Only a real id is safe to substitute textually; a short one would hit prose.
        r = json.loads(text.replace(old_id, new_id) if len(old_id) >= 32 else text)
        p = r.get("payload") or {}
        if r.get("type") == "turn_context":
            p["model"] = model
            if comp:
                p["comp_hash"] = comp
            settings = (p.get("collaboration_mode") or {}).get("settings")
            if isinstance(settings, dict):
                settings["model"] = model
        elif r.get("type") == "world_state" and isinstance(p.get("state"), dict):
            p["state"]["model"] = model
        elif r.get("type") == "session_meta":
            p["id"] = new_id
        out.append(r)
    return out


def replay(case: Case, model: str, max_seconds: int = CODEX_TIMEOUT) -> dict:
    model = default_model() if model in ("latest", "") else model
    tmp = Path(tempfile.mkdtemp(prefix="slash-bad-codex-"))
    home = tmp / "home"
    home.mkdir()
    for name in ("auth.json", "models_cache.json", "AGENTS.md"):
        src = codex_home() / name
        if src.exists():
            shutil.copy(src, home / name)
    # Same skills as the original session, or Codex re-sends a different skills list.
    if (codex_home() / "skills").is_dir():
        (home / "skills").symlink_to(codex_home() / "skills")
    effort = _user_config().get("model_reasoning_effort")
    (home / "config.toml").write_text(
        f'model = "{model}"\n' + (f'model_reasoning_effort = "{effort}"\n' if effort else "")
    )
    deny = tmp / "deny.json"
    deny.write_text(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": DENY_REASON,
                }
            }
        )
    )
    (home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [{"hooks": [{"type": "command", "command": f"cat '{deny}'"}]}]
                }
            }
        )
    )
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    day = home / "sessions" / now.strftime("%Y/%m/%d")
    day.mkdir(parents=True)
    rollout = day / f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{new_id}.jsonl"
    planted = prepare_rollout(case.context, case.meta.get("sessionId", ""), new_id, model)
    rollout.write_text("".join(json.dumps(r) + "\n" for r in planted))
    orig = case.meta.get("cwd") or ""
    workdir = orig if orig and Path(orig).is_dir() else str(tmp)
    cmd = [
        "codex",
        "exec",
        "resume",
        new_id,
        "",
        "-m",
        model,
        "--skip-git-repo-check",
        "--dangerously-bypass-hook-trust",
        "-c",
        'sandbox_mode="read-only"',
        "--json",
    ]
    env = {**os.environ, "CODEX_HOME": str(home), REPLAY_ENV: "1"}
    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=max_seconds,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            rc, err = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired:
            rc, err = 124, f"codex timed out after {max_seconds}s"
        written = next(iter(sorted((home / "sessions").rglob(f"*{new_id}*.jsonl"))), rollout)
        new = read_jsonl(written)[len(planted) :]
    finally:
        if not os.environ.get("SLASH_BAD_KEEP_TMP"):
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"slash-bad: kept {tmp}", file=sys.stderr)
    result = parse_replay(new)
    result.update(
        {
            "kind": "replay",
            "harness": HARNESS,
            "replayVersion": REPLAY_VERSION,
            "requestedModel": model,
            "model": result.get("model") or model,
            "costUsd": None,
            "returncode": rc,
            "addedEmptyPrompt": True,
        }
    )
    if rc != 0 or not (result["text"] or result["toolAttempts"]):
        result["error"] = (err or "no output")[-600:]
    elif problem := unfaithful(new):
        result["error"] = f"replay not faithful: {problem}"
    return result


def unfaithful(new: list[dict]) -> str | None:
    """Why the new model didn't see the stored conversation as it was, if it didn't."""
    for r in new:
        if r.get("type") == "compacted":
            return "Codex compacted the history before the new model replied"
        item = _item(r)
        if item.get("type") == "message" and "<model_switch>" in _message_text(item):
            return "Codex told the model it had been switched"
    return None


def parse_replay(new: list[dict]) -> dict:
    first_text: list[str] = []
    first_tools: list[str] = []
    all_text: list[str] = []
    attempts: list[str] = []
    seen_output = False
    model = ""
    for r in new:
        if r.get("type") == "turn_context":
            model = (r.get("payload") or {}).get("model") or model
        for s in steps([r]):
            if s["kind"] == "assistant":
                all_text.append(s["text"])
                if not seen_output:
                    first_text.append(s["text"])
            elif s["kind"] == "tool_call":
                attempts.append(s["text"])
                if not seen_output:
                    first_tools.append(s["text"])
            elif s["kind"] == "tool_result":
                seen_output = True
    return {
        "model": model,
        "first": "\n\n".join(first_text),
        "firstToolCalls": first_tools,
        "text": "\n\n".join(all_text),
        "toolAttempts": attempts,
    }
