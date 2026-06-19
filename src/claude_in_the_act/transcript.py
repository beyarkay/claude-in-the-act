"""Shared helpers for reading Claude Code transcript records + our bundle jsonl."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# The record type we append to a captured bundle (the feedback/good/bad triple).
ANNOTATION_TYPE = "bad_annotation"

# Prepopulated value of the bad_annotation `goodResponse` field. Replace it (in
# the bundle jsonl's last line) with the response you actually wanted; the judge
# treats it as "no ideal response" until you do.
GOOD_RESPONSE_PLACEHOLDER = (
    "<<< REPLACE THIS with the response you wanted — then the judge will use it >>>"
)

_LINENO = re.compile(r"^\s*\d+\t")


def load_bundle(path: str | Path) -> list[dict]:
    """Parse a bundle jsonl into records, skipping blank or malformed lines (one
    corrupt line shouldn't sink the whole bundle)."""
    recs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def dump_jsonl(recs: list[dict]) -> str:
    """Serialise records as jsonl (the inverse of ``load_bundle``)."""
    return "".join(json.dumps(r) + "\n" for r in recs)


def annotation(recs: list[dict]) -> dict:
    """The bundle's bad_annotation record (last one wins), or {} if absent."""
    return next((r for r in reversed(recs) if r.get("type") == ANNOTATION_TYPE), {})


def denumber(rendered: str) -> str:
    """Recover raw file content from Read's ``<lineno>\\t<line>`` rendering."""
    return "\n".join(_LINENO.sub("", line) for line in rendered.split("\n"))


def assistant_text(rec: dict) -> str | None:
    """Joined text of an assistant record, or None if it isn't a text-bearing
    assistant turn (thinking-only / tool_use-only turns return None)."""
    if rec.get("type") != "assistant":
        return None
    content = rec.get("message", {}).get("content")
    if not isinstance(content, list):
        return None
    text = "\n".join(
        b.get("text", "") for b in content if b.get("type") == "text"
    ).strip()
    return text or None


def user_text(rec: dict) -> str | None:
    """The string content of a *real* (user-authored) prompt, or None.

    Two kinds of non-prompt ``user`` records must never be mistaken for the
    prompt to replay:

    * Tool results and the ``[Request interrupted by user]`` injection — these
      carry *list* content, so the ``isinstance(..., str)`` check drops them.
    * Harness-injected string-content turns — most notably background-task
      completion notifications (``<task-notification>…``), which Claude Code
      records as ``type:"user"`` with ``promptSource:"system"`` /
      ``origin.kind:"task-notification"``. Without this guard ``split_transcript``
      would pick the notification as the prompt and replay the model against a
      synthetic message (with a stale ``tool-use-id``) instead of your words.

    Detection keys on the structured fields, not the text — a genuinely *typed*
    or *queued* message may itself start with ``<`` (e.g. a queued turn carries a
    ``<system-reminder>Message sent at …</system-reminder>`` prefix), so a
    prefix match would wrongly drop real prompts.
    """
    if rec.get("type") != "user":
        return None
    if rec.get("promptSource") == "system":
        return None
    if (rec.get("origin") or {}).get("kind") == "task-notification":
        return None
    content = rec.get("message", {}).get("content")
    return content if isinstance(content, str) else None


def split_transcript(recs: list[dict]) -> dict:
    """Split records into replay inputs.

    Drops any ``bad_annotation``, finds the last string user turn (the prompt to
    replay), the original assistant response after it (all its text turns
    joined), the context before it, and the model id. Raises ValueError if there
    is no user turn.
    """
    recs = [r for r in recs if r.get("type") != ANNOTATION_TYPE]
    user_idxs = [i for i, r in enumerate(recs) if user_text(r) is not None]
    if not user_idxs:
        raise ValueError("no string user turn found to replay")
    last_user = user_idxs[-1]

    # All assistant text after the prompt (a turn spans several records;
    # thinking-only / tool_use-only ones carry no text and are skipped).
    original = "\n\n".join(
        t for r in recs[last_user + 1 :] if (t := assistant_text(r))
    ).strip()
    model = next(
        (
            r["message"]["model"]
            for r in reversed(recs)
            if r.get("type") == "assistant" and r.get("message", {}).get("model")
        ),
        "",
    )
    return {
        "prompt": user_text(recs[last_user]) or "",
        "original": original,
        "context": recs[:last_user],
        "model": model,
    }


def final_response(recs: list[dict]) -> str:
    """The final assistant response text (the 'bad' response). Empty if none."""
    try:
        return split_transcript(recs)["original"]
    except ValueError:
        return ""


def _tool_use_count(rec: dict) -> int:
    """Number of tool_use blocks in an assistant record (0 for anything else)."""
    content = rec.get("message", {}).get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
    )


def _parse_ts(value: object) -> datetime | None:
    """Parse a transcript ISO timestamp (``…Z``), or None if absent/malformed."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def bundle_metrics(recs: list[dict]) -> dict:
    """Nice-to-look-at counts derived from a captured transcript's records.

    Best-effort and side-effect-free: each field is computed independently, and
    a transcript that can't yield one (no timestamps, no assistant turn) just
    omits it rather than raising. The ``bad_annotation`` record itself is
    excluded so re-deriving metrics from an already-annotated bundle is stable.
    """
    body = [r for r in recs if r.get("type") != ANNOTATION_TYPE]
    prompt_idxs = [i for i, r in enumerate(body) if user_text(r) is not None]
    assistants = [r for r in body if r.get("type") == "assistant"]

    # The flagged turn is the assistant work after the last real user prompt.
    flagged = body[prompt_idxs[-1] + 1 :] if prompt_idxs else []

    metrics: dict = {
        "records": len(body),
        "userPrompts": len(prompt_idxs),
        "assistantRecords": len(assistants),
        "toolUses": sum(_tool_use_count(r) for r in assistants),
        "flaggedResponseChars": len(final_response(recs)),
        "flaggedTurnToolUses": sum(_tool_use_count(r) for r in flagged),
    }
    model = next(
        (
            r["message"]["model"]
            for r in reversed(assistants)
            if r.get("message", {}).get("model")
        ),
        "",
    )
    if model:
        metrics["model"] = model
    stamps = [t for r in body if (t := _parse_ts(r.get("timestamp")))]
    if len(stamps) >= 2:
        metrics["sessionSpanMinutes"] = round(
            (max(stamps) - min(stamps)).total_seconds() / 60
        )
    return metrics
