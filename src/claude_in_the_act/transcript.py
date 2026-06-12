"""Shared helpers for reading Claude Code transcript records."""

from __future__ import annotations


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
    """The string content of a real user prompt, or None.

    Tool results and the ``[Request interrupted by user]`` injection are
    list-content user records and return None — so they're never mistaken for a
    prompt."""
    if rec.get("type") != "user":
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
    recs = [r for r in recs if r.get("type") != "bad_annotation"]
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
