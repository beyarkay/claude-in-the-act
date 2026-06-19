"""Tests for transcript record classification — chiefly that harness-injected
``user`` records (task-notifications) are not mistaken for the prompt to replay.
"""

from __future__ import annotations

from claude_in_the_act.transcript import (
    bundle_metrics,
    final_response,
    split_transcript,
    user_text,
)


def _typed(text: str) -> dict:
    return {
        "type": "user",
        "promptSource": "typed",
        "message": {"role": "user", "content": text},
    }


def _queued(text: str) -> dict:
    # Queued turns are real user messages; they often carry a system-reminder
    # timestamp prefix and so start with "<" — must still count as a prompt.
    return {
        "type": "user",
        "promptSource": "queued",
        "message": {"role": "user", "content": text},
    }


def _task_notification(
    text: str = "<task-notification>\n<task-id>abc</task-id>",
) -> dict:
    return {
        "type": "user",
        "promptSource": "system",
        "origin": {"kind": "task-notification"},
        "message": {"role": "user", "content": text},
    }


def _assistant(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": text}],
        },
    }


def _tool_result() -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "tool_result"}]},
    }


def test_user_text_accepts_typed_and_queued():
    assert user_text(_typed("hello")) == "hello"
    assert (
        user_text(_queued("<system-reminder>Message sent at …</system-reminder> hi"))
        == "<system-reminder>Message sent at …</system-reminder> hi"
    )


def test_user_text_rejects_task_notification():
    # Both the promptSource and the origin.kind signal must independently reject.
    assert user_text(_task_notification()) is None
    assert (
        user_text(
            {
                "type": "user",
                "promptSource": "system",
                "message": {"role": "user", "content": "x"},
            }
        )
        is None
    )
    assert (
        user_text(
            {
                "type": "user",
                "origin": {"kind": "task-notification"},
                "message": {"role": "user", "content": "x"},
            }
        )
        is None
    )


def test_user_text_rejects_tool_result():
    assert user_text(_tool_result()) is None


def test_split_skips_trailing_task_notification():
    """A task-notification after the real prompt must not become the prompt, and
    the assistant turn that followed it is still the response to evaluate."""
    recs = [
        _typed("kick off the sweep"),
        _assistant("launching the sweep now"),
        _task_notification(),
        _assistant("Go-ahead received and smoke green, so launching the full run"),
    ]
    split = split_transcript(recs)
    assert split["prompt"] == "kick off the sweep"
    # Both assistant turns after the real prompt are joined into the response.
    assert "launching the sweep now" in split["original"]
    assert "Go-ahead received" in split["original"]
    assert final_response(recs) == split["original"]


def test_split_prefers_last_real_prompt_over_notification():
    recs = [
        _typed("first question"),
        _assistant("first answer"),
        _queued("second question"),
        _task_notification(),
        _assistant("answer after the notification"),
    ]
    split = split_transcript(recs)
    assert split["prompt"] == "second question"
    assert "first answer" not in split["original"]
    assert "answer after the notification" in split["original"]


def _assistant_tool(name: str = "Bash") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "tool_use", "name": name, "input": {}}],
        },
    }


def test_bundle_metrics_counts():
    recs = [
        _typed("do a thing"),
        _assistant("sure, working on it"),
        _assistant_tool(),  # tool use, inside the flagged turn
        _assistant("done — here is the result"),
        {"type": "bad_annotation", "feedback": "x"},  # excluded from every count
    ]
    m = bundle_metrics(recs)
    assert m["records"] == 4  # annotation not counted
    assert m["userPrompts"] == 1
    assert m["assistantRecords"] == 3
    assert m["toolUses"] == 1
    assert m["flaggedTurnToolUses"] == 1
    assert m["flaggedResponseChars"] == len(final_response(recs))
    assert m["model"] == "claude-x"
    assert "sessionSpanMinutes" not in m  # no timestamps in these records


def test_bundle_metrics_span_and_no_prompt():
    def at(ts: str, text: str) -> dict:
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
            "timestamp": ts,
        }

    recs = [at("2026-06-15T10:00:00Z", "orphan"), at("2026-06-15T10:30:00Z", "later")]
    m = bundle_metrics(recs)
    assert m["userPrompts"] == 0
    assert m["sessionSpanMinutes"] == 30
    assert m["flaggedResponseChars"] == 0  # no real prompt -> no flagged response
