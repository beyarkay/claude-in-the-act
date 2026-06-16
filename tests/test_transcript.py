"""Tests for transcript record classification — chiefly that harness-injected
``user`` records (task-notifications) are not mistaken for the prompt to replay.
"""

from __future__ import annotations

from claude_in_the_act.transcript import (
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
