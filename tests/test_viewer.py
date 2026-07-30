"""Tests for the HTML viewer's bundle→view extraction and safe embedding."""

from __future__ import annotations

import json
import re

from claude_in_the_act import viewer
from claude_in_the_act.transcript import ANNOTATION_TYPE, dump_jsonl


def _assistant(blocks: list[dict]) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "model": "claude-x", "content": blocks},
    }


def _prompt(text: str) -> dict:
    return {
        "type": "user",
        "promptSource": "typed",
        "message": {"role": "user", "content": text},
    }


def _tool_result(text: str) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": text}],
        },
    }


def test_messages_keeps_chat_and_drops_bookkeeping():
    recs = [
        {"type": "queue-operation"},  # harness noise → dropped
        _prompt("first prompt"),
        _assistant(
            [
                {"type": "thinking", "thinking": ""},  # empty → dropped
                {"type": "thinking", "thinking": "pondering"},
                {"type": "text", "text": "answer one"},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]
        ),
        _tool_result("file listing"),
        {
            "type": "user",
            "promptSource": "system",
            "origin": {"kind": "task-notification"},
            "message": {"role": "user", "content": "<task-notification>"},
        },  # dropped
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "[Request interrupted by user]"}],
            },
        },
    ]
    msgs = viewer._messages(recs)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "interrupt"]
    kinds = [b["k"] for b in msgs[1]["blocks"]]
    assert kinds == ["thinking", "text", "tool_use"]  # empty thinking dropped
    assert msgs[1]["blocks"][2]["name"] == "Bash"
    assert msgs[2]["results"] == ["file listing"]


def test_messages_truncates_tool_noise_but_keeps_prose():
    big = "x" * 50_000
    recs = [
        _prompt("p"),
        _tool_result(big),
        _assistant([{"type": "text", "text": big}]),
    ]
    msgs = viewer._messages(recs)
    assert len(msgs[1]["results"][0]) <= viewer._TOOL_RESULT_CAP + len(viewer._CLIP)
    # prose is kept whole up to the generous text cap
    assert len(msgs[2]["blocks"][0]["text"]) <= viewer._TEXT_CAP + len(viewer._CLIP)
    assert msgs[1]["results"][0].endswith(viewer._CLIP)


def test_build_view_reads_bundle_with_annotation(tmp_path):
    recs = [
        _prompt("do the thing"),
        _assistant([{"type": "text", "text": "the bad answer"}]),
        {
            "type": ANNOTATION_TYPE,
            "feedback": "too wordy",
            "capturedAt": "20260618T122734",
            "cwd": "/home/me/myrepo",
        },
    ]
    (tmp_path / "20260618T122734-abc.jsonl").write_text(dump_jsonl(recs))
    cases = viewer.build_view(tmp_path)
    assert len(cases) == 1
    c = cases[0]
    assert c["feedback"] == "too wordy"
    assert c["repo"] == "myrepo"
    assert c["model"] == "claude-x"
    assert c["capturedAt"] == "2026-06-18 12:27"
    # the feedback is rendered verbatim as the final BAD: turn (see the viewer JS)
    assert [m["role"] for m in c["messages"]] == ["user", "assistant"]


def test_render_html_embeds_parseable_json_and_neutralises_script_close(tmp_path):
    recs = [
        _prompt("a prompt mentioning </script> in passing"),
        _assistant([{"type": "text", "text": "ok"}]),
        {
            "type": ANNOTATION_TYPE,
            "feedback": "x",
            "capturedAt": "20260101T000000",
            "cwd": "/r",
        },
    ]
    (tmp_path / "20260101T000000-z.jsonl").write_text(dump_jsonl(recs))
    html = viewer.render_html(viewer.build_view(tmp_path))
    payload = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    ).group(1)
    # the only </script> in the data block must be neutralised so the tag can't
    # close early; JSON.parse still recovers it because \/ is a valid escape.
    assert "</script>" not in payload
    data = json.loads(payload)
    assert data[0]["messages"][0]["text"].endswith("in passing")
