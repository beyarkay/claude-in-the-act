import json
import subprocess
from pathlib import Path

import pytest

from slash_bad import hook
from slash_bad.store import all_cases

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("/bad too wordy", ("bad", "too wordy")),
        ("/slash-bad:bad too wordy", ("bad", "too wordy")),
        ("$bad too wordy", ("bad", "too wordy")),
        ("BAD: too wordy", ("bad", "too wordy")),
        ("/bad-list", ("bad-list", "")),
        ("/bad", ("bad", "")),
        ("/badger please", None),
        ("this was /bad", None),
        ("/bad-compare latest", None),
    ],
)
def test_parse(prompt, expected):
    assert hook.parse(prompt) == expected


def test_capture_saves_case_and_blocks(cc_transcript):
    code, msg = hook.handle(
        {
            "prompt": "/bad it never checked the server",
            "transcript_path": str(cc_transcript),
            "session_id": "s1",
            "cwd": "",
        }
    )
    assert code == 2
    assert "saved case" in msg
    (case,) = all_cases()
    assert case.complaint == "it never checked the server"
    assert case.meta["flagged"].startswith("Great question!")


def test_empty_complaint_blocks_without_saving(cc_transcript):
    code, msg = hook.handle({"prompt": "/bad", "transcript_path": str(cc_transcript)})
    assert code == 2 and "say what went wrong" in msg
    assert all_cases() == []


def test_passthrough_and_replay_guard(monkeypatch, cc_transcript):
    assert hook.handle({"prompt": "hello"}) == (0, "")
    monkeypatch.setenv("SLASH_BAD_REPLAY", "1")
    assert hook.handle({"prompt": "/bad x", "transcript_path": str(cc_transcript)}) == (0, "")


def test_detect_harness():
    assert (
        hook.detect_harness({"transcript_path": "/x/.claude/projects/a/b.jsonl"}) == "claude-code"
    )
    assert hook.detect_harness({"transcript_path": "/x/.codex/sessions/rollout-1.jsonl"}) == "codex"
    assert hook.detect_harness({"turn_id": "t"}) == "codex"


def test_shell_gate_skips_python_for_ordinary_prompts(tmp_path):
    gate = ROOT / "bin" / "slash-bad-hook"
    ordinary = subprocess.run(
        [gate], input=json.dumps({"prompt": "hello"}), capture_output=True, text=True, check=False
    )
    assert ordinary.returncode == 0
    flagged = subprocess.run(
        [gate],
        input=json.dumps({"prompt": "/bad x", "transcript_path": str(tmp_path / "missing.jsonl")}),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "SLASH_BAD_HOME": str(tmp_path / "h")},
    )
    assert flagged.returncode == 2
    assert "couldn't find the session transcript" in flagged.stderr


def test_queued_bad_is_captured_once_at_the_moment_it_was_typed(tmp_path):
    from conftest import cc_assistant, cc_user

    recs = [
        cc_user("summarise"),
        cc_assistant([{"type": "text", "text": "You're right, here it is."}], "m1"),
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "t1",
            "content": "/bad don't say you're right",
        },
        cc_assistant([{"type": "text", "text": "Later message."}], "m2"),
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    payload = {"hook_event_name": "Stop", "transcript_path": str(path), "session_id": "s1"}
    assert "saved case" in hook.handle_stop(payload)
    (case,) = all_cases()
    assert case.meta["flagged"] == "You're right, here it is."
    assert case.complaint == "don't say you're right"
    assert hook.handle_stop(payload) == ""
