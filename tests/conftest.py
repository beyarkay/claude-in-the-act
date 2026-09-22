import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own case store."""
    import slash_bad
    import slash_bad.store as store

    home = tmp_path / "slash-bad-home"
    monkeypatch.setattr(slash_bad, "HOME", home)
    monkeypatch.setattr(slash_bad, "CASES_DIR", home / "cases")
    monkeypatch.setattr(store, "CASES_DIR", home / "cases")
    monkeypatch.delenv("SLASH_BAD_REPLAY", raising=False)
    return home


def cc_user(text, sid="s1", **extra):
    return {
        "type": "user",
        "sessionId": sid,
        "cwd": "/Users/alice/proj",
        "message": {"role": "user", "content": text},
        **extra,
    }


def cc_assistant(blocks, msg_id, model="claude-opus-5", sid="s1"):
    return {
        "type": "assistant",
        "sessionId": sid,
        "message": {"id": msg_id, "role": "assistant", "model": model, "content": blocks},
    }


def cc_tool_result(tool_id, content, sid="s1"):
    return {
        "type": "user",
        "sessionId": sid,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
        },
    }


@pytest.fixture
def cc_transcript(tmp_path) -> Path:
    """user → assistant(tool_use) → tool_result → assistant(thinking, text)."""
    recs = [
        {"type": "queue-operation"},
        cc_user("is the server up?"),
        cc_assistant(
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "curl localhost:8000"},
                }
            ],
            "m1",
        ),
        cc_tool_result("t1", "curl: (7) Failed to connect"),
        {"type": "attachment", "sessionId": "s1"},
        cc_assistant([{"type": "thinking", "thinking": "hmm"}], "m2"),
        cc_assistant(
            [{"type": "text", "text": "Great question! The server is running fine."}], "m2"
        ),
        {"type": "system", "sessionId": "s1"},
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path
