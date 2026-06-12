"""Unit tests for the tool record-replay pure functions (no `claude` calls)."""

from __future__ import annotations

from claude_in_the_act import replay


def test_denumber_strips_read_line_numbers():
    rendered = "1\t# header\n2\t\n3\tdef f():\n4\t    return 1"
    assert replay.denumber(rendered) == "# header\n\ndef f():\n    return 1"


def _transcript_with_tools():
    return [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "a\nb\n"},
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "Read",
                        "input": {"file_path": "/x/foo.py"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": "1\thello\n",
                        "is_error": False,
                    },
                ]
            },
        },
    ]


def test_build_recording_pairs_calls_with_results_in_order():
    rec = replay.build_recording(_transcript_with_tools())
    assert [e["tool_name"] for e in rec] == ["Bash", "Read"]
    assert rec[0]["input"] == {"command": "ls"}
    assert rec[0]["content"] == "a\nb\n"
    assert rec[1]["input"] == {"file_path": "/x/foo.py"}
    assert rec[1]["content"] == "1\thello\n"


def test_match_exact_input():
    rec = replay.build_recording(_transcript_with_tools())
    assert replay.match(rec, "Bash", {"command": "ls"}) is rec[0]
    assert replay.match(rec, "Bash", {"command": "pwd"}) is None


def test_match_read_falls_back_to_basename():
    """The model passes a different absolute path on replay; basename still hits."""
    rec = replay.build_recording(_transcript_with_tools())
    hit = replay.match(rec, "Read", {"file_path": "/totally/different/dir/foo.py"})
    assert hit is rec[1]


def test_handle_pre_denies_unrecorded_call(tmp_path):
    rec = replay.build_recording(_transcript_with_tools())
    out = replay.handle_pre(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/x/missing.py"},
            "tool_use_id": "z",
        },
        rec,
        tmp_path,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_handle_pre_read_rewrites_to_temp_with_recorded_content(tmp_path):
    rec = replay.build_recording(_transcript_with_tools())
    out = replay.handle_pre(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/anywhere/foo.py"},
            "tool_use_id": "z",
        },
        rec,
        tmp_path,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    new_path = out["hookSpecificOutput"]["updatedInput"]["file_path"]
    from pathlib import Path

    assert Path(new_path).read_text() == "hello\n"  # de-numbered (keeps trailing \n)


def test_handle_pre_bash_rewrites_to_cat_recorded_stdout(tmp_path):
    rec = replay.build_recording(_transcript_with_tools())
    out = replay.handle_pre(
        {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "z"},
        rec,
        tmp_path,
    )
    cmd = out["hookSpecificOutput"]["updatedInput"]["command"]
    assert cmd.startswith("cat ")
    from pathlib import Path

    scratch = cmd.split("cat ", 1)[1].split(";")[0].strip()
    assert Path(scratch).read_text() == "a\nb\n"
