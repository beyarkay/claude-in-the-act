"""Tests for the replay path.

Offline tests cover the pure transcript-splitting logic. The ``e2e`` test does
the full capture → destroy-original → replay round-trip against the real
``claude`` CLI; it is skipped by default (run with ``-m e2e``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from claude_in_the_act import capture, recreate


def test_slug_matches_claude_code_scheme():
    assert recreate._slug("/private/tmp/a.b-c") == "-private-tmp-a-b-c"


def test_split_transcript_picks_last_turn_and_drops_annotation():
    recs = [
        {"type": "user", "message": {"content": "remember 7"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "ok"}],
            },
        },
        {"type": "user", "message": {"content": "what number?"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "7"}],
            },
        },
        {"type": "bad_annotation", "critique": "bad"},
    ]
    out = recreate.split_transcript(recs)
    assert out["prompt"] == "what number?"
    assert out["original"] == "7"
    assert out["model"] == "claude-opus-4-8"
    assert len(out["context"]) == 2  # only the first remember/ok pair
    assert out["context"][0]["message"]["content"] == "remember 7"


def test_split_transcript_raises_without_user_turn():
    with pytest.raises(ValueError):
        recreate.split_transcript([{"type": "assistant", "message": {"content": []}}])


def test_split_transcript_skips_thinking_and_tool_only_turns():
    """A turn spans several records; thinking-only / tool_use-only ones have no
    text and must not be mistaken for the response (regression)."""
    recs = [
        {"type": "user", "message": {"content": "summarise the repo"}},
        {
            "type": "assistant",
            "message": {
                "model": "m",
                "content": [{"type": "thinking", "thinking": "hmm"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "model": "m",
                "content": [{"type": "text", "text": "Let me look."}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "model": "m",
                "content": [{"type": "tool_use", "name": "Read", "input": {}}],
            },
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "x"}]},
        },
        {
            "type": "assistant",
            "message": {
                "model": "m",
                "content": [{"type": "text", "text": "Here is the summary."}],
            },
        },
    ]
    out = recreate.split_transcript(recs)
    assert (
        out["prompt"] == "summarise the repo"
    )  # tool_result (list) isn't a user prompt
    assert (
        out["original"] == "Let me look.\n\nHere is the summary."
    )  # both text turns, joined


@pytest.mark.e2e
def test_capture_then_replay_roundtrip_is_self_contained(tmp_path):
    """Full loop: real 2-turn session → capture → delete original dir + session
    → replay reproduces the context-dependent codeword from the stored bundle
    alone."""
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")

    work = Path(tempfile.mkdtemp()).resolve()
    codeword = f"CODA-{uuid.uuid4().hex[:6].upper()}"
    sid = str(uuid.uuid4())

    def claude(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["claude", *args], cwd=work, capture_output=True, text=True
        )

    claude(
        "--session-id",
        sid,
        "-p",
        f"Remember the codeword {codeword}. Reply: ok.",
        "--output-format",
        "json",
        "--allowed-tools",
        "",
    )
    r = claude(
        "--resume",
        sid,
        "-p",
        "What was the codeword? Reply with ONLY the codeword.",
        "--output-format",
        "json",
        "--allowed-tools",
        "",
    )
    original = json.loads(r.stdout)["result"]
    assert codeword in original

    transcript = (
        Path.home()
        / ".claude"
        / "projects"
        / recreate._slug(str(work))
        / f"{sid}.jsonl"
    )
    assert transcript.is_file()

    data = tmp_path / "data"
    code, _ = capture.process(
        {
            "prompt": "BAD: just parrots the codeword",
            "transcript_path": str(transcript),
            "session_id": sid,
            "cwd": str(work),
        },
        data,
    )
    assert code == 2
    stored = next(data.glob("*.jsonl"))

    # Destroy the original working tree AND the original session transcript:
    # replay must work from the stored bundle alone.
    shutil.rmtree(work, ignore_errors=True)
    transcript.unlink()

    out = recreate.recreate(stored)
    assert codeword in out["recreated"]
    assert out["recreated"] == out["original"]


@pytest.mark.e2e
def test_tool_replay_read_edit_write_codebase_free(tmp_path):
    """A turn that Reads + Edits + Writes, captured, then replayed AFTER the repo
    is deleted — every tool served from the recording (seeded files + Bash hook),
    nothing on the real filesystem."""
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")

    work = Path(tempfile.mkdtemp()).resolve()
    (work / "config.py").write_text("DEBUG = False\nNAME = 'app'\nVERSION = '1.0'\n")
    sid = str(uuid.uuid4())
    r = subprocess.run(
        [
            "claude",
            "--session-id",
            sid,
            "-p",
            "Read config.py, use Edit to change DEBUG from False to True, "
            "and use Write to create flag.txt containing just: enabled",
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--allowed-tools",
            "Read",
            "Edit",
            "Write",
            "Bash",
        ],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    transcript = (
        Path.home()
        / ".claude"
        / "projects"
        / recreate._slug(str(work))
        / f"{sid}.jsonl"
    )
    assert transcript.is_file()

    from claude_in_the_act.replay import build_recording

    recs = [
        json.loads(line) for line in transcript.read_text().splitlines() if line.strip()
    ]
    tools = {e["tool_name"] for e in build_recording(recs)}
    assert {"Read", "Edit", "Write"} <= tools, f"recorded tools: {tools}"

    data = tmp_path / "data"
    capture.process(
        {
            "prompt": "BAD: x",
            "transcript_path": str(transcript),
            "session_id": sid,
            "cwd": str(work),
        },
        data,
    )
    stored = next(data.glob("*.jsonl"))

    # destroy the codebase + original session
    shutil.rmtree(work, ignore_errors=True)
    transcript.unlink()

    out = recreate.recreate_with_tools(stored)
    assert out["returncode"] == 0, out["stderr"]
    assert out["recreated"].strip()  # produced a response with the repo gone
    assert "True" in out["recreated"] or "DEBUG" in out["recreated"]
