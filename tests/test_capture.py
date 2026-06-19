"""Offline unit tests for the capture path (no `claude` calls)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from claude_in_the_act import capture


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_transcript(path: Path) -> None:
    recs = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hello"}],
            },
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _dirty_repo(repo: Path) -> None:
    """A repo with one tracked edit, one untracked file, one ignored file."""
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("build/\n")
    (repo / "main.py").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    (repo / "main.py").write_text("base\nmodified\n")  # tracked edit
    (repo / "extra.py").write_text("new untracked\n")  # untracked
    (repo / "build").mkdir()
    (repo / "build" / "x.bin").write_text("junk\n")  # ignored


def test_passthrough_on_non_bad_prompt(tmp_path):
    code, msg = capture.process({"prompt": "just a normal prompt"}, tmp_path)
    assert code == 0
    assert msg is None
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_missing_transcript_blocks_but_saves_nothing(tmp_path):
    code, msg = capture.process(
        {"prompt": "BAD: x", "transcript_path": "/no/such/file", "session_id": "s"},
        tmp_path,
    )
    assert code == 2
    assert "could not locate" in msg
    assert list(tmp_path.glob("*.jsonl")) == []


def test_capture_writes_bundle_with_git_and_patch(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _make_transcript(transcript)
    repo = tmp_path / "repo"
    _dirty_repo(repo)
    data = tmp_path / "data"

    code, msg = capture.process(
        {
            "prompt": "BAD: parrots the codeword",
            "transcript_path": str(transcript),
            "session_id": "sess1",
            "cwd": str(repo),
        },
        data,
    )

    assert code == 2
    assert "captured" in msg
    assert "goodResponse" in msg  # points the user at the slot to fill in
    assert "parrots the codeword" in msg  # the feedback snippet
    assert "session:" in msg  # the nice-to-look-at metrics line

    jsonls = list(data.glob("*.jsonl"))
    patches = list(data.glob("*.patch"))
    assert len(jsonls) == 1
    assert len(patches) == 1

    lines = jsonls[0].read_text().splitlines()
    assert len(lines) == 3  # 2 verbatim transcript records + 1 annotation
    ann = json.loads(lines[-1])
    assert ann["type"] == "bad_annotation"
    assert ann["feedback"] == "parrots the codeword"
    assert ann["goodResponse"].startswith("<<<")  # placeholder to fill in later
    assert "badResponse" in ann
    assert ann["sessionId"] == "sess1"
    assert ann["git"]["sha"]
    assert ann["git"]["branch"] in ("main", "master")
    assert ann["git"]["patchFile"] == patches[0].name

    m = ann["metrics"]  # derived counts ride along in the annotation
    assert m["records"] == 2  # the 2 transcript records (annotation excluded)
    assert m["userPrompts"] == 1
    assert m["assistantRecords"] == 1
    assert m["toolUses"] == 0
    assert m["flaggedResponseChars"] == len("hello")
    assert m["model"] == "claude-opus-4-8"
    assert m["transcriptBytes"] > 0

    patch = patches[0].read_text()
    assert "main.py" in patch  # tracked edit
    assert "extra.py" in patch  # untracked file captured
    assert "build/" not in patch  # ignored files excluded


def test_capture_does_not_touch_real_index(tmp_path):
    """The throwaway-index trick must leave the repo's staging area untouched."""
    transcript = tmp_path / "t.jsonl"
    _make_transcript(transcript)
    repo = tmp_path / "repo"
    _dirty_repo(repo)
    before = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout

    capture.process(
        {
            "prompt": "BAD: x",
            "transcript_path": str(transcript),
            "session_id": "s",
            "cwd": str(repo),
        },
        tmp_path / "data",
    )

    after = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout
    assert before == after  # nothing staged/unstaged changed
