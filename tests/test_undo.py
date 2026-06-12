"""Unit tests for the pre-turn undo: reverse the bad turn's Write/Edit, flag Bash."""

from __future__ import annotations

from claude_in_the_act import recreate


def _user(cwd: str, text: str = "do it") -> dict:
    return {"type": "user", "cwd": cwd, "message": {"content": text}}


def _tool(name: str, inp: dict, cwd: str) -> dict:
    return {
        "type": "assistant",
        "cwd": cwd,
        "message": {
            "content": [{"type": "tool_use", "id": name, "name": name, "input": inp}]
        },
    }


def test_mutating_bash_heuristic():
    assert recreate._mutating_bash("rm -rf build")
    assert recreate._mutating_bash("echo hi > out.txt")
    assert recreate._mutating_bash("sed -i 's/a/b/' f.py")
    assert recreate._mutating_bash("cat a >> b")
    assert not recreate._mutating_bash("git log --oneline -20")
    assert not recreate._mutating_bash("cat config.py")
    assert not recreate._mutating_bash("ls -la 2>&1")
    assert not recreate._mutating_bash("rg DEBUG .")


def test_undo_deletes_new_file_write(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "new.txt"
    f.write_text("created by the bad turn\n")  # post-turn state
    recs = [
        _user(str(repo), "make a file"),
        _tool(
            "Write",
            {"file_path": str(f), "content": "created by the bad turn\n"},
            str(repo),
        ),
    ]
    out = recreate.undo_last_turn(recs, repo)
    assert not f.exists()  # the new file is removed → pre-turn state
    assert out["preTurnExact"] is True
    assert any("delete" in u for u in out["undone"])


def test_undo_reverses_edit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "config.py"
    f.write_text("DEBUG = True\nNAME = 'app'\n")  # post-turn (already edited)
    recs = [
        _user(str(repo), "flip DEBUG"),
        _tool(
            "Edit",
            {
                "file_path": str(f),
                "old_string": "DEBUG = False",
                "new_string": "DEBUG = True",
            },
            str(repo),
        ),
    ]
    recreate.undo_last_turn(recs, repo)
    assert f.read_text() == "DEBUG = False\nNAME = 'app'\n"  # reversed to pre-turn


def test_undo_flags_mutating_bash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    recs = [
        _user(str(repo), "clean up"),
        _tool("Bash", {"command": "rm -rf build && echo done"}, str(repo)),
    ]
    out = recreate.undo_last_turn(recs, repo)
    assert out["preTurnExact"] is False
    assert out["bashMutations"]


def test_undo_no_user_turn_is_exact_noop():
    out = recreate.undo_last_turn(
        [{"type": "assistant", "message": {"content": []}}], None
    )
    assert out == {"preTurnExact": True, "undone": [], "bashMutations": []}
