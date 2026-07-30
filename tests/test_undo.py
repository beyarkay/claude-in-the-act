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


def test_reconstruct_uses_worktree_off_local_repo(tmp_path):
    """reconstruct() prefers a fast detached worktree off the local repo, and
    release_worktree() cleans it up."""
    import subprocess

    def git(*a, cwd):
        subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)

    repo = tmp_path / "orig"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    git("config", "user.email", "t@t", cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    (repo / "a.txt").write_text("hello\n")
    git("add", ".", cwd=repo)
    git("commit", "-qm", "x", cwd=repo)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    dest = tmp_path / "wt"
    src = recreate.reconstruct({"cwd": str(repo), "git": {"sha": sha}}, None, dest)
    assert src == f"worktree:{repo}"
    assert (dest / "a.txt").read_text() == "hello\n"

    recreate.release_worktree(src, dest)
    wl = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True
    ).stdout
    assert str(dest) not in wl  # worktree released


# --- _mutating_bash: the gate that decides whether a replay is comparable -----
#
# It runs on the flagged turn's shell commands; a False here means the replay is
# treated as a faithful reconstruction. Precision matters both ways — a missed
# mutation silently corrupts a before/after comparison, a spurious one throws
# away a usable transcript.

NOT_MUTATIONS = [
    # scratch space tells us nothing about the repo
    "printf '%s' 'goal' > /tmp/claude-task-abc.txt",  # the status-line hook
    "modal run lpf.py::smoke_spo > /tmp/smoke_spo.log 2>&1",
    "uvx ruff check f.py >/dev/null 2>&1",
    "zsh -c 'source ~/.zshrc >/dev/null 2>&1; ports'",
    # comparison operators are not redirects
    "awk 'NR>=2 {print $1}' file.txt",
    'python3 -c "x = a -> b"',
    # plain reads
    "grep -n foo bar.py | tail -5",
    "ls -la && git status --short",
    "git -C /repo --no-pager diff --stat",
    "the package was installed already",
]

MUTATIONS = [
    "git -C /repo add sft.py && git commit -m x",
    "gh pr create --base main --head feat --title t",
    "git checkout feat-branch",
    "rm -f runs/levels-0.json",
    "chmod +x script.py",
    "./build_levels.py --indices 0 > runs/run-index0.log 2>&1",
    "uvx ruff format mod.py >/dev/null",  # rewrites the file despite the >/dev/null
    "echo hi > notes.txt",
    "pip install requests",
    "sed -i '' s/a/b/ f.py",
]


def test_mutating_bash_ignores_scratch_and_comparisons():
    for cmd in NOT_MUTATIONS:
        assert not recreate._mutating_bash(cmd), f"false positive: {cmd!r}"


def test_mutating_bash_catches_real_repo_writes():
    for cmd in MUTATIONS:
        assert recreate._mutating_bash(cmd), f"missed mutation: {cmd!r}"
