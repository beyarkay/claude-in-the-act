from pathlib import Path

from slash_bad import setup


def test_install_and_uninstall_leave_foreign_skills_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".codex").mkdir()
    mine = tmp_path / ".claude/skills/bad-list/SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("my own bad-list skill")

    written = setup.install(Path("/opt/slash-bad"))
    assert mine.read_text() == "my own bad-list skill"
    assert len(written) == 7  # 4 codex + 3 claude (bad-list was the user's)
    compare = (tmp_path / ".claude/skills/bad-compare/SKILL.md").read_text()
    assert '"/opt/slash-bad/bin/slash-bad" compare $ARGUMENTS' in compare
    assert "disable-model-invocation: true" in compare

    setup.uninstall()
    assert mine.exists()
    assert not (tmp_path / ".claude/skills/bad").exists()
    assert not (tmp_path / ".codex/skills/bad").exists()
