import json
from pathlib import Path

from conftest import cc_assistant, cc_user

from slash_bad import mine
from slash_bad.store import all_cases


def _session(home: Path, recs: list[dict]) -> Path:
    path = home / ".claude/projects/-Users-alice-proj/s1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({**r, "entrypoint": "cli"}) + "\n" for r in recs))
    return path


def test_scan_finds_reactions_and_stock_phrases(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    _session(
        tmp_path,
        [
            cc_user("how did it go?"),
            cc_assistant([{"type": "text", "text": "It went fine, the fabex rate is 0.3."}], "a1"),
            cc_user("wtf is fabex"),
            cc_assistant([{"type": "text", "text": "You're right, I should explain."}], "a2"),
            cc_user("ok"),
            cc_assistant([{"type": "text", "text": "Plain answer."}], "a3"),
        ],
    )
    found = mine.scan()
    assert [c["signal"] for c in found] == ["reaction", "phrase:you're right"]
    assert found[0]["reaction"] == "wtf is fabex"
    assert found[0]["prev"] == "how did it go?"


def test_non_interactive_sessions_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    path = _session(
        tmp_path,
        [cc_user("q"), cc_assistant([{"type": "text", "text": "You're right."}], "a1")],
    )
    path.write_text(path.read_text().replace('"entrypoint": "cli"', '"entrypoint": "sdk-cli"'))
    assert mine.scan() == []


def test_select_caps_phrase_finds_but_not_reactions():
    def cand(i, signal, mode="praise for the user"):
        return {
            "key": str(i),
            "signal": signal,
            "verdict": {"failure": True, "confidence": 0.9, "mode": mode},
        }

    judged = [cand(i, "phrase:you're right") for i in range(10)]
    judged += [cand(100 + i, "reaction") for i in range(8)]
    judged.append(
        {"key": "low", "signal": "reaction", "verdict": {"failure": True, "confidence": 0.5}}
    )
    chosen = mine.select(judged)
    assert sum(c["signal"] == "reaction" for c in chosen) == 8
    assert sum(c["signal"] != "reaction" for c in chosen) == mine.PER_MODE_CAP


def test_save_cuts_at_the_flagged_message_and_tags_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    _session(
        tmp_path,
        [
            cc_user("how did it go?"),
            cc_assistant([{"type": "text", "text": "the fabex rate is 0.3"}], "a1"),
            cc_user("wtf is fabex"),
            cc_assistant([{"type": "text", "text": "sorry"}], "a2"),
        ],
    )
    (c,) = mine.scan()
    c["verdict"] = {"failure": True, "confidence": 0.9, "complaint": "never explains fabex"}
    (case,) = mine.save([c])
    assert case.meta["flagged"] == "the fabex rate is 0.3"
    assert case.meta["origin"] == "mined"
    assert mine.save([c]) == []
    assert len(all_cases()) == 1
