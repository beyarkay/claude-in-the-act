import pytest

from slash_bad import claude_code, compare, judge, report
from slash_bad.store import Case


@pytest.fixture
def case(cc_transcript):
    meta, context = claude_code.build_case(
        cc_transcript, "it claims the server is up without checking"
    )
    return Case.create(meta, context)


@pytest.fixture
def fakes(monkeypatch):
    calls = {"replay": [], "judge": []}

    def fake_replay(case, model, proxy_url, max_turns=4):
        calls["replay"].append(model)
        text = "It's down: the curl failed." if model == "new-model" else "It's probably fine."
        return {
            "kind": "replay",
            "replayVersion": claude_code.REPLAY_VERSION,
            "model": model,
            "requestedModel": model,
            "first": text,
            "firstToolCalls": [],
            "text": text,
            "toolAttempts": [],
            "costUsd": 0.5,
        }

    def fake_score(complaint, earlier, turn, response, judge_model="opus"):
        calls["judge"].append(response)
        score = 9 if "down" in response else (5 if "probably" in response else 0)
        return {"score": score, "rationale": "r", "judgeModel": "judge-x", "costUsd": 0.01}

    monkeypatch.setattr(claude_code, "replay", fake_replay)
    monkeypatch.setattr(judge, "score", fake_score)
    return calls


def test_new_model_is_compared_with_the_old_model_rerun(case, fakes):
    row = compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert (row["flagged"], row["control"], row["new"]) == (0, 5, 9)
    assert row["vs"] == "control" and row["outcome"] == "better"
    assert sorted(fakes["replay"]) == ["claude-opus-5", "new-model"]


def test_results_are_cached(case, fakes):
    compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    n_replays, n_judges = len(fakes["replay"]), len(fakes["judge"])
    compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert (len(fakes["replay"]), len(fakes["judge"])) == (n_replays, n_judges)


def test_without_control_falls_back_to_flagged(case, fakes):
    row = compare.compare_case(
        case, "new-model", "opus", "deny", "http://proxy", fresh=False, control=False
    )
    assert row["vs"] == "flagged" and row["reference"] == 0 and row["outcome"] == "better"


def test_samples_average(case, fakes):
    row = compare.compare_case(
        case, "new-model", "opus", "deny", "http://proxy", fresh=False, samples=3
    )
    assert row["newScores"] == [9, 9, 9]
    assert fakes["replay"].count("new-model") == 3


def test_replay_error_is_reported_not_scored(case, monkeypatch, fakes):
    monkeypatch.setattr(
        claude_code,
        "replay",
        lambda *a, **k: {"kind": "replay", "requestedModel": "x", "error": "boom"},
    )
    row = compare.compare_case(case, "x", "opus", "deny", "http://proxy", fresh=False)
    assert row["outcome"] == "error" and row["error"] == "boom"


@pytest.mark.parametrize(
    "old,new,want", [(0, 2, "better"), (5, 4, "same"), (6, 3, "worse"), (None, 3, "error")]
)
def test_outcome(old, new, want):
    assert compare.outcome(old, new) == want


def test_response_for_variants():
    rep = {"first": "Checking.", "firstToolCalls": ["Bash: ps"], "text": "Checking.\n\nIt's down."}
    assert compare.response_for(rep, "deny") == "Checking.\n\nIt's down."
    assert compare.response_for(rep, "stop") == "Checking.\n\n[calls tool Bash: ps]"


def test_reports_render(case, fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    rows = [compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)]
    text = report.terminal(rows, "opus", color=False)
    assert "better   1" in text and "1 transcripts" in text
    paths = report.write_all(rows, "opus")
    page = paths["html"].read_text()
    assert "server is up" in page and 'id="scorecard"' in page
    assert report.rerender(paths["json"]).read_text() == page


@pytest.mark.parametrize(
    "raw,want",
    [
        ("claude-opus-5-5", "Claude Opus 5.5"),
        ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ("gpt-6-sol", "GPT-6 Sol"),
        ("gpt-5.6-sol", "GPT-5.6 Sol"),
        ("opus", "opus"),
    ],
)
def test_pretty_model(raw, want):
    assert report.pretty_model(raw) == want


def test_judge_parse():
    v = judge.parse_verdict('```json\n{"rationale": "x", "evidence": ["q"], "score": 7}\n```')
    assert (v["score"], v["rationale"], v["evidence"]) == (7, "x", ["q"])
    assert judge.parse_verdict('{"score": 11}')["score"] is None
    assert judge.parse_verdict("no json here")["score"] is None
    v = judge.parse_verdict('{"rationale": "r", "score": null, "rejected": "only replay talk"}')
    assert v["score"] is None and v["rejected"] == "only replay talk"


def test_a_rejected_reply_skips_the_case(case, fakes, monkeypatch):
    from slash_bad import judge as judge_mod

    def score(complaint, earlier, turn, response, judge_model="opus"):
        if "down" in response:
            return {"score": None, "rejected": "talks about CLAUDE.md", "rationale": "r"}
        return {"score": 1, "rationale": "r", "judgeModel": "j"}

    monkeypatch.setattr(judge_mod, "score", score)
    row = compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert row["outcome"] == "rejected"
    assert row["rejected"] == "the new model's reply: talks about CLAUDE.md"


def test_an_old_model_that_behaves_is_not_reproduced(case, fakes, monkeypatch):
    from slash_bad import judge as judge_mod

    monkeypatch.setattr(
        judge_mod,
        "score",
        lambda c, e, t, response, judge_model="opus": {
            "score": 9 if "probably" in response else 1,
            "rationale": "r",
            "judgeModel": "j",
        },
    )
    row = compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert row["outcome"] == "not-reproduced"


def test_unmatched_complaint_moves_to_the_earlier_message(tmp_path, monkeypatch, fakes):
    import json

    from conftest import cc_assistant, cc_user

    from slash_bad import judge as judge_mod

    recs = [
        cc_user("how did the run go?"),
        cc_assistant([{"type": "text", "text": "Great question! It went well."}], "a1"),
        cc_assistant([{"type": "text", "text": "Anything else?"}], "a2"),
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta, context = claude_code.build_case(path, "don't open with great question")
    case = Case.create(meta, context)

    def score(complaint, earlier, turn, response, judge_model="opus"):
        s = 0 if "Great question" in response else 9
        return {"score": s, "rationale": "r", "judgeModel": "j", "costUsd": 0}

    monkeypatch.setattr(judge_mod, "score", score)
    row = compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert row.get("relocated") and row["flagged"] == 0
    assert case.meta["flagged"] == "Great question! It went well."
    assert (case.path / "context.before-relocate.jsonl").exists()


def test_complaint_found_nowhere_is_unmatched(case, monkeypatch, fakes):
    from slash_bad import judge as judge_mod

    monkeypatch.setattr(
        judge_mod, "score", lambda *a, **k: {"score": 10, "rationale": "absent", "judgeModel": "j"}
    )
    row = compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)
    assert row["outcome"] == "unmatched"
    assert fakes["replay"] == []


def test_a_case_from_the_model_under_test_is_skipped(case, fakes):
    row = compare.compare_case(case, "claude-opus-5", "opus", "deny", "http://proxy", fresh=False)
    assert row["outcome"] == "same-model"
    assert fakes["replay"] == []


def test_headline_reads_as_a_sentence(case, fakes):
    rows = [compare.compare_case(case, "new-model", "opus", "deny", "http://proxy", fresh=False)]
    assert report.headline(rows) == (
        "According to slash-bad, new-model is better on 1 of the 1 moments that annoyed me, "
        "compared with claude-opus-5."
    )
