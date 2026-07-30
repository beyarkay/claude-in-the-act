"""Unit tests for the judge's pure helpers (no `claude` calls)."""

from __future__ import annotations

import json

import pytest

from claude_in_the_act import judge


def test_judge_prompt_includes_the_pieces():
    p = judge.build_judge_prompt(
        feedback="too verbose",
        user_prompt="summarise this",
        response="here is a very long summary",
        good=None,
    )
    assert "<user-prompt>\nsummarise this\n</user-prompt>" in p
    assert "<response>\nhere is a very long summary\n</response>" in p
    assert "<commentary>\ntoo verbose\n</commentary>" in p
    assert "<ideal-response>\n" not in p  # only the instructions mention the tag


def test_judge_prompt_includes_ideal_when_present():
    p = judge.build_judge_prompt("fb", "prompt", "the reply", "THE IDEAL ANSWER")
    assert "<ideal-response>\nTHE IDEAL ANSWER\n</ideal-response>" in p


def test_judge_prompt_is_blind():
    """The rubric must not reveal whether it's looking at the flagged original or
    a regeneration — that framing would floor the original and inflate the new
    response, which is exactly what makes the two scores incomparable."""
    p = judge.build_judge_prompt("too verbose", "summarise", "a reply", None)
    lowered = p.lower()
    for leak in ("bad-response", "new-response", "marked", "regenerat", "previous"):
        assert leak not in lowered, f"prompt leaks which side it is judging: {leak!r}"
    # and it must actively warn the judge not to assume the complaint applies
    assert "not necessarily the one shown here" in p
    assert "Do NOT assume it applies" in p


def test_judge_prompt_scores_the_named_defect_only():
    p = judge.build_judge_prompt("too verbose", "summarise", "a reply", None)
    assert "Judge the defect, not the average." in p
    assert "0 = <response> fully commits that failure" in p
    # symmetry: unrelated flaws must not drag the score down either
    assert "do NOT lower the score for flaws the commentary does not" in p


def test_parse_json_plain():
    assert judge._parse_json_obj('{"score": 7, "rationale": "ok"}') == {
        "score": 7,
        "rationale": "ok",
    }


def test_parse_json_with_code_fence():
    raw = '```json\n{"score": 3, "rationale": "meh"}\n```'
    assert judge._parse_json_obj(raw)["score"] == 3


def test_parse_json_embedded_in_prose():
    raw = 'Here is my verdict: {"score": 9, "rationale": "great"} — done.'
    assert judge._parse_json_obj(raw)["score"] == 9


def test_parse_json_raises_when_no_object():
    with pytest.raises(ValueError):
        judge._parse_json_obj("no json here at all")


def _bundle(tmp_path, extra: list[dict] | None = None):
    recs = [
        {"type": "user", "message": {"role": "user", "content": "do X"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "the bad answer"}],
            },
        },
        {
            "type": "bad_annotation",
            "feedback": "too verbose",
            "badResponse": "the bad answer",
            "goodResponse": judge.GOOD_RESPONSE_PLACEHOLDER,
        },
        *(extra or []),
    ]
    bundle = tmp_path / "b.jsonl"
    bundle.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return bundle


def test_judge_bundle_stored_judges_captured_response_without_replay(
    tmp_path, monkeypatch
):
    """It must pull the stored response + feedback + prompt straight from the
    bundle and hand them to the judge — never regenerate."""
    bundle = _bundle(tmp_path)
    seen = {}

    def fake_score(feedback, user_prompt, response, good=None, judge_model=None):
        seen.update(
            feedback=feedback, user_prompt=user_prompt, response=response, good=good
        )
        return {"score": 2, "rationale": "agrees it's bad", "cost": {"costUsd": 0.001}}

    monkeypatch.setattr(judge, "score_response", fake_score)
    monkeypatch.setattr(
        judge.recreate,
        "recreate_with_tools",
        lambda *a, **k: pytest.fail("must not regenerate"),
    )

    out = judge.judge_bundle_stored(bundle)
    assert seen == {
        "feedback": "too verbose",
        "user_prompt": "do X",
        "response": "the bad answer",
        "good": None,  # placeholder collapses to None
    }
    assert out["score"] == 2
    assert out["mode"] == "stored"
    assert out["responseModel"] == "claude-opus-4-8"
    assert out["hadGoodResponse"] is False
    assert out["judgeCost"] == {"costUsd": 0.001}


def test_judge_bundle_stored_refuses_empty_response(tmp_path, monkeypatch):
    """An empty captured response must raise, not be judged — judging "" yields a
    meaningless high score."""
    recs = [
        {"type": "user", "message": {"role": "user", "content": "do X"}},
        {"type": "bad_annotation", "feedback": "bad", "badResponse": ""},
    ]
    bundle = tmp_path / "b.jsonl"
    bundle.write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(
        judge, "score_response", lambda *a, **k: pytest.fail("must not judge empty")
    )
    with pytest.raises(RuntimeError, match="empty"):
        judge.judge_bundle_stored(bundle)


def test_judge_bundle_regen_scores_the_new_response_and_deltas(tmp_path, monkeypatch):
    """The regen judge must be handed the REGENERATED text (not the captured one)
    and must report the movement against a same-rubric baseline."""
    prior = {
        "type": "judge_result",
        "mode": "stored",
        "rubric": judge.RUBRIC,
        "score": 1,
    }
    bundle = _bundle(tmp_path, [prior])
    seen = {}

    def fake_score(feedback, user_prompt, response, good=None, judge_model=None):
        seen.update(response=response)
        return {"score": 8, "rationale": "defect gone", "cost": {"costUsd": 0.002}}

    monkeypatch.setattr(judge, "score_response", fake_score)
    monkeypatch.setattr(
        judge.recreate,
        "recreate",
        lambda *a, **k: {
            "model": "claude-opus-5",
            "prompt": "do X",
            "original": "the bad answer",
            "recreated": "the new answer",
            "cost": {"costUsd": 0.5},
        },
    )

    out = judge.judge_bundle_regen(bundle)
    assert seen["response"] == "the new answer"
    assert out["mode"] == "regen"
    assert out["score"] == 8
    assert out["baselineScore"] == 1
    assert out["delta"] == 7
    assert out["responseModel"] == "claude-opus-5"


def test_judge_bundle_regen_refuses_empty_regeneration(tmp_path, monkeypatch):
    """A replay that produced nothing is a harness failure, not a 0 — it must
    raise rather than hand the judge an empty string."""
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        judge, "score_response", lambda *a, **k: pytest.fail("must not judge empty")
    )
    monkeypatch.setattr(
        judge.recreate,
        "recreate",
        lambda *a, **k: {"recreated": "   ", "returncode": 1},
    )
    with pytest.raises(RuntimeError, match="no output"):
        judge.judge_bundle_regen(bundle)


def test_baseline_score_ignores_stale_rubric_and_regen_records(tmp_path):
    """Only a stored-mode score under the CURRENT rubric is a valid baseline —
    the June scores came from a differently-scaled judge and must not be
    subtracted from a new one."""
    recs = [
        {"type": "judge_result", "mode": "existing", "score": 1},  # pre-rubric
        {"type": "judge_result", "mode": "stored", "rubric": "older-v0", "score": 3},
        {"type": "judge_result", "mode": "regen", "rubric": judge.RUBRIC, "score": 9},
    ]
    assert judge.baseline_score(recs) is None
    recs.append(
        {"type": "judge_result", "mode": "stored", "rubric": judge.RUBRIC, "score": 2}
    )
    assert judge.baseline_score(recs) == 2
