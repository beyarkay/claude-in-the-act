"""Unit tests for the judge's pure helpers (no `claude` calls)."""

from __future__ import annotations

import json

import pytest

from claude_in_the_act import judge


def test_build_prompt_includes_the_pieces():
    p = judge.build_prompt(
        feedback="too verbose",
        user_prompt="summarise this",
        bad="here is a very long summary",
        good=None,
        recreated="short summary",
    )
    assert "<user-feedback>\ntoo verbose\n</user-feedback>" in p
    assert "<bad-response>\nhere is a very long summary\n</bad-response>" in p
    assert "<new-response>\nshort summary\n</new-response>" in p
    assert (
        "<ideal-response>\n" not in p
    )  # no ideal-response block (only the instruction mentions the tag)


def test_build_prompt_includes_ideal_when_present():
    p = judge.build_prompt("fb", "prompt", "bad", "THE IDEAL ANSWER", "regen")
    assert "<ideal-response>\nTHE IDEAL ANSWER\n</ideal-response>" in p


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


def test_build_score_prompt_scores_the_response_alone():
    p = judge.build_score_prompt(
        feedback="too verbose",
        user_prompt="summarise this",
        bad="here is a very long summary",
        good=None,
    )
    assert "<response>\nhere is a very long summary\n</response>" in p
    assert "<user-feedback>\ntoo verbose\n</user-feedback>" in p
    assert "<new-response>" not in p  # nothing regenerated to compare against
    assert "<ideal-response>\n" not in p  # only the instruction mentions the tag
    # the rubric scores the named defect, not overall quality
    assert "Judge the defect, not the average." in p
    assert "0 = the response fully commits that failure" in p


def test_build_score_prompt_includes_ideal_when_present():
    p = judge.build_score_prompt("fb", "prompt", "the reply", "THE IDEAL ANSWER")
    assert "<ideal-response>\nTHE IDEAL ANSWER\n</ideal-response>" in p


def test_judge_bundle_existing_judges_stored_response_without_replay(
    tmp_path, monkeypatch
):
    """It must pull the stored badResponse + feedback + prompt straight from the
    bundle and hand them to the judge — never regenerate."""
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
    ]
    bundle = tmp_path / "b.jsonl"
    bundle.write_text("".join(json.dumps(r) + "\n" for r in recs))

    seen = {}

    def fake_score(feedback, user_prompt, bad, good=None, judge_model=None):
        seen.update(feedback=feedback, user_prompt=user_prompt, bad=bad, good=good)
        return {"score": 2, "rationale": "agrees it's bad", "cost": {"costUsd": 0.001}}

    # replaying would call recreate; assert it never happens
    monkeypatch.setattr(judge, "score_existing", fake_score)
    monkeypatch.setattr(
        judge.recreate,
        "recreate_with_tools",
        lambda *a, **k: pytest.fail("must not regenerate"),
    )

    out = judge.judge_bundle_existing(bundle)
    assert seen == {
        "feedback": "too verbose",
        "user_prompt": "do X",
        "bad": "the bad answer",
        "good": None,  # placeholder collapses to None
    }
    assert out["score"] == 2
    assert out["judgedModel"] == "claude-opus-4-8"
    assert out["hadGoodResponse"] is False
    assert out["judgeCost"] == {"costUsd": 0.001}


def test_judge_bundle_existing_refuses_empty_response(tmp_path, monkeypatch):
    """An empty captured response must raise, not be judged — judging "" yields a
    meaningless high score."""
    recs = [
        {"type": "user", "message": {"role": "user", "content": "do X"}},
        {"type": "bad_annotation", "feedback": "bad", "badResponse": ""},
    ]
    bundle = tmp_path / "b.jsonl"
    bundle.write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(
        judge, "score_existing", lambda *a, **k: pytest.fail("must not judge empty")
    )
    with pytest.raises(RuntimeError, match="empty"):
        judge.judge_bundle_existing(bundle)
