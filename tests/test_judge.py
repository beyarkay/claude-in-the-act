"""Unit tests for the judge's pure helpers (no `claude` calls)."""

from __future__ import annotations

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
