"""Unit tests for the aggregate eval (judge_bundle is monkeypatched — no claude)."""

from __future__ import annotations

from claude_in_the_act import evaluate, judge


def test_format_report_overall_and_lines():
    results = [
        {
            "bundle": "a.jsonl",
            "feedback": "too long",
            "score": 8,
            "rationale": "terser now",
        },
        {
            "bundle": "b.jsonl",
            "feedback": "creds exposed",
            "score": 0,
            "rationale": "same mistake",
        },
        {
            "bundle": "c.jsonl",
            "feedback": "x",
            "score": None,
            "rationale": "ERROR: boom",
        },
    ]
    rep = evaluate.format_report(
        results, "claude-fable-5", "claude-sonnet-4-6", "2026-06-12"
    )
    assert "8/30" in rep  # total 8, max 10*3
    assert "avg 4.0/10 over 2 scored" in rep
    assert "claude-fable-5" in rep
    assert "too long" in rep and "terser now" in rep
    assert "ERROR: boom" in rep
    # worst-first ordering: the 0/10 appears before the 8/10
    assert rep.index("[0/10]") < rep.index("[8/10]")


def test_run_eval_runs_all_in_order(monkeypatch):
    def fake(bundle, regen_model=None, judge_model=evaluate.DEFAULT_JUDGE_MODEL):
        from pathlib import Path

        return {
            "bundle": Path(bundle).name,
            "feedback": "f",
            "score": 5,
            "rationale": "r",
        }

    monkeypatch.setattr(judge, "judge_bundle", fake)
    res = evaluate.run_eval(["x/a.jsonl", "y/b.jsonl", "z/c.jsonl"], "m", max_workers=3)
    assert [r["bundle"] for r in res] == ["a.jsonl", "b.jsonl", "c.jsonl"]
    assert all(r["score"] == 5 for r in res)


def test_run_eval_captures_per_bundle_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(judge, "judge_bundle", boom)
    res = evaluate.run_eval(["a.jsonl"], None, max_workers=1)
    assert res[0]["score"] is None
    assert "ERROR" in res[0]["rationale"]
