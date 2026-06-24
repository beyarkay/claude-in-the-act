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
    assert "avg 4.0/10" in rep  # (8 + 0) / 2 scored
    assert "min 0" in rep and "max 8" in rep
    assert "2 scored, 1 error(s), 3 total" in rep
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


def test_run_eval_uses_custom_scorer():
    """The cheap judge-only mode is wired in by passing scorer=; run_eval must
    call it instead of the default judge_bundle."""
    calls = []

    def scorer(bundle, regen_model, judge_model):
        calls.append(bundle)
        return {"bundle": bundle, "feedback": "f", "score": 3, "rationale": "r"}

    res = evaluate.run_eval(["a.jsonl", "b.jsonl"], None, max_workers=2, scorer=scorer)
    assert sorted(calls) == ["a.jsonl", "b.jsonl"]
    assert all(r["score"] == 3 for r in res)


def test_cost_summary_sums_replay_and_judge():
    results = [
        {"score": 5, "regenCost": {"costUsd": 0.10}, "judgeCost": {"costUsd": 0.02}},
        {"score": 3, "judgeCost": {"costUsd": 0.03}},  # existing-mode row (no replay)
    ]
    assert "$0.15 total" in evaluate._cost_summary(results)
    # results without any cost data (older runs / mocks) yield no cost line
    assert evaluate._cost_summary([{"score": 1}]) is None
