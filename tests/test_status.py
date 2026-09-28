from slash_bad import claude_code, judge, report, status
from slash_bad.store import Case


def _usage(n_in, read, write):
    return {
        "type": "assistant",
        "timestamp": "2026-06-11T16:48:37Z",
        "message": {
            "usage": {
                "input_tokens": n_in,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write,
                "output_tokens": 99,
            }
        },
    }


def test_tokens_come_from_the_last_usage_record():
    ctx = [_usage(1, 100, 0), _usage(5, 700_000, 390), {"type": "user"}]
    assert status.context_tokens("claude-code", ctx) == 700_395
    assert status.collected_at(ctx) == "2026-06-11T16:48:37Z"


def test_codex_tokens_come_from_the_last_token_count():
    tc = {
        "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 50_127}}}
    }
    assert status.context_tokens("codex", [tc, {"payload": {"type": "message"}}]) == 50_127


def _case(results):
    case = Case.create(
        {"harness": "claude-code", "model": "claude-opus-4-8", "complaint": "too wordy"},
        [_usage(1, 2, 3)],
    )
    for r in results:
        case.add_result(r)
    return case


def _replay(rid, model, **extra):
    return {
        "kind": "replay",
        "replayId": rid,
        "model": model,
        "replayVersion": claude_code.REPLAY_VERSION,
        **extra,
    }


def _verdict(target, score):
    return {
        "kind": "verdict",
        "target": target,
        "judge": "luna",
        "rubric": judge.RUBRIC,
        "score": score,
    }


def test_each_model_shows_the_replay_compare_would_use():
    case = _case(
        [
            {**_replay("old", "claude-opus-5"), "replayVersion": 1},
            _replay("bad", "claude-opus-5-5", error="resume dropped the end"),
            _replay("r1", "claude-opus-5-5"),
            _verdict("r1", 7),
            _replay("r2", "claude-opus-4-8"),
            _verdict("original", 1),
        ]
    )
    [row], judge_name = status.rows(cases=[case])
    assert judge_name == "luna"
    assert row["original"]["score"] == 1
    assert row["models"]["claude-opus-5-5"]["score"] == 7
    assert row["models"]["claude-opus-5"]["state"] == "outdated"
    assert row["models"]["claude-opus-4-8"]["state"] == "not judged"
    html = status.section([row], judge_name)
    assert "Opus 5.5" in html and "old replay" in html and "not judged" in html


def test_the_report_lists_every_saved_transcript():
    _case([_replay("r1", "claude-opus-5-5"), _verdict("r1", 4)])
    assert "All saved transcripts" in report.html_report([], "luna")
