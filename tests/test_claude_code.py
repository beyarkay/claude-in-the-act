import json

from conftest import cc_assistant, cc_tool_result, cc_user

from slash_bad import claude_code
from slash_bad.proxy import SENTINEL
from slash_bad.store import read_jsonl


def test_split_cuts_before_the_whole_flagged_message(cc_transcript):
    recs = read_jsonl(cc_transcript)
    sp = claude_code.split(recs)
    # The thinking record shares the flagged message's id, so it's regenerated too.
    assert recs[sp["cut"]]["message"]["content"][0]["type"] == "thinking"
    assert recs[sp["cut"] - 1]["type"] == "attachment"
    assert sp["flagged"] == "Great question! The server is running fine."
    assert sp["model"] == "claude-opus-5"


def test_split_skips_synthetic_messages():
    recs = [
        cc_user("hi"),
        cc_assistant([{"type": "text", "text": "real answer"}], "m1"),
        cc_assistant(
            [{"type": "text", "text": "No response requested."}], "m2", model="<synthetic>"
        ),
    ]
    assert claude_code.split(recs)["flagged"] == "real answer"


def test_is_prompt_ignores_tool_results_and_injections():
    assert claude_code.is_prompt(cc_user("fix the bug"))
    assert not claude_code.is_prompt(cc_tool_result("t", "out"))
    assert not claude_code.is_prompt(cc_user("<local-command-stdout>x</local-command-stdout>"))
    assert not claude_code.is_prompt(cc_user("x", origin={"kind": "task-notification"}))
    assert not claude_code.is_prompt(cc_user("x", isMeta=True))


def test_build_case_records_context_and_stats(cc_transcript):
    meta, context = claude_code.build_case(cc_transcript, "  it never checked  ")
    assert meta["complaint"] == "it never checked"
    assert meta["stats"]["afterToolResult"] is True
    assert meta["stats"]["toolCalls"] == 1
    assert meta["sessionId"] == "s1"
    assert all(r.get("message", {}).get("id") != "m2" for r in context)


def test_parse_replay_separates_first_message_from_later_prose():
    new = [
        cc_user(SENTINEL),
        cc_assistant([{"type": "text", "text": "Let me check."}], "n1", model="claude-opus-5-5"),
        cc_assistant(
            [{"type": "tool_use", "id": "x", "name": "Bash", "input": {"command": "ps"}}],
            "n1",
            model="claude-opus-5-5",
        ),
        cc_tool_result("x", "Permission denied"),
        cc_assistant(
            [{"type": "text", "text": "I can't check, so I don't know."}],
            "n2",
            model="claude-opus-5-5",
        ),
    ]
    out = claude_code.parse_replay(new)
    assert out["model"] == "claude-opus-5-5"
    assert out["first"] == "Let me check."
    assert out["firstToolCalls"] == ["Bash: ps"]
    assert out["text"] == "Let me check.\n\nI can't check, so I don't know."


def test_plant_text_sets_the_id_without_touching_content(cc_transcript):
    _, context = claude_code.build_case(cc_transcript, "c", session_id="x")
    planted = [json.loads(line) for line in claude_code.plant_text(context, "NEW").splitlines()]
    assert {r["sessionId"] for r in planted if "sessionId" in r} == {"NEW"}
    assert [r.get("message") for r in planted] == [r.get("message") for r in context]


def test_faithful_request_ends_where_the_context_ends():
    from conftest import cc_assistant, cc_user

    ctx = [
        cc_user("what does it do?"),
        cc_assistant([{"type": "text", "text": "Great question! It sums."}], "a1"),
        cc_user("Stop hook feedback: cut the compliment", isMeta=True),
    ]
    good = {"messages": [{"role": "user", "content": "Stop hook feedback: cut the compliment"}]}
    dropped = {"messages": [{"role": "user", "content": "what does it do?"}]}
    assert claude_code.faithful(good, ctx)
    assert not claude_code.faithful(dropped, ctx)


def test_a_long_session_is_replayed_with_the_long_window():
    from conftest import cc_assistant, cc_user

    small = cc_assistant([{"type": "text", "text": "ok"}], "a1")
    small["message"]["usage"] = {"input_tokens": 5, "cache_read_input_tokens": 90_000}
    big = cc_assistant([{"type": "text", "text": "ok"}], "a2")
    big["message"]["usage"] = {"input_tokens": 5, "cache_read_input_tokens": 400_000}
    assert claude_code.replay_model("claude-opus-4-8", [cc_user("hi"), small]) == "claude-opus-4-8"
    assert (
        claude_code.replay_model("claude-opus-4-8", [cc_user("hi"), small, big])
        == "claude-opus-4-8[1m]"
    )
