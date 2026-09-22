import json

from slash_bad import codex


def ri(payload):
    return {"type": "response_item", "payload": payload}


def msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return ri({"type": "message", "role": role, "content": [{"type": kind, "text": text}]})


ROLLOUT = [
    {
        "type": "session_meta",
        "payload": {"id": "old-id", "cwd": "/Users/alice/p", "cli_version": "0.155.1"},
    },
    msg("user", "<environment_context>cwd</environment_context>"),
    {
        "type": "turn_context",
        "payload": {
            "model": "gpt-5.5",
            "comp_hash": "2911",
            "collaboration_mode": {"settings": {"model": "gpt-5.5"}},
        },
    },
    msg("user", "is the server up?"),
    ri({"type": "reasoning", "encrypted_content": "gAAA"}),
    ri(
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": "curl localhost"}),
            "call_id": "c1",
        }
    ),
    ri({"type": "function_call_output", "call_id": "c1", "output": "connection refused"}),
    {"type": "event_msg", "payload": {"type": "item_completed"}},
    ri({"type": "reasoning", "encrypted_content": "gBBB"}),
    msg("assistant", "Yes, it's up."),
    {"type": "token_usage_record", "payload": {}},
]


def test_steps_skip_injected_context():
    kinds = [s["kind"] for s in codex.steps(ROLLOUT)]
    assert kinds == ["user", "tool_call", "tool_result", "assistant"]


def test_split_drops_the_reasoning_for_the_flagged_message():
    sp = codex.split(ROLLOUT)
    assert sp["flagged"] == "Yes, it's up."
    assert sp["model"] == "gpt-5.5"
    assert ROLLOUT[sp["cut"] - 1]["payload"]["type"] == "function_call_output"


def test_prepare_rollout_claims_the_target_model(monkeypatch):
    monkeypatch.setattr(codex, "_comp_hash", lambda m: "3000")
    out = codex.prepare_rollout(ROLLOUT[:7], "old-id", "new-id", "gpt-6-sol")
    tc = next(r for r in out if r["type"] == "turn_context")["payload"]
    assert tc["model"] == "gpt-6-sol" and tc["comp_hash"] == "3000"
    assert tc["collaboration_mode"]["settings"]["model"] == "gpt-6-sol"
    assert out[0]["payload"]["id"] == "new-id"
    assert not any(r.get("payload", {}).get("type") == "reasoning" for r in out)
    assert ROLLOUT[2]["payload"]["model"] == "gpt-5.5"  # the stored context is untouched


def test_parse_replay_first_message_stops_at_tool_output():
    new = [
        {"type": "turn_context", "payload": {"model": "gpt-6-sol"}},
        msg("user", ""),
        msg("assistant", "Let me look."),
        ri({"type": "function_call", "name": "exec_command", "arguments": '{"cmd": "ps"}'}),
        ri({"type": "function_call_output", "output": "denied"}),
        msg("assistant", "It's down."),
    ]
    out = codex.parse_replay(new)
    assert out["model"] == "gpt-6-sol"
    assert out["first"] == "Let me look."
    assert out["firstToolCalls"] == ["exec_command: ps"]
    assert out["text"] == "Let me look.\n\nIt's down."


def test_split_prefers_the_final_answer_over_later_commentary():
    recs = ROLLOUT[:-1] + [
        {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
        msg("user", "now fix it"),
        ri(
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Looking at the layout."}],
            }
        ),
    ]
    recs[9]["payload"]["phase"] = "final_answer"
    assert codex.split(recs)["flagged"] == "Yes, it's up."


def test_mid_turn():
    started = {"type": "event_msg", "payload": {"type": "task_started"}}
    done = {"type": "event_msg", "payload": {"type": "task_complete"}}
    assert codex.mid_turn([started])
    assert not codex.mid_turn([started, done])
    assert codex.mid_turn([started, done, started])
