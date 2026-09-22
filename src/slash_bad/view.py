"""A harness-neutral view of a conversation, for the judge and for display.

Each harness module turns its native records into a list of steps::

    {"kind": "user" | "assistant" | "tool_call" | "tool_result", "text": str,
     "name": str (tool_call only)}

and the judge only ever sees this rendering, so Claude Code and Codex cases are
scored on identical terms.
"""

from __future__ import annotations

import json

TOOL_CALL_CAP = 600
TOOL_RESULT_CAP = 1500
PROSE_CAP = 6000
EARLIER_CAP = 800


def clip(s: str, cap: int) -> str:
    s = s.strip()
    if len(s) <= cap:
        return s
    return s[: cap // 2] + f"\n… [{len(s) - cap} chars clipped] …\n" + s[-cap // 2 :]


def as_text(content: object) -> str:
    """Flatten a tool result / message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "image":
                    parts.append("[image]")
                else:
                    parts.append(str(b.get("text") or b.get("content") or ""))
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return json.dumps(content)


def tool_call_text(name: str, args: object) -> str:
    if isinstance(args, dict) and len(args) == 1:
        args = next(iter(args.values()))
    body = args if isinstance(args, str) else json.dumps(args)
    return f"{name}: {body}"


def render(steps: list[dict]) -> str:
    """Steps as tagged text: the flagged turn in full, tool I/O clipped."""
    out = []
    for s in steps:
        kind, text = s["kind"], s.get("text", "")
        if kind == "user":
            out.append(f"<user>\n{clip(text, PROSE_CAP)}\n</user>")
        elif kind == "assistant":
            out.append(f"<assistant>\n{clip(text, PROSE_CAP)}\n</assistant>")
        elif kind == "tool_call":
            out.append(f"<tool-call>\n{clip(text, TOOL_CALL_CAP)}\n</tool-call>")
        elif kind == "tool_result":
            out.append(f"<tool-result>\n{clip(text, TOOL_RESULT_CAP)}\n</tool-result>")
    return "\n".join(out)


def judge_context(steps: list[dict], earlier_steps: int = 6) -> tuple[str, str]:
    """``(earlier, turn)`` for the judge: a short tail of the conversation before
    the final user message (prose only), then everything from that message on."""
    user_idxs = [i for i, s in enumerate(steps) if s["kind"] == "user"]
    start = user_idxs[-1] if user_idxs else 0
    earlier = [
        {**s, "text": clip(s["text"], EARLIER_CAP)}
        for s in steps[:start]
        if s["kind"] in ("user", "assistant")
    ][-earlier_steps:]
    return render(earlier), render(steps[start:])
