import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from slash_bad.proxy import CONTINUE_NUDGE, NO_RESPONSE, SENTINEL, SentinelProxy, strip_sentinel


def _msgs():
    return [
        {"role": "user", "content": [{"type": "text", "text": "is it up?"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "down"},
                {"type": "text", "text": CONTINUE_NUDGE},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": NO_RESPONSE}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>env</system-reminder>"},
                {"type": "text", "text": SENTINEL},
            ],
        },
    ]


def test_strip_restores_the_context_exactly():
    out = strip_sentinel(_msgs())
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[-1]["content"] == [{"type": "tool_result", "tool_use_id": "t", "content": "down"}]


def test_strip_keeps_messages_after_the_sentinel():
    msgs = _msgs() + [
        {"role": "assistant", "content": [{"type": "text", "text": "new answer"}]},
    ]
    out = strip_sentinel(msgs)
    assert out[-1]["content"][0]["text"] == "new answer"
    assert out[-2]["role"] == "user"


def test_no_sentinel_is_untouched():
    msgs = _msgs()[:3]
    assert strip_sentinel(msgs) is msgs


def test_proxy_rewrites_and_forwards_streaming_body():
    seen = {}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            seen["body"] = json.loads(body)
            seen["auth"] = self.headers.get("authorization")
            payload = b"event: ping\ndata: {}\n\n" * 3
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        with SentinelProxy(f"http://127.0.0.1:{up.server_address[1]}") as url:
            req = Request(
                url + "/v1/messages",
                data=json.dumps({"model": "m", "messages": _msgs()}).encode(),
                headers={"authorization": "Bearer abc", "content-type": "application/json"},
                method="POST",
            )
            body = urlopen(req, timeout=10).read()
    finally:
        up.shutdown()
    assert body.count(b"event: ping") == 3
    assert seen["auth"] == "Bearer abc"
    assert len(seen["body"]["messages"]) == 3
    assert SENTINEL not in json.dumps(seen["body"])


def test_strip_handles_system_messages_between_the_patches():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "run it"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                {"type": "text", "text": CONTINUE_NUDGE},
            ],
        },
        {"role": "system", "content": "hook context"},
        {"role": "assistant", "content": [{"type": "text", "text": NO_RESPONSE}]},
        {"role": "system", "content": "You are powered by ..."},
        {"role": "user", "content": [{"type": "text", "text": SENTINEL}]},
    ]
    out = strip_sentinel(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "system", "system"]
    assert out[2]["content"] == [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]
