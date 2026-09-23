import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from slash_bad import llm


def test_luna_goes_through_openrouter(monkeypatch):
    seen = {}

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen["auth"] = self.headers["authorization"]
            out = json.dumps(
                {
                    "model": "openai/gpt-6-luna",
                    "choices": [{"message": {"content": '{"score": 3}'}}],
                    "usage": {"cost": 0.0004},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "OPENROUTER_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    try:
        text, cost, model = llm.ask("judge this", "luna", "be strict")
    finally:
        server.shutdown()
    assert (text, cost, model) == ('{"score": 3}', 0.0004, "openai/gpt-6-luna")
    assert seen["body"]["model"] == "openai/gpt-6-luna"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be strict"}
    assert seen["auth"] == "Bearer k"


def test_openrouter_without_a_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.ask("x", "openrouter:openai/gpt-6-luna") == (
        "",
        None,
        "OPENROUTER_API_KEY is not set",
    )
