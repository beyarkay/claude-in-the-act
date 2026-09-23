"""A local pass-through proxy that lets a replay resume *without* a new prompt.

``claude -p --resume`` refuses to continue a session unless it's given a fresh
user message, and when the stored context ends mid-turn (right after a tool
result) Claude Code also patches in a ``Continue from where you left off.``
nudge and a synthetic ``No response requested.`` reply. All three would change
what the model sees, and a replay is only worth comparing if the model faces
exactly the context the flagged message was written in.

So the replay sends a sentinel prompt through this proxy
(``ANTHROPIC_BASE_URL``), which deletes the sentinel message and the two
patches from every outgoing request before forwarding it. The real harness
still builds the system prompt, tool schemas and auth, so replays pick up the
current Claude Code for free. Everything else passes through byte-for-byte.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SENTINEL = "SLASH_BAD_RESUME_SENTINEL_9c41e7"
CONTINUE_NUDGE = "Continue from where you left off."
NO_RESPONSE = "No response requested."

_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "keep-alive",
}


def _texts(msg: dict) -> list[str]:
    content = msg.get("content")
    if isinstance(content, str):
        return [content]
    return [
        b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"
    ]


def _has_sentinel(msg: dict) -> bool:
    return msg.get("role") == "user" and any(SENTINEL in t for t in _texts(msg))


def _is_patch(m: dict) -> bool:
    return m.get("role") == "assistant" and [t.strip() for t in _texts(m)] == [NO_RESPONSE]


def _drop_nudge(m: dict) -> dict | None:
    content = m.get("content")
    if m.get("role") != "user" or not isinstance(content, list):
        return m
    kept = [
        b
        for b in content
        if not (
            isinstance(b, dict)
            and b.get("type") == "text"
            and b.get("text", "").strip() == CONTINUE_NUDGE
        )
    ]
    return {**m, "content": kept} if kept else None


def split_sentinel(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict], str] | None:
    """``(context, resume_notes, new_turns, directive)``, or None without a sentinel.

    ``context`` is the stored conversation with the resume patches removed.
    ``resume_notes`` are the role:"system" messages Claude Code adds when it
    resumes (today's date, the environment, files changed on disk since, hook
    output); they sit next to the sentinel once the patches are gone.
    ``new_turns`` is what the model has done since, and ``directive`` the text
    after the sentinel.
    """
    k = next((i for i, m in enumerate(messages) if _has_sentinel(m)), None)
    if k is None:
        return None
    directive = next(t for t in _texts(messages[k]) if SENTINEL in t).split(SENTINEL, 1)[1]

    def clean(ms: list[dict]) -> list[dict]:
        out = []
        for m in ms:
            if not _is_patch(m) and (m := _drop_nudge(m)) is not None:
                out.append(m)
        return out

    before, after = clean(messages[:k]), clean(messages[k + 1 :])
    notes = []
    while before and before[-1].get("role") == "system":
        notes.insert(0, before.pop())
    while after and after[0].get("role") == "system":
        notes.append(after.pop(0))
    return before, notes, after, directive.strip()


def strip_sentinel(messages: list[dict]) -> list[dict]:
    """The request without the sentinel turn, the resume patches or the resume notes.

    Returns ``messages`` unchanged when there is no sentinel (side requests such
    as title generation, or the judge).
    """
    parts = split_sentinel(messages)
    if parts is None:
        return messages
    before, _, after, _ = parts
    return before + after


def _as_reminders(messages: list[dict]) -> list[dict]:
    """role:"system" messages as user text, for models that reject the system role."""
    out = []
    for m in messages:
        if m.get("role") == "system":
            m = {"role": "user", "content": [{"type": "text", "text": t} for t in _texts(m)]}
        out.append(m)
    return out


def _portable(messages: list[dict]) -> list[dict]:
    """Messages with only role and content: newer models get extra per-message fields
    (such as an effort setting, sometimes in an otherwise empty message) that older
    models' APIs reject."""
    kept = [{k: v for k, v in m.items() if k in ("role", "content")} for m in messages]
    return [m for m in kept if m.get("content")]


def _capture(path: Path, obj: dict, context: list[dict]) -> None:
    if path.exists():
        return
    snap = {
        "model": obj.get("model"),
        "system": obj.get("system"),
        "tools": obj.get("tools"),
        "messages": _portable(context),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap))
    tmp.replace(path)


def _use(path: Path, obj: dict, notes: list[dict], after: list[dict]) -> dict:
    """``obj`` with the saved request's system prompt, tools and conversation."""
    snap = json.loads(path.read_text())
    messages = _portable(snap["messages"]) + after
    live_roles = {m.get("role") for m in [*obj["messages"], *notes]}
    if "system" not in live_roles:
        messages = _as_reminders(messages)
    return {**obj, "system": snap["system"], "tools": snap["tools"], "messages": messages}


def rewrite_body(body: bytes) -> bytes:
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return body
    parts = split_sentinel(obj["messages"])
    if parts is None:
        return body
    before, notes, after, directive = parts
    mode, _, where = directive.partition("=")
    if mode == "use" and where:
        obj = _use(Path(where), obj, notes, after)
    else:
        if mode == "capture" and where:
            _capture(Path(where), obj, before)
            before = _portable(before)
        obj["messages"] = before + after
    if log := os.environ.get("SLASH_BAD_PROXY_LOG"):
        with open(log, "a") as f:
            f.write(json.dumps(obj) + "\n")
    return json.dumps(obj).encode()


class SentinelProxy:
    """``with SentinelProxy() as base_url:`` — a threaded proxy on a free port."""

    def __init__(self, upstream: str | None = None):
        self.upstream = (
            upstream or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        ).rstrip("/")
        self.rewrites = 0
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _forward(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else None
                if body is not None:
                    new = rewrite_body(body)
                    if new is not body:
                        with proxy._lock:
                            proxy.rewrites += 1
                    body = new
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
                req = urllib.request.Request(
                    proxy.upstream + self.path,
                    data=body,
                    headers=headers,
                    method=self.command,
                )
                try:
                    up = urllib.request.urlopen(req, timeout=900)
                except urllib.error.HTTPError as e:
                    up = e
                except (urllib.error.URLError, TimeoutError) as e:
                    msg = json.dumps({"error": f"slash-bad proxy: {e}"}).encode()
                    self.send_response(502)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(up.status)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP_HEADERS and k.lower() != "content-encoding":
                        self.send_header(k, v)
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                try:
                    while chunk := up.read1(65536):
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                finally:
                    up.close()

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _forward

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> str:
        self._thread.start()
        return self.url

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
