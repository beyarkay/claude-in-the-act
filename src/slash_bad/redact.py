"""Best-effort anonymisation of a case before it's shared.

Rules follow the Trace Commons ``scrub.py`` and DataClaw secret lists (known key
formats, credentials in URLs, key=value assignments, PEM blocks), plus what a
coding-agent transcript leaks on its own: the home directory in every path, the
username in Claude Code's dash-encoded project folders, and the git identity.

This is a filter, not a guarantee: code, file contents and project names pass
through untouched, which is why ``share`` always shows the result first.
"""

from __future__ import annotations

import getpass
import json
import re
import subprocess
from pathlib import Path

SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "private-key"),
    (r"(?<![A-Za-z0-9_\-])sk-ant-[A-Za-z0-9_\-]{20,}", "anthropic-key"),
    (r"(?<![A-Za-z0-9_\-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}", "openai-key"),
    (r"(?<![A-Za-z0-9_\-])hf_[A-Za-z0-9]{30,}", "huggingface-token"),
    (r"(?<![A-Za-z0-9_\-])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}", "github-token"),
    (r"(?<![A-Za-z0-9_\-])github_pat_[A-Za-z0-9_]{40,}", "github-token"),
    (r"(?<![A-Za-z0-9_\-])glpat-[A-Za-z0-9_\-]{20,}", "gitlab-token"),
    (r"(?<![A-Za-z0-9_\-])(?:AKIA|ASIA)[0-9A-Z]{16}", "aws-key"),
    (r"(?<![A-Za-z0-9_\-])xox[abprs]-[A-Za-z0-9\-]{10,}", "slack-token"),
    (r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+", "slack-webhook"),
    (r"https://discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_\-]+", "discord-webhook"),
    (r"(?<![A-Za-z0-9_\-])AIza[0-9A-Za-z_\-]{35}", "google-key"),
    (r"(?<![A-Za-z0-9_\-])ya29\.[0-9A-Za-z_\-]{20,}", "google-oauth"),
    (r"(?<![A-Za-z0-9_\-])(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{20,}", "stripe-key"),
    (r"(?<![A-Za-z0-9_\-])SG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}", "sendgrid-key"),
    (r"(?<![A-Za-z0-9_\-])npm_[A-Za-z0-9]{36}", "npm-token"),
    (r"(?<![A-Za-z0-9_\-])pypi-[A-Za-z0-9_\-]{50,}", "pypi-token"),
    (r"(?<![A-Za-z0-9_\-])gsk_[A-Za-z0-9]{40,}", "groq-key"),
    (r"[0-9]{8,10}:AA[0-9A-Za-z_\-]{33}", "telegram-token"),
    (r"FlyV1 [A-Za-z0-9_\-+/=,]{20,}", "fly-token"),
    (r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", "jwt"),
    (r"(?i)bearer\s+[A-Za-z0-9_\-.=:/+]{20,}", "bearer-token"),
    (r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s@/]+@", "url-credentials"),
    (
        r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*)"
        r"(\s*[:=]\s*)([\"']?)[^\s\"']{8,}\3",
        "assignment",
    ),
]
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
)
_HOME = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\?Users\\\\?)([^/\\\s\"']+)")
_ALLOWED_EMAILS = ("noreply@anthropic.com", "noreply@github.com", "noreply@openai.com")


def _git_identity() -> list[str]:
    out = []
    for key in ("user.name", "user.email"):
        try:
            v = subprocess.run(
                ["git", "config", "--global", key],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            v = ""
        if len(v) >= 3:
            out.append(v)
    return out


class Redactor:
    def __init__(self, extra: list[str] | None = None):
        self.user = getpass.getuser()
        self.literals = sorted(
            {s for s in [*(extra or []), *_git_identity()] if s and len(s) >= 3},
            key=len,
            reverse=True,
        )
        self.counts: dict[str, int] = {}
        self._secrets = [(re.compile(p), label) for p, label in SECRET_PATTERNS]
        self._user_word = (
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(self.user)}(?![A-Za-z0-9])")
            if len(self.user) >= 3
            else None
        )

    def _hit(self, label: str, n: int = 1) -> None:
        self.counts[label] = self.counts.get(label, 0) + n

    def text(self, s: str) -> str:
        for rx, label in self._secrets:
            if label == "assignment":
                s, n = rx.subn(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", s)
            elif label == "url-credentials":
                s, n = rx.subn(lambda m: m.group(0).split("://")[0] + "://<redacted>@", s)
            else:
                s, n = rx.subn(f"<redacted:{label}>", s)
            if n:
                self._hit(label, n)
        s, n = _HOME.subn(lambda m: m.group(1) + "user", s)
        if n:
            self._hit("home-path", n)
        s, n = re.subn(r"-(Users|home)-[A-Za-z0-9_.]+-", r"-\1-user-", s)
        if n:
            self._hit("home-path", n)
        for lit in self.literals:
            if lit in s:
                self._hit("custom", s.count(lit))
                s = s.replace(lit, "<redacted>")
        s, n = _EMAIL.subn(
            lambda m: m.group(0) if m.group(0).lower() in _ALLOWED_EMAILS else "<email>", s
        )
        if n:
            self._hit("email", n)
        s, n = _PRIVATE_IP.subn("<private-ip>", s)
        if n:
            self._hit("private-ip", n)
        if self._user_word:
            s, n = self._user_word.subn("user", s)
            if n:
                self._hit("username", n)
        return s

    def value(self, v: object) -> object:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            if v.get("type") == "image" or (
                isinstance(v.get("source"), dict) and v["source"].get("type") == "base64"
            ):
                self._hit("image")
                return {"type": "text", "text": "[image removed]"}
            return {k: self.value(x) for k, x in v.items()}
        return v


def _is_instructions(rec: dict) -> bool:
    """The user's CLAUDE.md / AGENTS.md: an ``instructions`` attachment in Claude
    Code, an injected ``# AGENTS.md`` / ``<user_instructions>`` message in Codex."""
    if rec.get("type") == "attachment":
        return (rec.get("attachment") or {}).get("type") == "instructions"
    payload = rec.get("payload") or {}
    if rec.get("type") != "response_item" or payload.get("type") != "message":
        return False
    text = "".join(
        b.get("text", "") for b in payload.get("content") or [] if isinstance(b, dict)
    ).lstrip()
    return text.startswith(("# AGENTS.md", "<user_instructions>"))


def redact_case_dir(case_dir: Path, redactor: Redactor, keep_instructions: bool = False) -> dict:
    """``{meta, context}`` of a case, redacted. CLAUDE.md contents are dropped
    unless ``keep_instructions``: they're the user's private configuration."""
    meta = json.loads((case_dir / "case.json").read_text())
    context = []
    for line in (case_dir / "context.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _is_instructions(rec) and not keep_instructions:
            redactor._hit("instructions-file")
            continue
        context.append(redactor.value(rec))
    return {"meta": redactor.value(meta), "context": context}
