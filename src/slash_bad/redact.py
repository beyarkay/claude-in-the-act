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
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "anthropic-key"),
    (r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}", "openai-key"),
    (r"hf_[A-Za-z0-9]{30,}", "huggingface-token"),
    (r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}", "github-token"),
    (r"github_pat_[A-Za-z0-9_]{40,}", "github-token"),
    (r"glpat-[A-Za-z0-9_\-]{20,}", "gitlab-token"),
    (r"(?:AKIA|ASIA)[0-9A-Z]{16}", "aws-key"),
    (r"xox[abprs]-[A-Za-z0-9\-]{10,}", "slack-token"),
    (r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+", "slack-webhook"),
    (r"https://discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_\-]+", "discord-webhook"),
    (r"AIza[0-9A-Za-z_\-]{35}", "google-key"),
    (r"ya29\.[0-9A-Za-z_\-]{20,}", "google-oauth"),
    (r"(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{20,}", "stripe-key"),
    (r"SG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}", "sendgrid-key"),
    (r"npm_[A-Za-z0-9]{36}", "npm-token"),
    (r"pypi-[A-Za-z0-9_\-]{50,}", "pypi-token"),
    (r"gsk_[A-Za-z0-9]{40,}", "groq-key"),
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


def redact_case_dir(case_dir: Path, redactor: Redactor) -> dict:
    """``{meta, context}`` of a case, redacted."""
    meta = json.loads((case_dir / "case.json").read_text())
    lines = (case_dir / "context.jsonl").read_text().splitlines()
    context = [redactor.value(json.loads(line)) for line in lines if line.strip()]
    return {"meta": redactor.value(meta), "context": context}
