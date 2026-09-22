"""Backfill your benchmark from sessions you never flagged.

Scans every Claude Code and Codex session on this machine for two signals:

* a frustrated reply ("ffs", "wtf", "that's not what I asked", "why did you…"),
  which points at the agent message just before it;
* stock phrases agents overuse ("You're right", "worth knowing", "bites",
  "deliberately didn't"), which point at the message they appear in.

A model then reads each candidate, decides whether that message really shows
something this user would object to (using their own ``/bad`` complaints as the
reference for their taste), and writes the complaint. Confirmed ones become
cases tagged ``origin: mined``. Only sessions a person typed in are read:
``claude -p`` runs, Codex reviewers and subagents are skipped.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from slash_bad import HOME, REPLAY_ENV, claude_code, codex
from slash_bad.store import Case, all_cases

DEFAULT_MODEL = "sonnet"
MIN_CONFIDENCE = 0.7
PER_MODE_CAP = 6  # stock-phrase finds per failure mode, so one habit can't swamp the set

FRUSTRATED = re.compile(
    r"\b(ffs|istg|wtf|wtaf|fuck\w*|fck|shit\w*|jfc|omfg|goddamn\w*|dammit|bloody hell|ugh+|argh+)\b|"
    r"\bi (already )?(said|told you)\b|\bthat'?s not what i\b|\bnot what i asked\b|"
    r"\bwhy (did|would|are|do) you\b|\byou (didn'?t|never|keep|forgot|ignored)\b|\bno,? (stop|no)\b|"
    r"\bstop \w+ing\b|\bwhat do you mean\b|\bwdym\b|\?!|\bseriously\b|\bthat'?s wrong\b|"
    r"\bnonsense\b|\bmeaningless\b|\bconfus",
    re.I,
)
STOCK = re.compile(
    r"\bbites?\b|\bbiting\b|worth (naming|stating|knowing|noting|flagging|saying)|"
    r"deliberately (didn'?t|did not|left|kept|avoided)|corrections? to the premise|\bdearer\b|"
    r"rather than (recit|paraphras|guess|going from memory)|\b0 errors\b|all tests pass|"
    r"\bthe trap\b|you'?re (absolutely )?right|good call|fair (challenge|point)|"
    r"great question|good question|that changes things",
    re.I,
)

_PROMPT = """You are helping a user build a personal benchmark of moments where their AI coding agent behaved badly. Below is one message the agent wrote, the user's request before it, and the user's next message.

What this user objects to, in their own words from earlier complaints:
<their-complaints>
{taste}
</their-complaints>

<user-request>
{prev}
</user-request>

<agent-message>
{assistant}
</agent-message>

<user-next-message>
{reaction}
</user-next-message>

Decide whether <agent-message> ITSELF contains something this user would object to. Most agent messages are fine; say false unless the problem is plainly visible in the text of <agent-message>. Frustration in the user's next message may be about the task, a tool, or an earlier message, so it is a hint, not proof. A passing use of a common phrase counts only if it is the kind of thing this user complains about.

Reply with ONLY raw JSON:
{{"failure": true|false, "mode": "<a few words naming the behaviour>", "complaint": "<if failure: one or two blunt sentences in the user's voice, lowercase, quoting the offending words>", "confidence": <0.0-1.0>}}"""

_DEFAULT_TASTE = """- praising my ideas ("you're right", "good question", "good call")
- long preamble before the actual answer; too wordy
- stating things as fact without checking them
- jargon or abbreviations it never explained
- stock phrases: "worth knowing", "the trap", things that "bite", "deliberately didn't"
- doing work I didn't ask for; not understanding the task
- claiming success ("all tests pass", "0 errors") too early"""


def sources() -> list[tuple[str, Path]]:
    home = Path.home()
    out = []
    projects = claude_code.claude_home() / "projects"
    if projects.is_dir():
        out += [("claude-code", p) for p in projects.rglob("*.jsonl") if "subagents" not in p.parts]
    legacy = home / ".claude" / "claude-in-the-act" / "data"
    if legacy.is_dir():
        out += [("claude-code", p) for p in legacy.glob("*.jsonl")]
    codex_home = codex.codex_home()
    for base in (codex_home / "sessions", codex_home / "archived_sessions"):
        if base.is_dir():
            out += [("codex", p) for p in base.rglob("*.jsonl")]
    return out


def _load(path: Path) -> list[dict]:
    recs = []
    with open(path, errors="replace") as f:
        for line in f:
            with contextlib.suppress(json.JSONDecodeError):
                recs.append(json.loads(line))
    return recs


def _interactive(harness: str, recs: list[dict]) -> bool:
    if os.environ.get(REPLAY_ENV):
        return False
    if harness == "codex":
        meta = next((r.get("payload") or {} for r in recs if r.get("type") == "session_meta"), {})
        return meta.get("source") in ("cli", "vscode")
    entry = next((r.get("entrypoint") for r in recs if r.get("entrypoint")), "cli")
    return entry == "cli"


def turns(harness: str, recs: list[dict]) -> list[tuple[str, str, int]]:
    """``[(kind, text, index)]``: user prompts and agent messages, one entry per
    agent message, ``index`` being its last record."""
    out: list[tuple[str, str, int]] = []
    last_id = None
    for i, r in enumerate(recs):
        if harness == "codex":
            item = codex._item(r)
            if codex._is_prompt(item):
                out.append(("user", codex._message_text(item), i))
            elif (
                item.get("type") == "message"
                and item.get("role") == "assistant"
                and (t := codex._message_text(item))
            ):
                out.append(("assistant", t, i))
            continue
        if r.get("isSidechain"):
            continue
        if claude_code.is_prompt(r):
            out.append(("user", claude_code.as_text(r["message"]["content"]), i))
            last_id = None
        elif claude_code._is_real_assistant(r) and (t := claude_code._assistant_text(r)):
            mid = (r.get("message") or {}).get("id")
            if out and out[-1][0] == "assistant" and mid and mid == last_id:
                out[-1] = ("assistant", out[-1][1] + "\n\n" + t, i)
            else:
                out.append(("assistant", t, i))
            last_id = mid
    return out


def scan(progress=None) -> list[dict]:
    srcs = sources()
    found, seen = [], set()
    for n, (harness, path) in enumerate(srcs, 1):
        if progress:
            progress(n, len(srcs))
        recs = _load(path)
        if not _interactive(harness, recs):
            continue
        ts = turns(harness, recs)
        for j, (kind, text, idx) in enumerate(ts):
            if kind != "assistant":
                continue
            nxt = ts[j + 1][1] if j + 1 < len(ts) and ts[j + 1][0] == "user" else ""
            signal = "reaction" if nxt and FRUSTRATED.search(nxt) else None
            if not signal and (m := STOCK.search(text)):
                signal = f"phrase:{m.group(0).lower()}"
            if not signal:
                continue
            key = hashlib.sha1((text[:500] + nxt[:200]).encode()).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            prev = next((t for k, t, _ in reversed(ts[:j]) if k == "user"), "")
            found.append(
                {
                    "key": key,
                    "signal": signal,
                    "harness": harness,
                    "path": str(path),
                    "upto": idx + 1,
                    "prev": prev[-1500:],
                    "assistant": text[-4000:],
                    "reaction": nxt[:1200],
                }
            )
    return found


def taste() -> str:
    mine = [c.complaint for c in all_cases() if c.meta.get("origin") != "mined" and c.complaint]
    if not mine:
        return _DEFAULT_TASTE
    return "\n".join(f"- {' '.join(c.split())[:200]}" for c in mine[-40:])


def _ask(prompt: str, model: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="slash-bad-mine-") as cwd:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                model,
                "--system-prompt",
                "You answer with a single JSON object.",
                "--tools",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--output-format",
                "json",
            ],
            input=prompt,
            cwd=cwd,
            env={**os.environ, REPLAY_ENV: "1"},
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    try:
        raw = json.loads(proc.stdout).get("result", "")
        return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError, AttributeError):
        return {"failure": None}


def triage(cands: list[dict], model: str, workers: int = 12, progress=None) -> list[dict]:
    """Each candidate with a ``verdict``; verdicts are cached across runs."""
    cache_path = HOME / "mine-cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["key"]] = r["verdict"]
    their_taste = taste()
    todo = [c for c in cands if c["key"] not in cache]

    def one(c: dict) -> tuple[str, dict]:
        prompt = _PROMPT.format(
            taste=their_taste,
            prev=c["prev"] or "(none)",
            assistant=c["assistant"],
            reaction=c["reaction"] or "(none)",
        )
        return c["key"], _ask(prompt, model)

    HOME.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(workers) as ex, cache_path.open("a") as f:
        for n, (key, verdict) in enumerate(ex.map(one, todo), 1):
            if verdict.get("failure") is not None:
                f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
                f.flush()
            cache[key] = verdict
            if progress:
                progress(n, len(todo))
    return [{**c, "verdict": cache.get(c["key"], {})} for c in cands]


def select(judged: list[dict]) -> list[dict]:
    keep, per_mode = [], {}
    for c in sorted(judged, key=lambda c: -(c["verdict"].get("confidence") or 0)):
        v = c["verdict"]
        if not v.get("failure") or (v.get("confidence") or 0) < MIN_CONFIDENCE:
            continue
        if c["signal"] != "reaction":
            mode = " ".join((v.get("mode") or "").lower().split()[:3])
            per_mode[mode] = per_mode.get(mode, 0) + 1
            if per_mode[mode] > PER_MODE_CAP:
                continue
        keep.append(c)
    return keep


def save(chosen: list[dict]) -> list[Case]:
    have = {c.meta.get("mined", {}).get("key") for c in all_cases()}
    saved = []
    for c in chosen:
        if c["key"] in have:
            continue
        mod = codex if c["harness"] == "codex" else claude_code
        try:
            meta, context = mod.build_case(
                Path(c["path"]), c["verdict"]["complaint"], upto=c["upto"]
            )
        except ValueError:
            continue
        meta["origin"] = "mined"
        meta["mined"] = {
            "key": c["key"],
            "signal": c["signal"],
            "confidence": c["verdict"].get("confidence"),
            "mode": c["verdict"].get("mode"),
            "reaction": c["reaction"][:500],
        }
        saved.append(Case.create({**meta, "source": c["path"]}, context))
    return saved


def _bar(label: str):
    def draw(n: int, total: int) -> None:
        if sys.stderr.isatty():
            sys.stderr.write(f"\r{label} {n}/{total}\x1b[K")
            if n == total:
                sys.stderr.write("\n")
        elif n == total or n % 25 == 0:
            print(f"{label} {n}/{total}", file=sys.stderr)

    return draw


def main(args) -> int:
    cands = scan(progress=_bar("scanning sessions"))
    print(f"{len(cands)} candidate moments", file=sys.stderr)
    if args.limit:
        cands = cands[-args.limit :]
    judged = triage(cands, args.model, progress=_bar("reading them"))
    chosen = select(judged)
    print(f"{len(chosen)} look like real failures (confidence ≥ {MIN_CONFIDENCE})")
    for c in chosen[:15]:
        print(f"  - {' '.join(c['verdict'].get('complaint', '').split())[:110]}")
    if len(chosen) > 15:
        print(f"  … and {len(chosen) - 15} more")
    if args.dry_run:
        return 0
    saved = save(chosen)
    print(f"saved {len(saved)} new case(s), tagged mined. `slash-bad list` shows them; ")
    print("`slash-bad delete <id>` removes any you disagree with.")
    return 0
