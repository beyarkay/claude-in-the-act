"""Scoresheets: a terminal summary, a shareable 1200×675 card (SVG, plus PNG
when a converter is installed), and a detailed HTML report."""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from slash_bad import REPORTS_DIR

GREEN, RED, GREY, BOLD, DIM, RESET = (
    "\x1b[32m",
    "\x1b[31m",
    "\x1b[90m",
    "\x1b[1m",
    "\x1b[2m",
    "\x1b[0m",
)
MARK = {"better": "▲", "worse": "▼", "same": "●", "error": "✗"}


def _color(outcome: str) -> str:
    return {"better": GREEN, "worse": RED, "same": GREY}.get(outcome, RED)


def summary(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["outcome"] != "error"]
    counts = Counter(r["outcome"] for r in rows)
    old = [r["old"] for r in scored]
    new = [r["new"] for r in scored]
    models = Counter(r.get("newModel") or r["requestedModel"] for r in scored)
    old_models = Counter(r["oldModel"] for r in scored)
    return {
        "better": counts["better"],
        "same": counts["same"],
        "worse": counts["worse"],
        "errors": counts["error"],
        "scored": len(scored),
        "oldMean": sum(old) / len(old) if old else None,
        "newMean": sum(new) / len(new) if new else None,
        "newModel": models.most_common(1)[0][0] if models else "",
        "oldModels": [m for m, _ in old_models.most_common()],
        "costUsd": sum(r.get("costUsd") or 0 for r in rows),
        "toolTries": sum(1 for r in scored if r.get("toolAttempts")),
    }


def _ordered(rows: list[dict]) -> list[dict]:
    rank = {"better": 0, "worse": 1, "same": 2, "error": 3}
    return sorted(
        rows,
        key=lambda r: (rank[r["outcome"]], -abs((r.get("new") or 0) - (r.get("old") or 0))),
    )


def _one_line(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def terminal(rows: list[dict], judge: str, color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    c = (lambda code: code) if color else (lambda code: "")
    s = summary(rows)
    olds = ", ".join(s["oldModels"]) or "?"
    out = [
        f"{c(BOLD)}slash-bad · your personal benchmark{c(RESET)}",
        f"{olds} → {c(BOLD)}{s['newModel']}{c(RESET)}   {c(DIM)}judged blind by {judge}{c(RESET)}",
        "",
    ]
    width = max(s["better"], s["same"], s["worse"], 1)
    for key, label in (("better", "better"), ("same", "same  "), ("worse", "worse ")):
        bar = "■" * round(20 * s[key] / width) if s[key] else ""
        out.append(f"  {c(_color(key))}{MARK[key]} {label} {s[key]:>3}  {bar}{c(RESET)}")
    if s["errors"]:
        out.append(f"  {c(RED)}✗ errors {s['errors']:>3}{c(RESET)}  (see the report for why)")
    if s["oldMean"] is not None:
        delta = s["newMean"] - s["oldMean"]
        out += [
            "",
            f"  mean score {s['oldMean']:.1f} → {s['newMean']:.1f} ({delta:+.1f})"
            f"   {c(DIM)}10 = your complaint no longer applies{c(RESET)}",
        ]
    out += ["", f"{c(BOLD)}per case{c(RESET)}"]
    for r in _ordered(rows):
        mark = f"{c(_color(r['outcome']))}{MARK[r['outcome']]}"
        if r["outcome"] == "error":
            out.append(f"  {mark} error{c(RESET)}  {_one_line(r['complaint'], 70)}")
            out.append(f"      {c(DIM)}{_one_line(r.get('error', ''), 100)}{c(RESET)}")
            continue
        out.append(
            f"  {mark} {r['old']:>2} → {r['new']:<2}{c(RESET)} {_one_line(r['complaint'], 80)}"
        )
        out.append(f"      {c(DIM)}now: {_one_line(r.get('newResponse', ''), 96)}{c(RESET)}")
    if s["toolTries"]:
        out += [
            "",
            f"{c(DIM)}{s['toolTries']} replay(s) tried to call a tool first; the call was refused "
            f"and the model answered without it.{c(RESET)}",
        ]
    return "\n".join(out)


W, H = 1200, 675
BG, FG, MUTED, LINE = "#0e1116", "#eceae6", "#8d93a0", "#262b34"
C_BETTER, C_WORSE, C_SAME = "#4ade80", "#f87171", "#a1a6b0"
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"


def _t(x, y, text, size, fill=FG, weight=400, anchor="start", family=FONT, extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{html.escape(str(text))}</text>"
    )


def card_svg(rows: list[dict], judge: str, show_complaints: bool = True) -> str:
    s = summary(rows)
    olds = ", ".join(s["oldModels"][:2]) + (" …" if len(s["oldModels"]) > 2 else "")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        _t(64, 84, "/bad", 40, weight=800, family=MONO),
        _t(170, 84, "my personal benchmark", 26, fill=MUTED),
        _t(64, 146, f"{olds}  →  {s['newModel']}", 34, weight=700),
        _t(
            64,
            184,
            f"re-ran {s['scored']} moments where a model annoyed me · judged blind by {judge}",
            20,
            fill=MUTED,
        ),
    ]
    x = 64
    for key, col, word in (
        ("better", C_BETTER, "better"),
        ("same", C_SAME, "same"),
        ("worse", C_WORSE, "worse"),
    ):
        parts.append(_t(x, 300, s[key], 96, fill=col, weight=800))
        parts.append(_t(x + 4, 336, word, 24, fill=col, weight=600))
        x += 150
    if s["oldMean"] is not None:
        parts.append(_t(64, 404, "average score (10 = complaint gone)", 18, fill=MUTED))
        parts.append(
            _t(64, 446, f"{s['oldMean']:.1f} → {s['newMean']:.1f}", 36, weight=700)
        )

    # Right panel: one slope per case, old score → new score on a 0-10 axis.
    px, pw, top = 560, 576, 236
    shown = [r for r in _ordered(rows) if r["outcome"] != "error"][:8]
    parts.append(f'<line x1="{px}" y1="{top - 28}" x2="{px + pw}" y2="{top - 28}" stroke="{LINE}"/>')
    ax0, ax1 = px + (300 if show_complaints else 40), px + pw - 16
    parts.append(_t(ax0, top - 40, "0", 14, fill=MUTED, anchor="middle"))
    parts.append(_t(ax1, top - 40, "10", 14, fill=MUTED, anchor="middle"))
    for i, r in enumerate(shown):
        y = top + i * 46
        col = {"better": C_BETTER, "worse": C_WORSE}.get(r["outcome"], C_SAME)
        if show_complaints:
            parts.append(_t(px, y + 6, _one_line(r["complaint"], 30), 16, fill=FG))
        x_old = ax0 + (ax1 - ax0) * r["old"] / 10
        x_new = ax0 + (ax1 - ax0) * r["new"] / 10
        parts.append(f'<line x1="{ax0}" y1="{y}" x2="{ax1}" y2="{y}" stroke="{LINE}" stroke-width="2"/>')
        parts.append(
            f'<line x1="{x_old}" y1="{y}" x2="{x_new}" y2="{y}" stroke="{col}" stroke-width="4" '
            f'stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{x_old}" cy="{y}" r="6" fill="{BG}" stroke="{MUTED}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x_new}" cy="{y}" r="8" fill="{col}"/>')
    more = len([r for r in rows if r["outcome"] != "error"]) - len(shown)
    if more > 0:
        parts.append(_t(px, top + len(shown) * 46 + 4, f"+ {more} more", 16, fill=MUTED))
    parts.append(
        _t(px, H - 96, "○ old response    ● new model's response", 15, fill=MUTED)
    )
    parts.append(f'<line x1="64" y1="{H - 64}" x2="{W - 64}" y2="{H - 64}" stroke="{LINE}"/>')
    parts.append(
        _t(64, H - 30, "slash-bad · your personal benchmark for the things you actually care about", 18, fill=MUTED)
    )
    parts.append(_t(W - 64, H - 30, datetime.now(UTC).strftime("%Y-%m-%d"), 18, fill=MUTED, anchor="end"))
    parts.append("</svg>")
    return "\n".join(parts)


def svg_to_png(svg_path: Path) -> Path | None:
    png = svg_path.with_suffix(".png")
    if shutil.which("rsvg-convert"):
        proc = subprocess.run(
            ["rsvg-convert", "-w", str(W * 2), "-o", str(png), str(svg_path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return png
    for chrome in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ):
        if chrome and Path(chrome).exists():
            proc = subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    f"--window-size={W},{H}",
                    f"--screenshot={png}",
                    svg_path.as_uri(),
                ],
                capture_output=True,
                check=False,
                timeout=60,
            )
            if proc.returncode == 0 and png.exists():
                return png
    return None


_CSS = """
:root { --bg:#fbfaf8; --fg:#1d1f23; --muted:#6b7079; --line:#e6e3de; --card:#fff;
        --better:#15803d; --worse:#b91c1c; --same:#6b7079; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0e1116; --fg:#eceae6; --muted:#8d93a0; --line:#262b34; --card:#151922;
          --better:#4ade80; --worse:#f87171; --same:#a1a6b0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.55 Inter, -apple-system, 'Helvetica Neue', sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 28px; margin: 0 0 4px; } .muted { color: var(--muted); }
.card-img { width: 100%; border-radius: 12px; border: 1px solid var(--line); margin: 24px 0; }
.case { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 18px 20px; margin: 16px 0; }
.case h2 { font-size: 17px; margin: 0 0 12px; display: flex; gap: 12px; align-items: baseline; }
.pill { font: 600 13px/1 'SF Mono', Menlo, monospace; padding: 5px 8px; border-radius: 999px;
        border: 1px solid currentColor; white-space: nowrap; }
.better { color: var(--better); } .worse { color: var(--worse); } .same, .error { color: var(--same); }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
.resp { font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 320px;
        overflow: auto; border-left: 3px solid var(--line); padding: 4px 12px; }
.why { font-size: 13px; color: var(--muted); margin-top: 8px; }
.label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
"""


def html_report(rows: list[dict], judge: str, card_name: str | None) -> str:
    s = summary(rows)
    esc = html.escape
    errors = f" · {s['errors']} errors" if s["errors"] else ""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>slash-bad · {esc(s['newModel'])}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{esc(', '.join(s['oldModels']))} → {esc(s['newModel'])}</h1>",
        f"<div class='muted'>{s['better']} better · {s['same']} same · {s['worse']} worse"
        f"{errors}"
        f" · judged blind by {esc(judge)} · ${s['costUsd']:.2f}</div>",
    ]
    if card_name:
        parts.append(f"<img class='card-img' src='{esc(card_name)}' alt='scorecard'>")
    for r in _ordered(rows):
        o = r["outcome"]
        score = f"{r.get('old')} → {r.get('new')}" if o != "error" else "error"
        parts.append(f"<section class='case'><h2><span class='pill {o}'>{MARK[o]} {score}</span>")
        parts.append(f"<span>{esc(r['complaint'])}</span></h2>")
        if o == "error":
            parts.append(f"<div class='why'>{esc(r.get('error', ''))}</div></section>")
            continue
        parts.append("<div class='cols'><div>")
        parts.append(f"<div class='label'>original · {esc(r['oldModel'])} · {r['old']}/10</div>")
        parts.append(f"<div class='resp'>{esc(r.get('oldResponse', ''))}</div>")
        parts.append(f"<div class='why'>{esc(r.get('oldWhy', ''))}</div></div><div>")
        parts.append(
            f"<div class='label'>replay · {esc(r.get('newModel', ''))} · {r['new']}/10</div>"
        )
        parts.append(f"<div class='resp'>{esc(r.get('newResponse', ''))}</div>")
        if r.get("toolAttempts"):
            tried = "; ".join(_one_line(t, 80) for t in r["toolAttempts"][:3])
            parts.append(f"<div class='why'>tried tools first (refused): {esc(tried)}</div>")
        parts.append(f"<div class='why'>{esc(r.get('newWhy', ''))}</div></div></div></section>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_all(rows: list[dict], judge: str, show_complaints: bool = True) -> dict[str, Path]:
    s = summary(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in s["newModel"] or "model")
    base = REPORTS_DIR / f"{stamp}-{safe}"
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    svg = base.with_suffix(".svg")
    svg.write_text(card_svg(rows, judge, show_complaints))
    paths["svg"] = svg
    if png := svg_to_png(svg):
        paths["png"] = png
    card = paths.get("png", svg).name
    page = base.with_suffix(".html")
    page.write_text(html_report(rows, judge, card))
    paths["html"] = page
    data = base.with_suffix(".json")
    data.write_text(json.dumps({"judge": judge, "summary": s, "rows": rows}, indent=2))
    paths["json"] = data
    return paths
