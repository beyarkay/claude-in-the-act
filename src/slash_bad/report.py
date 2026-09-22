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
from datetime import datetime, timezone
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
MARK = {"better": "▲", "worse": "▼", "same": "●", "error": "✗", "unmatched": "?"}
SKIPPED = ("error", "unmatched")


def _color(outcome: str) -> str:
    return {"better": GREEN, "worse": RED, "same": GREY}.get(outcome, RED)


def fmt(x: float | None) -> str:
    if x is None:
        return "–"
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summary(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["outcome"] not in SKIPPED]
    counts = Counter(r["outcome"] for r in rows)
    models = Counter(r.get("newModel") or r["requestedModel"] for r in scored or rows)
    old_models = Counter(r["oldModel"] for r in scored or rows)
    judges = Counter(r.get("judgeModel") for r in scored if r.get("judgeModel"))
    with_control = [r for r in scored if r.get("control") is not None]
    return {
        "better": counts["better"],
        "same": counts["same"],
        "worse": counts["worse"],
        "errors": counts["error"],
        "unmatched": counts["unmatched"],
        "scored": len(scored),
        "flaggedMean": _mean([r["flagged"] for r in scored if r.get("flagged") is not None]),
        "controlMean": _mean([r["control"] for r in with_control]),
        "newMean": _mean([r["new"] for r in scored]),
        "controlled": len(with_control),
        "newModel": models.most_common(1)[0][0] if models else "",
        "oldModels": [m for m, _ in old_models.most_common()],
        "judge": judges.most_common(1)[0][0] if judges else "",
        "costUsd": sum(r.get("costUsd") or 0 for r in rows),
        "toolTries": sum(1 for r in scored if r.get("toolAttempts")),
        "selfJudged": bool(judges)
        and judges.most_common(1)[0][0] == (models.most_common(1)[0][0] if models else None),
    }


def groups(rows: list[dict]) -> list[dict]:
    """Rows pooled by behaviour: mean scores and one outcome per behaviour."""
    from slash_bad.compare import outcome

    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["outcome"] not in SKIPPED and r.get("behaviour"):
            by.setdefault(r["behaviour"], []).append(r)
    out = []
    for name, rs in by.items():
        ref, new = _mean([r["reference"] for r in rs]), _mean([r["new"] for r in rs])
        ctl = [r["control"] for r in rs if r.get("control") is not None]
        out.append(
            {
                "complaint": name,
                "n": len(rs),
                "flagged": _mean([r["flagged"] for r in rs if r.get("flagged") is not None]),
                "control": _mean(ctl) if len(ctl) == len(rs) else None,
                "reference": ref,
                "new": new,
                "outcome": outcome(ref, new),
                "cases": [r["case"] for r in rs],
            }
        )
    return out


def _ordered(rows: list[dict]) -> list[dict]:
    rank = {"better": 0, "worse": 1, "same": 2, "unmatched": 3, "error": 4}
    return sorted(
        rows,
        key=lambda r: (rank[r["outcome"]], -abs((r.get("new") or 0) - (r.get("reference") or 0))),
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
        f"{olds} → {c(BOLD)}{s['newModel']}{c(RESET)}   "
        f"{c(DIM)}judged blind by {s['judge'] or judge}{c(RESET)}",
        "",
    ]
    width = max(s["better"], s["same"], s["worse"], 1)
    for key, label in (("better", "better"), ("same", "same  "), ("worse", "worse ")):
        bar = "■" * round(20 * s[key] / width) if s[key] else ""
        out.append(f"  {c(_color(key))}{MARK[key]} {label} {s[key]:>3}  {bar}{c(RESET)}")
    if s["errors"]:
        out.append(f"  {c(RED)}✗ errors {s['errors']:>3}{c(RESET)}  (see the report for why)")
    if s["newMean"] is not None:
        out += ["", "  average score, 10 = your complaint no longer applies:"]
        if s["flaggedMean"] is not None:
            out.append(f"    {'the responses you flagged':<32} {s['flaggedMean']:.1f}")
        if s["controlMean"] is not None:
            out.append(f"    {(olds[:22] + ' again'):<32} {s['controlMean']:.1f}")
        out.append(f"    {s['newModel'][:32]:<32} {s['newMean']:.1f}")
    gs = groups(rows)
    if len(gs) >= 2:
        n = Counter(g["outcome"] for g in gs)
        out += [
            "",
            f"{c(BOLD)}by behaviour{c(RESET)}: better on {n['better']}, same on {n['same']}, "
            f"worse on {n['worse']}  {c(DIM)}(old model again → new model){c(RESET)}",
        ]
        for g in _ordered(gs):
            out.append(
                f"  {c(_color(g['outcome']))}{MARK[g['outcome']]} {fmt(g['reference']):>4} → "
                f"{fmt(g['new']):<4}{c(RESET)} {g['complaint']} {c(DIM)}({g['n']}){c(RESET)}"
            )
    out += [
        "",
        f"{c(BOLD)}per case{c(RESET)}  {c(DIM)}flagged · old model again · new model{c(RESET)}",
    ]
    for r in _ordered(rows):
        mark = f"{c(_color(r['outcome']))}{MARK[r['outcome']]}"
        if r["outcome"] == "error":
            out.append(f"  {mark} error{c(RESET)}  {_one_line(r['complaint'], 70)}")
            out.append(f"      {c(DIM)}{_one_line(r.get('error', ''), 100)}{c(RESET)}")
            continue
        if r["outcome"] == "unmatched":
            out.append(f"  {c(DIM)}? skipped   {_one_line(r['complaint'], 76)}{c(RESET)}")
            continue
        scores = f"{fmt(r['flagged']):>3} · {fmt(r['control']):>3} · {fmt(r['new']):<3}"
        out.append(f"  {mark} {scores}{c(RESET)} {_one_line(r['complaint'], 76)}")
        out.append(f"      {c(DIM)}now: {_one_line(r.get('newResponse', ''), 96)}{c(RESET)}")
    notes = []
    if s["unmatched"]:
        notes.append(
            f"{s['unmatched']} case(s) skipped: the judge couldn't find the complaint in the flagged "
            "message or the ones just before it, so a replay wouldn't measure it."
        )
    if s["selfJudged"]:
        notes.append(
            f"The judge ({s['judge']}) is the model being tested, and models tend to rate their own "
            "writing kindly. Pass --judge with a different model for a stricter comparison."
        )
    if s["scored"] and s["controlled"] < s["scored"]:
        notes.append(
            f"{s['scored'] - s['controlled']} case(s) couldn't re-run the old model, so the new "
            "model is compared with the flagged response there."
        )
    if s["toolTries"]:
        notes.append(
            f"{s['toolTries']} case(s) had the new model try a tool first; the call was refused "
            "and it answered without it."
        )
    if notes:
        out.append("")
        out += [f"{c(DIM)}{n}{c(RESET)}" for n in notes]
    return "\n".join(out)


W, H = 1200, 675
BG, FG, MUTED, LINE = "#0e1116", "#eceae6", "#8d93a0", "#262b34"
C_BETTER, C_WORSE, C_SAME = "#4ade80", "#f87171", "#a1a6b0"
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"


def _t(x, y, text, size, fill=FG, weight=400, anchor="start", family=FONT):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{html.escape(str(text))}</text>"
    )


def card_svg(rows: list[dict], judge: str, show_complaints: bool = True) -> str:
    s = summary(rows)
    gs = groups(rows)
    use_groups = len(gs) >= 2
    items = gs if use_groups else rows
    tally = Counter(i["outcome"] for i in items)
    olds = ", ".join(s["oldModels"][:2]) + (" …" if len(s["oldModels"]) > 2 else "")
    vs = "the old model re-run on the same moments" if s["controlled"] else "the flagged responses"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        _t(64, 84, "/bad", 40, weight=800, family=MONO),
        _t(170, 84, "my personal benchmark", 26, fill=MUTED),
        _t(64, 146, f"{olds}  →  {s['newModel']}", 34, weight=700),
        _t(
            64,
            184,
            f"{s['scored']} moments where a model annoyed me, re-run and judged blind by "
            f"{s['judge'] or judge}" + (" (itself)" if s["selfJudged"] else ""),
            18,
            fill=MUTED,
        ),
    ]
    x = 64
    for key, col in (("better", C_BETTER), ("same", C_SAME), ("worse", C_WORSE)):
        parts.append(_t(x, 300, tally[key], 96, fill=col, weight=800))
        parts.append(_t(x + 4, 336, key, 24, fill=col, weight=600))
        x += 150
    unit = f"on {len(gs)} behaviours I care about, " if use_groups else ""
    parts.append(_t(64, 372, f"{unit}vs {vs}", 16, fill=MUTED))
    if s["newMean"] is not None:
        y = 428
        parts.append(_t(64, y, "average score (10 = complaint gone)", 18, fill=MUTED))
        lines = [("flagged", s["flaggedMean"], MUTED)]
        if s["controlMean"] is not None:
            lines.append(("old model again", s["controlMean"], C_SAME))
        lines.append(("new model", s["newMean"], FG))
        for label, val, col in lines:
            y += 40
            parts.append(_t(64, y, f"{val:.1f}", 30, fill=col, weight=700))
            parts.append(_t(136, y, label, 20, fill=col))

    px, pw, top = 560, 576, 236
    shown = [r for r in _ordered(items) if r["outcome"] not in SKIPPED][:7]
    ax0, ax1 = px + (280 if show_complaints or use_groups else 40), px + pw - 16
    parts.append(_t(ax0, top - 40, "0", 14, fill=MUTED, anchor="middle"))
    parts.append(_t(ax1, top - 40, "10", 14, fill=MUTED, anchor="middle"))
    parts.append(
        f'<line x1="{px}" y1="{top - 28}" x2="{px + pw}" y2="{top - 28}" stroke="{LINE}"/>'
    )

    def at(v: float) -> float:
        return ax0 + (ax1 - ax0) * v / 10

    for i, r in enumerate(shown):
        y = top + i * 46
        col = {"better": C_BETTER, "worse": C_WORSE}.get(r["outcome"], C_SAME)
        if show_complaints or use_groups:
            label = f"{r['complaint']} ({r['n']})" if use_groups else r["complaint"]
            parts.append(_t(px, y + 6, _one_line(label, 30), 16, fill=FG))
        parts.append(
            f'<line x1="{ax0}" y1="{y}" x2="{ax1}" y2="{y}" stroke="{LINE}" stroke-width="2"/>'
        )
        ref = r["reference"]
        parts.append(
            f'<line x1="{at(ref):.1f}" y1="{y}" x2="{at(r["new"]):.1f}" y2="{y}" stroke="{col}" '
            f'stroke-width="4" stroke-linecap="round"/>'
        )
        if r.get("flagged") is not None:
            parts.append(
                f'<circle cx="{at(r["flagged"]):.1f}" cy="{y}" r="5" fill="{BG}" stroke="{MUTED}" '
                f'stroke-width="2"/>'
            )
        if r.get("control") is not None:
            cx = at(r["control"])
            parts.append(
                f'<rect x="{cx - 6:.1f}" y="{y - 6}" width="12" height="12" fill="{C_SAME}" '
                f'transform="rotate(45 {cx:.1f} {y})"/>'
            )
        parts.append(f'<circle cx="{at(r["new"]):.1f}" cy="{y}" r="8" fill="{col}"/>')
    more = len([r for r in items if r["outcome"] not in SKIPPED]) - len(shown)
    legend_y = top + len(shown) * 46 + 8
    if more > 0:
        parts.append(_t(px, legend_y, f"+ {more} more", 16, fill=MUTED))
        legend_y += 28
    legend = "○ flagged   ◆ old model again   ● new model"
    parts.append(_t(ax0 if show_complaints or use_groups else px, legend_y, legend, 15, fill=MUTED))
    parts.append(f'<line x1="64" y1="{H - 64}" x2="{W - 64}" y2="{H - 64}" stroke="{LINE}"/>')
    parts.append(
        _t(
            64,
            H - 30,
            "slash-bad · your personal benchmark for the things you actually care about",
            18,
            fill=MUTED,
        )
    )
    parts.append(
        _t(
            W - 64,
            H - 30,
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            18,
            fill=MUTED,
            anchor="end",
        )
    )
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
            try:
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
            except subprocess.TimeoutExpired:
                continue
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
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 28px; margin: 0 0 4px; } .muted { color: var(--muted); }
.card-img { width: 100%; max-width: 900px; border-radius: 12px; border: 1px solid var(--line);
            margin: 24px 0; display: block; }
.case { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 18px 20px; margin: 16px 0; }
.case h2 { font-size: 17px; margin: 0 0 12px; display: flex; gap: 12px; align-items: baseline; }
.pill { font: 600 13px/1 'SF Mono', Menlo, monospace; padding: 5px 8px; border-radius: 999px;
        border: 1px solid currentColor; white-space: nowrap; }
.better { color: var(--better); } .worse { color: var(--worse); } .same, .error { color: var(--same); }
.cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
@media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
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
        f"<div class='muted'>{s['better']} better · {s['same']} same · {s['worse']} worse{errors}"
        f" · judged blind by {esc(s['judge'] or judge)} · ${s['costUsd']:.2f}</div>",
    ]
    if card_name:
        parts.append(f"<img class='card-img' src='{esc(card_name)}' alt='scorecard'>")
    for r in _ordered(rows):
        o = r["outcome"]
        score = f"{fmt(r.get('reference'))} → {fmt(r.get('new'))}" if o not in SKIPPED else o
        parts.append(f"<section class='case'><h2><span class='pill {o}'>{MARK[o]} {score}</span>")
        parts.append(f"<span>{esc(r['complaint'])}</span></h2>")
        if o in SKIPPED:
            why = r.get("error") or "the judge couldn't find this complaint in the flagged message"
            parts.append(f"<div class='why'>{esc(why)}</div></section>")
            continue
        parts.append("<div class='cols'>")
        for label, model, val, resp, why in (
            ("flagged", r["oldModel"], r.get("flagged"), r.get("oldResponse"), r.get("oldWhy")),
            (
                "old model again",
                r["oldModel"],
                r.get("control"),
                r.get("controlResponse"),
                r.get("controlWhy") or r.get("controlError"),
            ),
            (
                "new model",
                r.get("newModel", ""),
                r.get("new"),
                r.get("newResponse"),
                r.get("newWhy"),
            ),
        ):
            parts.append(
                f"<div><div class='label'>{label} · {esc(model or '')} · {fmt(val)}/10</div>"
            )
            parts.append(f"<div class='resp'>{esc(resp or '')}</div>")
            parts.append(f"<div class='why'>{esc(why or '')}</div></div>")
        parts.append("</div>")
        if r.get("toolAttempts"):
            tried = "; ".join(_one_line(t, 80) for t in r["toolAttempts"][:3])
            parts.append(
                f"<div class='why'>the new model tried tools first (refused): {esc(tried)}</div>"
            )
        parts.append("</section>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_all(rows: list[dict], judge: str, show_complaints: bool = True) -> dict[str, Path]:
    s = summary(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
