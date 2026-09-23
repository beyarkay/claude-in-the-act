"""Scoresheets: a terminal summary and an HTML page with a shareable scorecard."""

from __future__ import annotations

import html
import json
import os
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
MARK = {
    "better": "▲",
    "worse": "▼",
    "same": "●",
    "error": "✗",
    "unmatched": "?",
    "same-model": "=",
    "rejected": "⊘",
    "not-reproduced": "○",
}
SKIPPED = ("error", "unmatched", "same-model", "rejected", "not-reproduced")
WARN_ICON = (
    "<svg class='icon' viewBox='0 0 24 24' aria-hidden='true'><path d='M12 3.5 1.8 20.5h20.4Z' "
    "fill='none' stroke='currentColor' stroke-width='2' stroke-linejoin='round'/>"
    "<path d='M12 9.5v5' stroke='currentColor' stroke-width='2' stroke-linecap='round'/>"
    "<circle cx='12' cy='17.6' r='1.2' fill='currentColor'/></svg>"
)
LABEL = {"same": "no improvement", "not-reproduced": "couldn't reproduce"}
COUNT_ORDER = ("better", "worse", "same", "not-reproduced", "rejected")
WHY_SKIPPED = {
    "unmatched": "the judge couldn't find the complaint in the flagged reply",
    "same-model": "the flagged reply came from the model being tested",
    "not-reproduced": "couldn't reproduce: the old model, asked again, no longer showed the behaviour (scored 6+)",
}


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
        "sameModel": counts["same-model"],
        "rejected": counts["rejected"],
        "notReproduced": counts["not-reproduced"],
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
    from slash_bad.behaviours import goals
    from slash_bad.compare import GROUP_BETTER_BY, outcome

    wanted = goals()
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
                "goal": wanted.get(name, ""),
                "n": len(rs),
                "flagged": _mean([r["flagged"] for r in rs if r.get("flagged") is not None]),
                "control": _mean(ctl) if len(ctl) == len(rs) else None,
                "reference": ref,
                "new": new,
                "outcome": outcome(ref, new, GROUP_BETTER_BY),
                "cases": [r["case"] for r in rs],
                "tally": Counter(r["outcome"] for r in rs),
            }
        )
    return out


def _ordered(rows: list[dict]) -> list[dict]:
    rank = {k: i for i, k in enumerate(("better", "worse", "same", *SKIPPED))}
    return sorted(
        rows,
        key=lambda r: (rank[r["outcome"]], -abs((r.get("new") or 0) - (r.get("reference") or 0))),
    )


def _one_line(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def headline(rows: list[dict]) -> str:
    """One sentence you could post."""
    s = summary(rows)
    gs = groups(rows)
    olds = " and ".join(s["oldModels"][:2]) or "the old model"
    if len(gs) >= 2:
        n = Counter(g["outcome"] for g in gs)
        unit, total = f"behaviours I care about ({s['scored']} transcripts)", len(gs)
    else:
        n = Counter(r["outcome"] for r in rows if r["outcome"] not in SKIPPED)
        unit, total = "moments that annoyed me", s["scored"]
    parts = [f"better on {n['better']}"]
    if n["same"]:
        parts.append(f"no improvement on {n['same']}")
    if n["worse"]:
        parts.append(f"worse on {n['worse']}")
    joined = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return f"According to slash-bad, {s['newModel']} is {joined} of the {total} {unit}, compared with {olds}."


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
    out.append(f"  {s['scored']} transcripts:")
    width = max(s["better"], s["same"], s["worse"], 1)
    for key, label in (
        ("better", "better        "),
        ("same", "no improvement"),
        ("worse", "worse         "),
    ):
        bar = "■" * round(20 * s[key] / width) if s[key] else ""
        out.append(f"  {c(_color(key))}{MARK[key]} {label} {s[key]:>3}  {bar}{c(RESET)}")
    if s["errors"]:
        out.append(f"  {c(RED)}✗ errors {s['errors']:>3}{c(RESET)}  (see the report for why)")
    if s["newMean"] is not None:
        out += ["", "  average score, 10 = your complaint no longer applies:"]
        if s["controlMean"] is not None:
            out.append(f"    {olds[:32]:<32} {s['controlMean']:.1f}")
        elif s["flaggedMean"] is not None:
            out.append(f"    {'the responses you flagged':<32} {s['flaggedMean']:.1f}")
        out.append(f"    {s['newModel'][:32]:<32} {s['newMean']:.1f}")
    gs = groups(rows)
    if len(gs) >= 2:
        n = Counter(g["outcome"] for g in gs)
        out += [
            "",
            f"{c(BOLD)}{len(gs)} behaviours{c(RESET)}: better on {n['better']}, no improvement on {n['same']}, "
            f"worse on {n['worse']}  {c(DIM)}(old model → new model, transcripts){c(RESET)}",
        ]
        for g in _ordered(gs):
            out.append(
                f"  {c(_color(g['outcome']))}{MARK[g['outcome']]} {fmt(g['reference']):>4} → "
                f"{fmt(g['new']):<4}{c(RESET)} {g['complaint']} {c(DIM)}({g['n']} transcript{'s' * (g['n'] != 1)}){c(RESET)}"
            )
    out += [
        "",
        f"{c(BOLD)}per transcript{c(RESET)}  {c(DIM)}old model → new model{c(RESET)}",
    ]
    for r in _ordered(rows):
        mark = f"{c(_color(r['outcome']))}{MARK[r['outcome']]}"
        if r["outcome"] == "error":
            out.append(f"  {mark} error{c(RESET)}  {_one_line(r['complaint'], 70)}")
            out.append(f"      {c(DIM)}{_one_line(r.get('error', ''), 100)}{c(RESET)}")
            continue
        if r["outcome"] in SKIPPED:
            continue
        scores = f"{fmt(r['reference']):>3} → {fmt(r['new']):<3}"
        out.append(f"  {mark} {scores}{c(RESET)} {_one_line(r['complaint'], 76)}")
        out.append(f"      {c(DIM)}now: {_one_line(r.get('newResponse', ''), 96)}{c(RESET)}")
    notes = []
    if s["unmatched"]:
        notes.append(
            f"{s['unmatched']} case(s) skipped: the judge couldn't find the complaint in the flagged "
            "message or the ones just before it, so a replay wouldn't measure it."
        )
    if s["notReproduced"]:
        notes.append(
            f"{s['notReproduced']} case(s) skipped: asked again, the old model didn't repeat "
            "the problem, so the replay can't show whether the new model fixed it."
        )
    if s["rejected"]:
        notes.append(
            f"{s['rejected']} case(s) skipped: the judge said a reply couldn't be judged "
            "against the complaint (see the report for why)."
        )
    if s["sameModel"]:
        notes.append(
            f"{s['sameModel']} case(s) skipped: the flagged response came from {s['newModel'] or 'the model being tested'} itself."
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
    if s["scored"]:
        out += ["", f"{c(BOLD)}{headline(rows)}{c(RESET)}"]
    return "\n".join(out)


def pretty_model(name: str) -> str:
    """``claude-opus-5-5`` → ``Claude Opus 5.5``, ``gpt-6-sol`` → ``GPT-6 Sol``."""
    parts = [p for p in (name or "").split("-") if not (p.isdigit() and len(p) == 8)]
    if parts[:1] == ["claude"] and len(parts) > 1:
        version = ".".join(p for p in parts[2:] if p.replace(".", "").isdigit())
        rest = [p.title() for p in parts[2:] if not p.replace(".", "").isdigit()]
        return " ".join(["Claude", parts[1].title(), version, *rest]).replace("  ", " ").strip()
    if parts[:1] == ["gpt"] and len(parts) > 1:
        return " ".join([f"GPT-{parts[1]}", *(p.title() for p in parts[2:])])
    return name


def _score_hue(v: float) -> int:
    return round(120 * max(0.0, min(10.0, v)) / 10)


def _score(v: float | None, big: bool = False) -> str:
    if v is None:
        return "<span class='score none'>–</span>"
    cls = "score big" if big else "score"
    text = ("0" if round(v, 1) == 0 else f"{v:.1f}") if big else fmt(v)
    return f"<span class='{cls}' style='--h:{_score_hue(v)}'>{text}<small>/10</small></span>"


def _pretty_date(when: datetime) -> str:
    return f"{when.day} {when.strftime('%b').replace('Sep', 'Sept')} {when.year}"


_CSS = """
:root { --bg:#f4f6f8; --fg:#20242a; --muted:#626b78; --line:#d9dee7; --card:#ffffff;
        --better:#18794e; --worse:#b63e48; --same:#8a6500; --warn:#8a6500; --claude:#d97757;
        --accent:#0f766e; --quote:#f2f4f7; --l:38%; --tick:#20242a; --same-bar:#e6d49a; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#0d1117; --fg:#f0f2f5; --muted:#a3acb9; --line:#303844;
        --card:#151a22; --better:#54b87a; --worse:#e06c75; --same:#c9a227; --warn:#c9a227;
        --claude:#d97757; --accent:#2dd4bf; --quote:#1b222c; --l:62%; --tick:#0d1117;
        --same-bar:#6b5a1e; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.55 Inter, -apple-system, 'Helvetica Neue', sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 80px; }
.muted { color: var(--muted); }
.score { font: 700 15px/1 'SF Mono', Menlo, monospace; color: hsl(var(--h) 75% var(--l));
         white-space: nowrap; }
.score small { font-size: .7em; font-weight: 500; opacity: .75; }
.score.big { font-size: 34px; }
.score.none { color: var(--muted); }
.better { color: var(--better); } .worse { color: var(--worse); } .same { color: var(--same); }
.not-reproduced { color: var(--warn); } .rejected { color: var(--muted); }
summary { list-style: none; } summary::-webkit-details-marker { display: none; }
details > summary::before { content: '▸'; color: var(--muted); width: 1em; flex: none; }
details[open] > summary::before { content: '▾'; }

#scorecard { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
             padding: 32px 40px 30px; font-variant-numeric: tabular-nums; }
.sc-top { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }
.sc-title { font: 750 26px/30px 'SF Mono', Menlo, monospace; }
.sc-sub { color: var(--muted); font-size: 14px; line-height: 20px; margin-top: 4px; }
.sc-meta { color: var(--muted); font-size: 12px; line-height: 18px; text-align: right; white-space: nowrap; }
.sc-head { font-size: 28px; line-height: 34px; font-weight: 700; margin: 26px 0 4px; }
.model { white-space: nowrap; }
.logo { height: .8em; width: .8em; vertical-align: -.02em; margin-right: .3em; }
.sc-compare { color: var(--muted); font-size: 15px; margin-bottom: 22px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 720px) { .panels { grid-template-columns: 1fr; } #scorecard { padding: 24px 18px; } }
.panel { padding: 18px 22px; background: var(--quote); border-radius: 12px;
         display: flex; flex-direction: column; justify-content: space-between; gap: 10px; }
.panel-label { font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: .06em;
               color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
.panel-label span { font-weight: 400; text-transform: none; letter-spacing: 0; }
.avg { display: grid; grid-template-columns: 1fr auto 1fr; align-items: end; text-align: center; }
.avg .who { font-size: 13px; color: var(--muted); margin-bottom: 2px; }
.avg .big { font-size: 38px; line-height: 40px; font-weight: 700; }
.avg .big small { font-size: 21px; font-weight: 400; color: var(--muted); }
.avg .arrow { font-size: 22px; color: var(--muted); padding-bottom: 6px; }
.panel-foot { font-size: 13px; color: var(--muted); text-align: center; }
.panel-foot span { font-weight: 650; }
.tally { display: grid; grid-template-columns: repeat(3, 1fr); text-align: center; }
.tally div { font-size: 13px; }
.tally b { display: block; font-size: 38px; line-height: 40px; font-weight: 700; }
.findings { margin-top: 30px; }
.f-head, .finding { display: grid; grid-template-columns: minmax(0, 1fr) 4.5rem 8.5rem minmax(0, .85fr);
                    column-gap: 24px; align-items: baseline; }
.f-head { font-size: 12px; color: var(--muted); padding-bottom: 6px; }
.f-head span:nth-child(2) { text-align: right; }
.f-group { margin-top: 18px; }
.f-group-label { font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: .06em;
                 padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.f-group-label span { color: var(--muted); font-weight: 400; text-transform: none; letter-spacing: 0; }
.finding { padding: 12px 0; border-bottom: 1px solid var(--line); }
.f-name { font-size: 15px; line-height: 21px; font-weight: 650; }
.f-delta { font-size: 21px; font-weight: 700; text-align: right; white-space: nowrap; }
.f-scores { font-size: 13px; white-space: nowrap; }
.f-scores small { color: var(--muted); }
.f-evidence { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; flex: none;
       border: 1px solid var(--tick); opacity: .8; }
.seg { flex: 1; } .seg + .seg { border-left: 1px solid var(--tick); }
.seg.better { background: var(--better); } .seg.same { background: var(--same-bar); }
.seg.worse { background: var(--worse); }
.actions { display: flex; justify-content: flex-end; margin: 10px 0 40px; }
button { font: 600 14px/1 Inter, -apple-system, sans-serif; padding: 10px 14px; border-radius: 8px;
         border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }

h2.section { font-size: 20px; margin: 40px 0 4px; }
.behaviour { margin: 22px 0; }
.behaviour > summary { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
                       cursor: pointer; padding: 6px 0; border-bottom: 1px solid var(--line); }
.behaviour > summary h3 { margin: 0; font-size: 19px; }
.counts { font-size: 14px; font-weight: 600; }
.case { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 12px 16px; margin: 10px 0; }
.case > summary { display: grid; grid-template-columns: 1em 17em minmax(0, 1fr);
                  align-items: start; column-gap: 12px; cursor: pointer; }
.status { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }
.tag.flag { font-weight: 400; display: inline-flex; align-items: center; gap: 5px; }
.icon { width: 1.05em; height: 1.05em; flex: none; }
.case-title { flex: 1; min-width: 0; }
.case[open] .short, .case:not([open]) .full { display: none; }
.case-scores { font-size: 12px; line-height: 1.2; color: var(--muted); white-space: nowrap; }
.case[open] > summary { margin-bottom: 8px; }
.tag { font: 600 13px/1 'SF Mono', Menlo, monospace; white-space: nowrap; }
.tag.pill { padding: 4px 8px; border-radius: 999px; }
.case-scores .score { font-size: 12.5px; }
.complaint { color: var(--muted); font-size: 13.5px; margin: 4px 0 14px; }
.shared { border: 1px dashed var(--line); border-radius: 10px; padding: 8px 12px; margin-bottom: 12px; }
.shared > summary { display: flex; gap: 8px; align-items: baseline; cursor: pointer; }
.shared-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
                color: var(--muted); }
.shared-peek { color: var(--muted); font-size: 13px; min-width: 0; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
.then { font-size: 12.5px; color: var(--muted); margin: 6px 0 0 4px; }
.bubble { border-radius: 14px; padding: 10px 14px; font-size: 14px; overflow-wrap: anywhere; }
.bubble.user { background: var(--quote); max-height: 320px; overflow: auto; }
.bubble.bot { border: 1px solid var(--line); height: 420px; overflow: auto; }
.bubble-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
                color: var(--muted); margin-bottom: 6px; }
.replies { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
           grid-template-rows: auto auto auto; column-gap: 14px; }
@media (max-width: 900px) { .replies { grid-template-columns: 1fr; } }
.reply { display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 8px; min-width: 0; }
.reply-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
              font-size: 13.5px; }
.reply .why { background: var(--quote); border-radius: 10px; padding: 10px 12px; font-size: 13.5px; }
.reply .label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
                color: var(--muted); margin-right: 6px; }
.rej { font: 700 13px/1 'SF Mono', Menlo, monospace; color: var(--muted); }
.md { white-space: pre-wrap; }
.md.rendered { white-space: normal; }
.md.rendered > :first-child { margin-top: 0; } .md.rendered > :last-child { margin-bottom: 0; }
.md pre { max-width: 100%; overflow: auto; background: var(--quote); padding: 8px 10px; border-radius: 6px; font-size: 12.5px; }
.md code { font-size: .92em; } .md table { border-collapse: collapse; font-size: 12.5px; }
.md td, .md th { border: 1px solid var(--line); padding: 3px 6px; }
.md h1, .md h2, .md h3 { font-size: 15px; margin: 12px 0 4px; }
mark { background: color-mix(in srgb, var(--accent) 26%, transparent); color: inherit; border-radius: 3px; }
.reason { font-size: 12px; color: var(--muted); }
.section-count { font-size: 14px; margin-bottom: 2px; }
.counts .total { color: var(--muted); font-weight: 400; }
@media (max-width: 700px) {
  #scorecard { padding: 22px 16px; }
  .sc-top { flex-direction: column; gap: 6px; } .sc-meta { text-align: left; }
  .f-head { display: none; }
  .finding { grid-template-columns: minmax(0, 1fr) auto; row-gap: 4px; }
  .f-scores, .f-evidence { grid-column: 1 / -1; }
  .f-evidence { flex-wrap: wrap; }
  .behaviour > summary h3 { min-width: 0; overflow-wrap: anywhere; }
  .case > summary { grid-template-columns: 1rem minmax(0, 1fr); row-gap: 6px; }
  .case > summary .status, .case > summary .case-title { grid-column: 2; }
  .case-title, .reply, .replies, .md { min-width: 0; }
  .replies { display: block; }
  .reply { display: block; margin-block: 14px; }
  .bubble.bot { height: auto; max-height: none; }
  .shared-peek { white-space: normal; }
}
.note { font-size: 13px; color: var(--muted); margin-top: 10px; }
.not-replayed { margin-top: 32px; } .not-replayed > summary { cursor: pointer; color: var(--muted); }
"""

_MD_JS = """
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script>
function plain(s) { return s.replace(/[*`]/g, '').replace(/\\s+/g, ' ').trim(); }
function highlight(root, quotes) {
  // Whitespace-collapsed text of every text node, remembering where each char came from.
  const chars = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    for (let i = 0; i < node.nodeValue.length; i++) {
      const c = /\\s/.test(node.nodeValue[i]) ? ' ' : node.nodeValue[i];
      if (c === ' ' && chars.length && chars[chars.length - 1].c === ' ') continue;
      chars.push({c, node, i});
    }
  }
  const full = chars.map(x => x.c).join('');
  for (const q of quotes) {
    let want = plain(q);
    if (want.length < 4) continue;
    let at = full.indexOf(want);
    if (at < 0 && want.length > 40) { want = want.slice(0, 40); at = full.indexOf(want); }
    if (at < 0) continue;
    const spans = new Map();
    for (const x of chars.slice(at, at + want.length)) {
      const s = spans.get(x.node) || {start: x.i, end: x.i};
      s.end = x.i; spans.set(x.node, s);
    }
    for (const [n, s] of [...spans].reverse()) {
      const range = document.createRange();
      range.setStart(n, s.start); range.setEnd(n, s.end + 1);
      range.surroundContents(document.createElement('mark'));
    }
    return highlight(root, quotes.filter(x => x !== q));
  }
}
for (const el of document.querySelectorAll('.md')) {
  const src = el.textContent;
  if (window.marked && window.DOMPurify) {
    el.innerHTML = DOMPurify.sanitize(marked.parse(src, {breaks: true}));
    el.classList.add('rendered');
  }
  try { highlight(el, JSON.parse(el.dataset.evidence || '[]')); } catch (e) {}
}
</script>
"""

_SAVE_JS = """
<script src="https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.js"></script>
<script>
const btn = document.getElementById('save-png');
if (!window.htmlToImage) btn.hidden = true;
btn.addEventListener('click', async () => {
  const card = document.getElementById('scorecard');
  const bg = getComputedStyle(document.body).backgroundColor;
  try {
    const url = await htmlToImage.toPng(card, {pixelRatio: 2, backgroundColor: bg});
    const a = document.createElement('a');
    a.href = url; a.download = btn.dataset.name; a.click();
  } catch (e) { btn.textContent = 'Could not save the image'; }
});
</script>
"""


def _signed(d: float, whole: bool = False) -> str:
    """``+2.0``, ``−1.5``, ``0.0`` (``+2``, ``−1``, ``0`` with ``whole``)."""
    if round(d, 1) == 0:
        return "0" if whole else "0.0"
    value = fmt(abs(d)) if whole else f"{abs(d):.1f}"
    return f"{'+' if d > 0 else '−'}{value}"


def _logo(model: str) -> str:
    if not model.startswith("claude"):
        return ""
    return (Path(__file__).parent / "assets" / "claude-symbol.svg").read_text().strip()


def _finding(i: dict) -> str:
    """One row of the card's findings table: a behaviour, or a transcript without one."""
    esc = html.escape
    d = i["new"] - i["reference"]
    if "n" in i:
        name = i["complaint"] if i["outcome"] == "worse" else i.get("goal") or i["complaint"]
        t = i["tally"]
        segs = "".join(
            f"<span class='seg {k}'></span>"
            for k in ("better", "same", "worse")
            for _ in range(t[k])
        )
        parts = [f"{t[k]} {LABEL.get(k, k)}" for k in ("better", "same", "worse") if t[k]]
        evidence = (
            f"<span class='bar' style='width:{min(160, 12 * i['n'])}px'>{segs}</span>"
            f"<span>{i['n']} transcript{'s' * (i['n'] != 1)} · {', '.join(parts)}</span>"
        )
    else:
        name, evidence = f"“{_one_line(i['complaint'], 90)}”", "<span>1 transcript</span>"
    return (
        f"<div class='finding'><span class='f-name'>{esc(name)}</span>"
        f"<span class='f-delta {i['outcome']}'>{_signed(d)}</span>"
        f"<span class='f-scores'>{i['reference']:.1f} → {i['new']:.1f}<small> /10</small></span>"
        f"<span class='f-evidence'>{evidence}</span></div>"
    )


def _short_models(names: list[str]) -> list[str]:
    pretty = [pretty_model(n) for n in names]
    if all(n.startswith("Claude ") for n in pretty):
        pretty = [n.removeprefix("Claude ") for n in pretty]
    return pretty


def _scorecard(rows: list[dict], judge: str, when: datetime) -> str:
    esc = html.escape
    s = summary(rows)
    new = s["newModel"]
    gs = groups(rows)
    grouped = len(gs) >= 2
    items = gs if grouped else [r for r in rows if r["outcome"] not in SKIPPED]
    by = {k: [i for i in _ordered(items) if i["outcome"] == k] for k in ("better", "worse", "same")}
    unit = "behaviours" if grouped else "transcripts"

    def group(key: str, label: str) -> str:
        if not by[key]:
            return ""
        n = len(by[key])
        extra = ""
        if grouped:
            k = sum(i["n"] for i in by[key])
            extra = f" · {k} transcript{'s' * (k != 1)}"
        return (
            f"<div class='f-group'><div class='f-group-label {key}'>{label}"
            f"<span> · {n} {unit[:-1] if n == 1 else unit}{extra}</span></div>"
            + "".join(_finding(i) for i in by[key])
            + "</div>"
        )

    old_names = _short_models(s["oldModels"])
    avg = ""
    if s["newMean"] is not None:
        old_mean = s["controlMean"] if s["controlled"] else s["flaggedMean"]
        old_who = " & ".join(old_names) if len(old_names) <= 2 else "Older models"
        if not s["controlled"]:
            old_who = "Your flagged replies"
        gain = s["newMean"] - old_mean
        avg = (
            "<div class='panel'><div class='panel-label'>Average judge score</div>"
            "<div class='avg'>"
            f"<div><div class='who'>{esc(old_who)}</div><div class='big'>{old_mean:.1f}"
            "<small>/10</small></div></div><div class='arrow'>→</div>"
            f"<div><div class='who'>{esc(pretty_model(new))}</div><div class='big'>"
            f"{s['newMean']:.1f}<small>/10</small></div></div></div>"
            f"<div class='panel-foot'><span class='{outcome_word(gain)}'>{_signed(gain)} points"
            "</span> on average · 0 = problematic, 10 = fixed</div></div>"
        )
    improved = len(by["better"])
    return f"""
<div id="scorecard">
  <div class="sc-top">
    <div><div class="sc-title">/bad</div>
      <div class="sc-sub">Old replies that annoyed me, re-sampled on a newer model and judged
        blind against my complaint.</div></div>
    <div class="sc-meta">Judge: {esc(pretty_model(s["judge"] or judge))}<br>{_pretty_date(when)}</div>
  </div>
  <h1 class="sc-head">{_logo(new)}<span class="model">{esc(pretty_model(new))}</span>
    improved on {improved} of {len(items)} {unit}</h1>
  <div class="sc-compare">compared to older models ({esc(", ".join(old_names))}) ·
    {s["scored"]} transcripts</div>
  <div class="panels">
    {avg}
    <div class="panel"><div class="panel-label">Transcript outcomes<span>n = {s["scored"]}</span></div>
      <div class="tally">
        <div class="better"><b>{s["better"]}</b>↑ better</div>
        <div class="worse"><b>{s["worse"]}</b>↓ worse</div>
        <div class="same"><b>{s["same"]}</b>– no improvement</div>
      </div></div>
  </div>
  <div class="findings">
    <div class="f-head"><span>{unit.capitalize()}</span><span>Change</span><span>Mean score</span>
      <span>Evidence</span></div>
    {group("better", "Improved")}{group("worse", "Regressed")}{group("same", "No improvement")}
  </div>
</div>"""


def outcome_word(d: float) -> str:
    return "better" if d > 0 else "worse" if d < 0 else "same"


def _shared(case_id: str) -> str:
    """The conversation every model saw: the last user message and what followed it."""
    from slash_bad import compare
    from slash_bad.store import find_case

    esc = html.escape
    try:
        case = find_case(case_id)
        steps = compare.HARNESSES[case.harness].steps(case.context)
    except (KeyError, OSError, ValueError):
        return ""
    users = [i for i, s in enumerate(steps) if s["kind"] == "user"]
    if not users:
        return ""
    prompt = steps[users[-1]]["text"].strip()
    after = steps[users[-1] + 1 :]
    calls = Counter(s.get("name") or "tool" for s in after if s["kind"] == "tool_call")
    said = sum(1 for s in after if s["kind"] == "assistant")
    work = []
    if calls:
        n = sum(calls.values())
        names = ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in calls.most_common(4))
        work.append(f"{n} tool call{'s' * (n != 1)} ({names})")
    if said:
        work.append(f"{said} assistant message{'s' * (said != 1)}")
    then = f"<div class='then'>then {' and '.join(work)}</div>" if work else ""
    return (
        "<details class='shared'><summary><span class='shared-label'>Context</span>"
        f"<span class='shared-peek'>what you asked: “{esc(_one_line(prompt, 90))}”"
        + (f" · then {' and '.join(work)}" if work else "")
        + "</span></summary>"
        "<div class='muted' style='font-size:12.5px;margin:6px 0'>The conversation up to here "
        "is identical for all three replies below; only the last reply was re-sampled.</div>"
        f"<div class='bubble user'><div class='bubble-label'>You asked</div>"
        f"<div class='md'>{esc(prompt[:8000])}</div></div>{then}</details>"
    )


def _reply(title: str, model: str, score, why: str, rejected: str, response: str, evidence) -> str:
    esc = html.escape
    verdict = "<span class='rej'>rejected</span>" if rejected else _score(score)
    note = rejected or why
    return (
        "<div class='reply'>"
        f"<div class='reply-head'><span><b>{esc(title)}</b>"
        f"<span class='muted'> · {esc(pretty_model(model))}</span></span>{verdict}</div>"
        f"<div class='bubble bot'><div class='bubble-label'>Model's response</div>"
        f"<div class='md' data-evidence='{esc(json.dumps(evidence or []))}'>"
        f"{esc(response or '')}</div></div>"
        f"<div class='why'><span class='label'>Judge</span>{esc(note or '–')}</div></div>"
    )


def _case(r: dict) -> str:
    esc = html.escape
    o = r["outcome"]
    rejected = r.get("rejected") or ""

    def rej(who: str) -> str:
        return rejected.split(": ", 1)[1] if rejected.startswith(who) else ""

    replies = [
        _reply(
            "Original reply",
            r["oldModel"],
            r.get("flagged"),
            r.get("oldWhy", ""),
            rej("the flagged reply"),
            r.get("oldResponse", ""),
            r.get("oldEvidence"),
        ),
        _reply(
            "Old model, asked again",
            r["oldModel"],
            r.get("control"),
            r.get("controlWhy") or r.get("controlError") or "",
            rej("the old model's re-run"),
            r.get("controlResponse", ""),
            r.get("controlEvidence"),
        ),
        _reply(
            "New model",
            r.get("newModel", ""),
            r.get("new"),
            r.get("newWhy", ""),
            rej("the new model's reply"),
            r.get("newResponse", ""),
            r.get("newEvidence"),
        ),
    ]
    note = ""
    if r.get("toolAttempts"):
        tried = "; ".join(_one_line(a, 80) for a in r["toolAttempts"][:3])
        note = f"<div class='note'>The new model tried to use tools first and was refused: {esc(tried)}</div>"
    scores = " · ".join(
        f"<span>{who} {_score(r.get(k))}</span>"
        for who, k in (("Original", "flagged"), ("Asked again", "control"), ("New", "new"))
    )
    reason = {
        "not-reproduced": "the old model, asked again, no longer did it",
        "rejected": "the judge rejected "
        + (r.get("rejected") or "a reply").split(": ", 1)[0].removeprefix("the "),
    }.get(o, "")
    tag = f"{MARK[o]} {esc(LABEL.get(o, o))}"
    if o in ("not-reproduced", "rejected"):
        tag = f"{WARN_ICON} {esc(LABEL.get(o, o))}"
    style = ""
    ref, new = r.get("reference"), r.get("new")
    if o in ("better", "same", "worse") and ref is not None and new is not None:
        d = new - ref
        tag += f" {_signed(d, whole=True)}"
        pct = 18 if o == "same" else round(12 + 48 * min(1.0, abs(d) / 9))
        style = f" style='background: color-mix(in srgb, var(--{o}) {pct}%, transparent)'"
    kind = "pill" if style else "flag"
    return (
        f"<details class='case'><summary><span class='status'><span class='tag {kind} {o}'{style}>"
        f"{tag}</span><span class='case-scores'>{scores}</span>"
        + (f"<span class='reason'>{esc(reason)}</span>" if reason else "")
        + "</span>"
        f"<span class='case-title'><span class='short'>{esc(_one_line(r['complaint'], 160))}</span>"
        f"<span class='full'><span class='muted'>User complaint: </span>{esc(r['complaint'])}"
        "</span></span></summary>"
        f"{_shared(r['case'])}<div class='replies'>{''.join(replies)}</div>{note}</details>"
    )


def _behaviour_sections(rows: list[dict]) -> str:
    esc = html.escape
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("behaviour") or "no behaviour assigned", []).append(r)
    rank = {g["complaint"]: ("better", "worse", "same").index(g["outcome"]) for g in groups(rows)}
    out = []
    for name, rs in sorted(by.items(), key=lambda kv: (rank.get(kv[0], 3), -len(kv[1]))):
        n = Counter(r["outcome"] for r in rs)
        counts = ", ".join(
            f"<span class='{k}'>{n[k]} {LABEL.get(k, k)}</span>" for k in COUNT_ORDER if n[k]
        )
        out.append(
            f"<details class='behaviour' open><summary><h3>{esc(name)}</h3>"
            f"<span class='counts'><span class='total'>{len(rs)} transcript"
            f"{'s' * (len(rs) != 1)}:</span> {counts}</span></summary>"
            + "".join(_case(r) for r in _ordered(rs))
            + "</details>"
        )
    return "".join(out)


def html_report(rows: list[dict], judge: str, when: datetime | None = None) -> str:
    esc = html.escape
    when = when or datetime.now(timezone.utc)
    s = summary(rows)
    name = f"slash-bad-{s['newModel'] or 'model'}-{when:%Y-%m-%d}.png"
    replayed = [r for r in rows if r["outcome"] not in ("error", "unmatched", "same-model")]
    listed = [r for r in rows if r not in replayed]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>/bad scorecard</title><style>{_CSS}</style></head><body><main>",
        _scorecard(rows, judge, when),
        f"<div class='actions'><button id='save-png' data-name='{esc(name)}'>Save as PNG</button></div>",
        "<h2 class='section'>Every transcript, by behaviour</h2>",
        f"<div class='section-count'>{len(replayed)} replayed: {s['scored']} counted, "
        f"{s['notReproduced']} couldn't reproduce"
        + (f", {s['rejected']} rejected" if s["rejected"] else "")
        + (f" · {len(listed)} more not replayed (listed at the end)" if listed else "")
        + "</div>",
        "<div class='muted'>Each transcript shows three judge scores out of 10 (0 = problematic, "
        "10 = fixed): the original reply, the old model asked again, and the new model. The "
        "change in the badge is the new model minus the old model asked again.</div>",
        _behaviour_sections(replayed),
    ]
    if listed:
        items = "".join(
            f"<li>“{esc(_one_line(r['complaint'], 140))}”: "
            f"{esc(r.get('error') or WHY_SKIPPED.get(r['outcome'], r['outcome']))}</li>"
            for r in listed
        )
        parts.append(
            f"<details class='not-replayed'><summary>{len(listed)} not replayed</summary>"
            f"<ul>{items}</ul></details>"
        )
    parts.append(f"</main>{_MD_JS}{_SAVE_JS}</body></html>")
    return "\n".join(parts)


def write_all(rows: list[dict], judge: str) -> dict[str, Path]:
    when = datetime.now(timezone.utc)
    s = summary(rows)
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in s["newModel"] or "model")
    base = REPORTS_DIR / f"{when:%Y%m%d-%H%M%S}-{safe}"
    base.parent.mkdir(parents=True, exist_ok=True)
    data = base.with_suffix(".json")
    data.write_text(
        json.dumps(
            {"judge": judge, "runAt": when.isoformat(), "summary": s, "rows": rows}, indent=2
        )
    )
    page = base.with_suffix(".html")
    page.write_text(html_report(rows, judge, when))
    return {"html": page, "json": data}


def rerender(data_path: Path) -> Path:
    """Rebuild the HTML page from a saved ``.json`` report."""
    data = json.loads(data_path.read_text())
    when = (
        datetime.fromisoformat(data["runAt"])
        if data.get("runAt")
        else datetime.fromtimestamp(data_path.stat().st_mtime, timezone.utc)
    )
    from slash_bad.compare import classify

    page = data_path.with_suffix(".html")
    page.write_text(html_report([classify(r) for r in data["rows"]], data["judge"], when))
    return page
