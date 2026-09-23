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
MARK = {"better": "▲", "worse": "▼", "same": "●", "error": "✗", "unmatched": "?", "same-model": "="}
SKIPPED = ("error", "unmatched", "same-model")


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
    from slash_bad.compare import GROUP_BETTER_BY, outcome

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
                "outcome": outcome(ref, new, GROUP_BETTER_BY),
                "cases": [r["case"] for r in rs],
                "tally": Counter(r["outcome"] for r in rs),
            }
        )
    return out


def _ordered(rows: list[dict]) -> list[dict]:
    rank = {"better": 0, "worse": 1, "same": 2, "unmatched": 3, "same-model": 4, "error": 5}
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
        parts.append(f"the same on {n['same']}")
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
    for key, label in (("better", "better"), ("same", "same  "), ("worse", "worse ")):
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
            f"{c(BOLD)}{len(gs)} behaviours{c(RESET)}: better on {n['better']}, same on {n['same']}, "
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
        if r["outcome"] in ("unmatched", "same-model"):
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


def _models_text(names: list[str]) -> str:
    names = [pretty_model(n) for n in names]
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)


def _model_color(name: str) -> str:
    return "var(--claude)" if name.startswith("claude") else "var(--accent)"


def _score_hue(v: float) -> int:
    return round(120 * max(0.0, min(10.0, v)) / 10)


def _score(v: float | None, big: bool = False) -> str:
    if v is None:
        return "<span class='score none'>–</span>"
    cls = "score big" if big else "score"
    return f"<span class='{cls}' style='--h:{_score_hue(v)}'>{fmt(v)}<small>/10</small></span>"


def _pretty_date(when: datetime) -> str:
    return f"{when.day} {when.strftime('%b').replace('Sep', 'Sept')} {when.year}"


_CSS = """
:root { --bg:#f6f4f0; --fg:#1d1f23; --muted:#6b7079; --line:#e3dfd8; --card:#fff;
        --better:#15803d; --worse:#b91c1c; --same:#6b7079; --claude:#c2410c; --accent:#0f766e;
        --quote:#f1eee8; --l:38%; --tick:#1d1f23; --same-bar:#c4c7cc; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#0e1116; --fg:#eceae6; --muted:#8d93a0; --line:#262b34;
        --card:#151922; --better:#4ade80; --worse:#f87171; --same:#a1a6b0; --claude:#fb923c;
        --accent:#2dd4bf; --quote:#1c212b; --l:62%; --tick:#05070a;
        --same-bar:#555b66; } }
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

#scorecard { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
             padding: 36px 40px 28px; }
.sc-title { font: 800 44px/1 'SF Mono', Menlo, monospace; margin: 0; }
.sc-sub { color: var(--muted); margin: 10px 0 28px; max-width: 46em; }
.sc-head { font-size: 26px; line-height: 1.35; font-weight: 600; margin: 0 0 18px; }
.model { font-weight: 800; }
.tally { display: flex; gap: 36px; margin: 0 0 8px; flex-wrap: wrap; }
.tally div { font-size: 18px; font-weight: 600; }
.tally b { display: block; font-size: 56px; line-height: 1; font-weight: 800; }
.avg { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; margin: 26px 0 8px;
       padding: 16px 20px; background: var(--quote); border-radius: 12px; }
.avg .who { font-size: 14px; color: var(--muted); }
.avg .arrow { font-size: 26px; color: var(--muted); }
.cats { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin: 26px 0 8px; }
@media (max-width: 720px) { .cats { grid-template-columns: 1fr; } #scorecard { padding: 24px 18px; } }
.cats h3, .cats-same h3 { margin: 0 0 10px; font-size: 15px; text-transform: uppercase; letter-spacing: .06em; }
.cats ul, .cats-same ul { list-style: none; margin: 0; padding: 0; }
.cats li, .cats-same li { padding: 7px 0;
           border-top: 1px solid var(--line); }
.cat-row { display: flex; justify-content: space-between; gap: 12px; }
.cat-row > span:last-child { flex: none; white-space: nowrap; }
.h3-sub { font-size: 13px; color: var(--muted); margin: -6px 0 8px; }
.cat-row { font-weight: 500; }
.cat-meta { display: flex; align-items: center; gap: 10px; font-size: 12px; margin-top: 4px;
            color: var(--muted); opacity: .8; }
.cat-meta .count { flex: none; min-width: 7.5em; }
.cat-meta .bar { opacity: .65; }
.bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; flex: none;
       max-width: 45%; border: 1.5px solid var(--tick); }
.seg { flex: 1; } .seg + .seg { border-left: 1.5px solid var(--tick); }
.seg.better { background: var(--better); } .seg.same { background: var(--same-bar); }
.seg.worse { background: var(--worse); }
.cats-same { margin-top: 22px; }
.cats-same ul { columns: 2; column-gap: 28px; }
@media (max-width: 720px) { .cats-same ul { columns: 1; } }
.cats-same li { break-inside: avoid; }
.cats .none { color: var(--muted); padding: 7px 0; border-top: 1px solid var(--line); }
.unchanged { font-size: 14px; color: var(--muted); margin-top: 14px; }
.sc-foot { margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--line);
           font-size: 13px; color: var(--muted); }
.actions { display: flex; justify-content: flex-end; margin: 10px 0 40px; }
button { font: 600 14px/1 Inter, -apple-system, sans-serif; padding: 10px 14px; border-radius: 8px;
         border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }

h2.section { font-size: 20px; margin: 40px 0 4px; }
.case { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 18px 20px; margin: 16px 0; }
.complaint { font-size: 16px; font-weight: 600; margin: 0 0 14px; display: flex; gap: 10px; }
.complaint .tag { flex: none; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
.col-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
            margin-bottom: 6px; font-size: 14px; font-weight: 600; }
.resp { font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 300px;
        overflow: auto; border-left: 3px solid var(--line); padding: 4px 12px; }
.judge { margin-top: 10px; padding: 10px 12px; background: var(--quote); border-radius: 8px;
         font-size: 13.5px; }
.judge .label { display: block; font-size: 11px; font-weight: 700; text-transform: uppercase;
                letter-spacing: .07em; color: var(--muted); margin-bottom: 3px; }
.note { font-size: 13px; color: var(--muted); margin-top: 10px; }
details { margin-top: 32px; } summary { cursor: pointer; color: var(--muted); }
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


def _cat_item(i: dict) -> str:
    esc = html.escape
    scores = f"<span>{_score(i['reference'])} → {_score(i['new'])}</span>"
    if "n" not in i:
        return f"<li><div class='cat-row'><span>{esc(i['complaint'])}</span>{scores}</div></li>"
    t = i["tally"]
    segs = "".join(
        f"<span class='seg {k}'></span>" for k in ("better", "same", "worse") for _ in range(t[k])
    )
    parts = [
        f"{t[k]} {k if k != 'same' else 'the same'}" for k in ("better", "same", "worse") if t[k]
    ]
    return (
        f"<li><div class='cat-row'><span>{esc(i['complaint'])}</span>{scores}</div>"
        f"<div class='cat-meta'><span class='count'>{i['n']} transcript{'s' * (i['n'] != 1)}</span>"
        f"<span class='bar' style='width:{16 * i['n']}px'>{segs}</span>"
        f"<span class='muted'>{', '.join(parts)}</span></div></li>"
    )


def _scorecard(rows: list[dict], judge: str, when: datetime) -> str:
    esc = html.escape
    s = summary(rows)
    new = s["newModel"]
    olds = _models_text(s["oldModels"]) or "the old model"
    gs = groups(rows)
    items = gs if len(gs) >= 2 else [r for r in rows if r["outcome"] not in SKIPPED]
    by = {k: [i for i in _ordered(items) if i["outcome"] == k] for k in ("better", "worse", "same")}

    def cat_list(key: str) -> str:
        if not by[key]:
            return "<div class='none'>none</div>"
        return "<ul>" + "".join(_cat_item(i) for i in by[key]) + "</ul>"

    grouped = len(gs) >= 2
    unit = "behaviours" if grouped else "transcripts"

    def heading(key: str, label: str) -> str:
        n = len(by[key])
        sub = ""
        if grouped:
            k = sum(i["n"] for i in by[key])
            sub = f"<div class='h3-sub'>{n} of {len(gs)} behaviours · {k} transcript{'s' * (k != 1)}</div>"
        return f"<h3 class='{key}'>{label} {unit}</h3>{sub}"

    unchanged = ""
    if by["same"]:
        unchanged = f"<div class='cats-same'>{heading('same', 'Unchanged')}{cat_list('same')}</div>"
    legend = ""
    if grouped:
        legend = (
            "<div class='unchanged'>Each behaviour is a group of transcripts with the same kind of "
            "complaint. The bar splits its transcripts into <span class='better'>better</span>, "
            "<span class='same'>the same</span> and <span class='worse'>worse</span>; a behaviour "
            "with many transcripts is stronger evidence than one with a single transcript.</div>"
        )
    avg = ""
    if s["newMean"] is not None:
        old_mean = s["controlMean"] if s["controlled"] else s["flaggedMean"]
        old_who = olds if s["controlled"] else "the responses you flagged"
        avg = (
            "<div class='avg'>"
            f"<div><div class='who'>{esc(old_who)}</div>{_score(old_mean, big=True)}</div>"
            "<div class='arrow'>→</div>"
            f"<div><div class='who'>{esc(pretty_model(new))}</div>{_score(s['newMean'], big=True)}</div>"
            "<div class='who' style='margin-left:auto'>average judge score,<br>"
            "0 → problematic, 10 → fixed</div></div>"
        )
    return f"""
<div id="scorecard">
  <h1 class="sc-title">/bad</h1>
  <p class="sc-sub">A personal benchmark built from old transcripts where an AI model's reply
    annoyed me, each with a note saying why. Every one of those moments is re-sampled on a newer
    model and on the model that originally wrote it, and a judge that doesn't know which is which
    scores both replies against my note.</p>
  <p class="sc-head">Of {s["scored"]} transcripts,
    <span class="model" style="color:{_model_color(new)}">{esc(pretty_model(new))}</span> scored:</p>
  <div class="tally">
    <div class="better"><b>{s["better"]}</b>better</div>
    <div class="worse"><b>{s["worse"]}</b>worse</div>
    <div class="same"><b>{s["same"]}</b>the same</div>
  </div>
  <div class="muted">at resolving my old complaints about {esc(olds)}</div>
  {avg}
  <div class="cats">
    <div>{heading("better", "Better")}{cat_list("better")}</div>
    <div>{heading("worse", "Worse")}{cat_list("worse")}</div>
  </div>
  {unchanged}
  {legend}
  <div class="sc-foot">Judge: {esc(pretty_model(s["judge"] or judge))}, {_pretty_date(when)}</div>
</div>"""


def _case(r: dict) -> str:
    esc = html.escape
    o = r["outcome"]
    old_label = pretty_model(r["oldModel"])
    if r.get("control") is not None:
        old = (old_label, r["control"], r.get("controlResponse"), r.get("controlWhy"))
    else:
        old = (
            f"{old_label} (the reply you flagged)",
            r.get("flagged"),
            r.get("oldResponse"),
            r.get("oldWhy"),
        )
    new = (pretty_model(r.get("newModel", "")), r.get("new"), r.get("newResponse"), r.get("newWhy"))
    cols = ""
    for who, val, resp, why in (old, new):
        judge_note = (
            f"<div class='judge'><span class='label'>Judge's commentary</span>{esc(why)}</div>"
            if why
            else ""
        )
        cols += (
            f"<div><div class='col-head'><span>{esc(who)}</span>{_score(val)}</div>"
            f"<div class='resp'>{esc(resp or '')}</div>{judge_note}</div>"
        )
    note = ""
    if r.get("toolAttempts"):
        tried = "; ".join(_one_line(t, 80) for t in r["toolAttempts"][:3])
        note = f"<div class='note'>The new model tried to use tools first and was refused: {esc(tried)}</div>"
    behaviour = f" · {esc(r['behaviour'])}" if r.get("behaviour") else ""
    return (
        f"<section class='case'><p class='complaint'><span class='tag {o}'>{MARK[o]} {o}</span>"
        f"<span>“{esc(r['complaint'])}”<span class='muted' style='font-weight:400'>{behaviour}</span>"
        f"</span></p><div class='cols'>{cols}</div>{note}</section>"
    )


def html_report(rows: list[dict], judge: str, when: datetime | None = None) -> str:
    esc = html.escape
    when = when or datetime.now(timezone.utc)
    s = summary(rows)
    name = f"slash-bad-{s['newModel'] or 'model'}-{when:%Y-%m-%d}.png"
    scored = [r for r in _ordered(rows) if r["outcome"] not in SKIPPED]
    skipped = [r for r in rows if r["outcome"] in SKIPPED]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>/bad scorecard</title><style>{_CSS}</style></head><body><main>",
        _scorecard(rows, judge, when),
        f"<div class='actions'><button id='save-png' data-name='{esc(name)}'>Save as PNG</button></div>",
        f"<h2 class='section'>Every transcript ({len(scored)})</h2>",
        "<div class='muted'>Your note, then the old model's reply and the new model's reply to the "
        "same moment. Scores are out of 10; 10 means the complaint no longer applies.</div>",
        *(_case(r) for r in scored),
    ]
    if skipped:
        items = "".join(
            f"<li>“{esc(_one_line(r['complaint'], 140))}”: "
            f"{esc(r.get('error') or ('the flagged reply came from the new model' if r['outcome'] == 'same-model' else 'the judge could not find this complaint in the flagged reply'))}</li>"
            for r in skipped
        )
        parts.append(
            f"<details><summary>{len(skipped)} skipped</summary><ul>{items}</ul></details>"
        )
    parts.append(f"</main>{_SAVE_JS}</body></html>")
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
    page = data_path.with_suffix(".html")
    page.write_text(html_report(data["rows"], data["judge"], when))
    return page
