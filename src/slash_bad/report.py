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
LABEL = {"same": "no improvement", "not-reproduced": "couldn't reproduce"}
COUNT_ORDER = ("better", "worse", "same", "not-reproduced", "rejected")
WHY_SKIPPED = {
    "unmatched": "the judge couldn't find the complaint in the flagged reply",
    "same-model": "the flagged reply came from the model being tested",
    "not-reproduced": "couldn't reproduce: the old model, asked again, scored well above the original",
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
        --better:#15803d; --worse:#b91c1c; --same:#6b7079; --claude:#d97757; --accent:#0f766e;
        --quote:#f1eee8; --l:38%; --tick:#1d1f23; --same-bar:#c4c7cc; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#0e1116; --fg:#eceae6; --muted:#8d93a0; --line:#262b34;
        --card:#151922; --better:#4ade80; --worse:#f87171; --same:#a1a6b0; --claude:#d97757;
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
.sc-sub { color: var(--muted); margin: 8px 0 24px; }
.sc-head { font-size: 26px; line-height: 1.35; font-weight: 600; margin: 0 0 14px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 720px) { .panels { grid-template-columns: 1fr; } }
.panel { padding: 14px 20px 16px; background: var(--quote); border-radius: 12px; }
.panel-label { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 8px;
               display: flex; justify-content: space-between; gap: 8px; }
.panel-label span { font-weight: 400; }
.model { font-weight: 800; white-space: nowrap; }
.logo { height: .9em; width: .9em; vertical-align: -.08em; margin-right: .25em; }
.scale { text-align: center; font-size: 12px; color: var(--muted); margin-top: 6px; }
.tally { display: flex; justify-content: space-around; gap: 16px; }
.tally div { font-size: 15px; font-weight: 600; text-align: center; }
.tally b { display: block; font-size: 40px; line-height: 1.05; font-weight: 800; }
.avg { display: flex; align-items: center; justify-content: space-around; gap: 16px; }
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
.delta { display: flex; flex-direction: column; align-items: center; min-width: 8.5em; }
.delta b { font: 800 24px/1.1 'SF Mono', Menlo, monospace; }
.delta small .score { font-size: 11.5px; font-weight: 600; }
.delta small { color: var(--muted); font-size: 11.5px; }
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
.sc-foot { margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--line);
           font-size: 13px; color: var(--muted); }
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
.case > summary { display: flex; align-items: baseline; gap: 12px; cursor: pointer; }
.case-title { flex: 1; min-width: 0; }
.case-scores { font: 600 13px/1 'SF Mono', Menlo, monospace; color: var(--muted); white-space: nowrap; }
.case[open] > summary { margin-bottom: 8px; }
.tag { font: 600 13px/1 'SF Mono', Menlo, monospace; white-space: nowrap; }
.tag.pill { padding: 4px 8px; border-radius: 999px; }
.case-scores .score { font-size: 13px; }
.tag.rejected, .tag.not-reproduced { color: var(--muted); }
.complaint { color: var(--muted); font-size: 13.5px; margin: 4px 0 14px; }
.shared { border: 1px dashed var(--line); border-radius: 10px; padding: 10px 12px; }
.shared-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
                color: var(--muted); margin-bottom: 8px; }
.shared summary { color: inherit; }
.then { font-size: 12.5px; color: var(--muted); margin: 6px 0 0 4px; }
.fork { font-size: 12.5px; color: var(--muted); margin-top: 8px; text-align: center; }
.bubble { border-radius: 14px; padding: 10px 14px; font-size: 14px; overflow-wrap: anywhere; }
.bubble.user { background: var(--quote); margin-left: auto; max-width: 85%; width: fit-content; }
.bubble.bot { border: 1px solid var(--line); max-height: 460px; overflow: auto; }
.replies { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }
@media (max-width: 900px) { .replies { grid-template-columns: 1fr; } }
.reply { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.verdict { background: var(--quote); border-radius: 10px; padding: 10px 12px; font-size: 13.5px; }
.verdict-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.verdict .why { margin-top: 6px; }
.verdict .label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
                  color: var(--muted); margin-right: 6px; }
.rej { font: 700 13px/1 'SF Mono', Menlo, monospace; color: var(--muted); }
.md { white-space: pre-wrap; }
.md.rendered { white-space: normal; }
.md.rendered > :first-child { margin-top: 0; } .md.rendered > :last-child { margin-bottom: 0; }
.md pre { overflow: auto; background: var(--quote); padding: 8px 10px; border-radius: 6px; font-size: 12.5px; }
.md code { font-size: .92em; } .md table { border-collapse: collapse; font-size: 12.5px; }
.md td, .md th { border: 1px solid var(--line); padding: 3px 6px; }
.md h1, .md h2, .md h3 { font-size: 15px; margin: 12px 0 4px; }
mark { background: color-mix(in srgb, var(--worse) 28%, transparent); color: inherit; border-radius: 3px; }
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


def _logo(model: str) -> str:
    if not model.startswith("claude"):
        return ""
    return (Path(__file__).parent / "assets" / "claude-symbol.svg").read_text().strip()


def _cat_item(i: dict) -> str:
    esc = html.escape
    d = i["new"] - i["reference"]
    sign = "+" if d > 0 else "−" if d < 0 else "±"
    scores = (
        f"<span class='delta'><b class='{i['outcome']}'>{sign}{abs(d):.1f}</b>"
        f"<small>{_score(i['reference'])} → {_score(i['new'])}</small></span>"
    )
    if "n" not in i:
        return f"<li><div class='cat-row'><span>“{esc(i['complaint'])}”</span>{scores}</div></li>"
    label = i["complaint"] if i["outcome"] == "worse" else i.get("goal") or i["complaint"]
    t = i["tally"]
    segs = "".join(
        f"<span class='seg {k}'></span>" for k in ("better", "same", "worse") for _ in range(t[k])
    )
    parts = [f"{t[k]} {LABEL.get(k, k)}" for k in ("better", "same", "worse") if t[k]]
    return (
        f"<li><div class='cat-row'><span>{esc(label)}</span>{scores}</div>"
        f"<div class='cat-meta'><span class='count'>{i['n']} transcript{'s' * (i['n'] != 1)}</span>"
        f"<span class='bar' style='width:{16 * i['n']}px'>{segs}</span>"
        f"<span class='muted'>{', '.join(parts)}</span></div></li>"
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

    def section(key: str, label: str) -> str:
        sub = ""
        if grouped:
            k = sum(i["n"] for i in by[key])
            sub = (
                f"<div class='h3-sub'>{len(by[key])} of {len(gs)} behaviours · "
                f"{k} transcript{'s' * (k != 1)}</div>"
            )
        body = (
            "<ul>" + "".join(_cat_item(i) for i in by[key]) + "</ul>"
            if by[key]
            else "<div class='none'>none</div>"
        )
        return f"<h3 class='{key}'>{label}</h3>{sub}{body}"

    unchanged = (
        f"<div class='cats-same'>{section('same', 'No improvement')}</div>" if by["same"] else ""
    )
    avg = ""
    if s["newMean"] is not None:
        old_mean = s["controlMean"] if s["controlled"] else s["flaggedMean"]
        short = _short_models(s["oldModels"])
        old_who = " & ".join(short) if len(short) <= 2 else "Older models"
        if not s["controlled"]:
            old_who = "Your flagged replies"
        avg = (
            "<div class='panel'><div class='panel-label'>Average judge score</div><div class='avg'>"
            f"<div><div class='who'>{esc(old_who)}</div>{_score(old_mean, big=True)}</div>"
            "<div class='arrow'>→</div>"
            f"<div><div class='who'>{esc(pretty_model(new))}</div>"
            f"{_score(s['newMean'], big=True)}</div></div>"
            "<div class='scale'>0 = problematic, 10 = fixed</div></div>"
        )
    return f"""
<div id="scorecard">
  <h1 class="sc-title">/bad</h1>
  <p class="sc-sub">Old replies that annoyed me, re-sampled on a newer model and judged blind
    against my complaint.</p>
  <p class="sc-head"><span class="model" style="color:{_model_color(new)}">{_logo(new)}{esc(pretty_model(new))}</span>
    compared to older models ({esc(", ".join(_short_models(s["oldModels"])))})</p>
  <div class="panels">
    {avg}
    <div class="panel"><div class="panel-label">Per-transcript changes
      <span>{s["scored"]} transcripts</span></div>
      <div class="tally">
        <div class="better"><b>{s["better"]}</b>better</div>
        <div class="worse"><b>{s["worse"]}</b>worse</div>
        <div class="same"><b>{s["same"]}</b>no improvement</div>
      </div></div>
  </div>
  <div class="cats">
    <div>{section("better", "Improved")}</div>
    <div>{section("worse", "Regressed")}</div>
  </div>
  {unchanged}
  <div class="sc-foot">Judge: {esc(pretty_model(s["judge"] or judge))}, {_pretty_date(when)}</div>
</div>"""


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
    body = esc(_one_line(prompt, 400))
    if len(prompt) > 400:
        body = f"<details><summary>{body}</summary><div class='md'>{esc(prompt[:8000])}</div></details>"
    then = f"<div class='then'>then {' and '.join(work)}</div>" if work else ""
    return (
        "<div class='shared'><div class='shared-label'>Shared conversation · identical for all "
        f"three replies</div><div class='bubble user'>{body}</div>{then}"
        "<div class='fork'>↓ each reply below continues from exactly here</div></div>"
    )


def _reply(title: str, model: str, score, why: str, rejected: str, response: str, evidence) -> str:
    esc = html.escape
    verdict = "<span class='rej'>rejected</span>" if rejected else _score(score)
    note = rejected or why
    return (
        "<div class='reply'>"
        f"<div class='verdict'><div class='verdict-head'><span><b>{esc(title)}</b>"
        f"<span class='muted'> · {esc(pretty_model(model))}</span></span>{verdict}</div>"
        + (f"<div class='why'><span class='label'>Judge</span>{esc(note)}</div>" if note else "")
        + "</div>"
        f"<div class='bubble bot'><div class='md' data-evidence='{esc(json.dumps(evidence or []))}'>"
        f"{esc(response or '')}</div></div></div>"
    )


def _case(r: dict) -> str:
    esc = html.escape
    o = r["outcome"]
    rejected = r.get("rejected") or ""

    def rej(who: str) -> str:
        return rejected.split(": ", 1)[1] if rejected.startswith(who) else ""

    replies = [
        _reply(
            "Original",
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
    scores = " · ".join(_score(r.get(k)) for k in ("flagged", "control", "new"))
    tag = f"{MARK[o]} {esc(LABEL.get(o, o))}"
    style = ""
    ref, new = r.get("reference"), r.get("new")
    if o in ("better", "same", "worse") and ref is not None and new is not None:
        d = new - ref
        tag += f" {'+' if d > 0 else '−' if d < 0 else '±'}{fmt(abs(d))}"
        if o != "same":
            pct = round(12 + 48 * min(1.0, abs(d) / 9))
            style = f" style='background: color-mix(in srgb, var(--{o}) {pct}%, transparent)'"
    return (
        f"<details class='case'><summary><span class='tag pill {o}'{style}>{tag}"
        f"</span><span class='case-title'>{esc(_one_line(r['complaint'], 120))}</span>"
        f"<span class='case-scores' title='original · old model asked again · new model'>"
        f"{scores}</span></summary>"
        f"<div class='complaint'>User complaint: {esc(r['complaint'])}</div>"
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
            f"<span class='counts'>{counts}</span></summary>"
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
        f"<h2 class='section'>Every transcript ({len(replayed)}), by behaviour</h2>",
        "<div class='muted'>Scores are out of 10 (original · old model asked again · new model): "
        "0 means the complaint applies in full, 10 means it no longer applies.</div>",
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
