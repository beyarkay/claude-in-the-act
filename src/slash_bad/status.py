"""Where every saved case stands: when it was collected, how big it was, and each model's score."""

from __future__ import annotations

import html
import re
from collections import Counter
from datetime import datetime

import slash_bad
from slash_bad import judge, llm
from slash_bad.compare import HARNESSES, harness_for_model
from slash_bad.store import Case, all_cases

PAGE = "status.html"


def _model(rec: dict) -> str:
    return (rec.get("model") or rec.get("requestedModel") or "").removesuffix("[1m]")


def collected_at(context: list[dict]) -> str | None:
    """When the flagged moment happened: the last timestamp in the stored conversation."""
    return next((r["timestamp"] for r in reversed(context) if r.get("timestamp")), None)


def context_tokens(harness: str, context: list[dict]) -> int | None:
    """Tokens in the last request the original session sent before the flagged reply, as
    counted by the API in the session's own usage records."""
    for r in reversed(context):
        if harness == "codex":
            p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            usage = p.get("type") == "token_count" and (p.get("info") or {}).get("last_token_usage")
            if usage:
                return usage.get("input_tokens")
        elif r.get("type") == "assistant" and isinstance(r.get("message"), dict):
            if usage := r["message"].get("usage"):
                return sum(
                    usage.get(k) or 0
                    for k in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
    return None


def replay_harness_version(case: Case) -> str | None:
    """The Claude Code version whose system prompt and tools the saved replay request carries."""
    path = case.path / "request.json"
    if not path.exists():
        return None
    m = re.search(r"cc_version=([\d.]+?)(?:\.[0-9a-f]{3})?;", path.read_text()[:20000])
    return m.group(1) if m else None


def _verdict(results: list[dict], target: str, judge_name: str) -> dict | None:
    return next(
        (
            r
            for r in reversed(results)
            if r.get("kind") == "verdict"
            and r.get("target") == target
            and r.get("judge") == judge_name
            and r.get("rubric") == judge.RUBRIC
        ),
        None,
    )


def _scored(v: dict | None) -> dict:
    if v is None:
        return {"state": "not judged"}
    if v.get("rejected"):
        return {"state": "rejected", "why": v["rejected"]}
    if v.get("score") is None:
        return {"state": "error", "why": v.get("rationale") or "the judge failed"}
    return {"state": "scored", "score": v["score"], "why": v.get("rationale", "")}


def model_status(case: Case, results: list[dict], model: str, judge_name: str) -> dict:
    """The replay ``compare`` would use for ``model`` (the first current one that worked),
    and its verdict; or why there isn't one."""
    version = HARNESSES[case.harness].REPLAY_VERSION
    reps = [r for r in results if r.get("kind") == "replay" and _model(r) == model]
    current = [r for r in reps if r.get("replayVersion") == version]
    if not current:
        return {"state": "outdated"}
    good = [r for r in current if not r.get("error")]
    if not good:
        return {"state": "error", "why": current[-1]["error"], "at": current[-1].get("at")}
    rep = good[0]
    return {**_scored(_verdict(results, rep["replayId"], judge_name)), "at": rep.get("at")}


def default_judge(cases: list[Case]) -> str | None:
    """The judge with the most verdicts under the current rubric."""
    n = Counter(
        r.get("judge")
        for c in cases
        for r in c.results()
        if r.get("kind") == "verdict" and r.get("rubric") == judge.RUBRIC
    )
    return n.most_common(1)[0][0] if n else None


def rows(
    judge_name: str | None = None, cases: list[Case] | None = None
) -> tuple[list[dict], str | None]:
    cases = all_cases() if cases is None else cases
    judge_name = judge_name or default_judge(cases)
    results = {c.id: c.results() for c in cases}
    models = sorted(
        {_model(r) for rs in results.values() for r in rs if r.get("kind") == "replay"} - {""},
        key=lambda m: (not m.startswith("claude"), m),
    )
    out = []
    for c in cases:
        ctx = c.context
        rs = results[c.id]
        replayed = {_model(r) for r in rs if r.get("kind") == "replay"}
        out.append(
            {
                "case": c.id,
                "harness": c.harness,
                "harnessVersion": c.meta.get("harnessVersion"),
                "replayHarnessVersion": replay_harness_version(c)
                if c.harness == "claude-code"
                else None,
                "origin": c.meta.get("origin") or "flagged",
                "collected": collected_at(ctx),
                "tokens": context_tokens(c.harness, ctx),
                "model": c.meta.get("model") or "",
                "complaint": c.complaint,
                "behaviour": c.meta.get("behaviour"),
                "original": _scored(_verdict(rs, "original", judge_name)),
                "models": {
                    m: model_status(c, rs, m, judge_name)
                    for m in models
                    if m in replayed and harness_for_model(m) in (None, c.harness)
                },
            }
        )
    return out, judge_name


_TABLE_CSS = """
.status-tools { display: flex; gap: 10px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
.status-tools input { font: 14px Inter, -apple-system, sans-serif; padding: 7px 10px; border-radius: 8px;
                      border: 1px solid var(--line); background: var(--card); color: var(--fg);
                      flex: 1; min-width: 12em; max-width: 26em; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
table.transcripts { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
table.transcripts th { text-align: left; font-size: 12px; font-weight: 650; color: var(--muted); cursor: pointer;
                  padding: 9px 10px; border-bottom: 1px solid var(--line); white-space: nowrap;
                  position: sticky; top: 0; background: var(--card); user-select: none; }
table.transcripts th[aria-sort=ascending]::after { content: ' ▴'; }
table.transcripts th[aria-sort=descending]::after { content: ' ▾'; }
table.transcripts td { padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.transcripts tr:last-child td { border-bottom: none; }
table.transcripts td.num, table.transcripts th.num { text-align: right; white-space: nowrap; }
table.transcripts td.nowrap { white-space: nowrap; }
.st-complaint { min-width: 18em; max-width: 30em; }
.st-complaint .text { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                      overflow: hidden; cursor: pointer; }
.st-complaint.open .text { -webkit-line-clamp: unset; }
.st-complaint .behaviour-name { color: var(--muted); font-size: 12px; margin-top: 2px; }
.st-state { color: var(--muted); font-size: 12px; white-space: nowrap; }
.st-state.error { color: var(--warn); }
.st-control { border-bottom: 1px dotted var(--muted); }
"""

TABLE_JS = """
<script>
document.querySelectorAll('table.transcripts').forEach(t => {
  const body = t.tBodies[0];
  t.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => {
    const asc = th.getAttribute('aria-sort') !== 'ascending';
    t.querySelectorAll('th').forEach(h => h.removeAttribute('aria-sort'));
    th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
    const key = r => r.cells[i].dataset.sort ?? r.cells[i].textContent.trim().toLowerCase();
    const num = r => { const k = key(r); return k === '' ? null : (isNaN(+k) ? k : +k); };
    [...body.rows].sort((a, b) => {
      const x = num(a), y = num(b);
      if (x === null) return 1; if (y === null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
    }).forEach(r => body.appendChild(r));
  }));
});
document.querySelectorAll('.st-complaint .text').forEach(t =>
  t.addEventListener('click', () => t.parentElement.classList.toggle('open')));
const box = document.getElementById('status-search');
if (box) box.addEventListener('input', () => {
  const q = box.value.toLowerCase();
  document.querySelectorAll('table.transcripts tbody tr').forEach(r => {
    r.hidden = q && !r.textContent.toLowerCase().includes(q);
  });
});
</script>
"""


def _tokens(n: int | None) -> str:
    if n is None:
        return "–"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{round(n / 1000)}k"


def _date(iso: str | None) -> str:
    if not iso:
        return "–"
    when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"{when.day} {when:%b %Y}"


def _cell(st: dict | None, control: bool = False) -> str:
    from slash_bad.report import WARN_ICON, _score

    esc = html.escape
    if st is None:
        return "<td class='num' data-sort=''><span class='st-state'>–</span></td>"
    state = st["state"]
    if state == "scored":
        title = ("The original model asked again. " if control else "") + st.get("why", "")
        cls = " class='st-control'" if control else ""
        return (
            f"<td class='num' data-sort='{st['score']}' title='{esc(title)}'>"
            f"<span{cls}>{_score(st['score'])}</span></td>"
        )
    label = {
        "not judged": "not judged",
        "outdated": "old replay",
        "rejected": "rejected",
        "error": f"{WARN_ICON} failed",
    }[state]
    why = {
        "not judged": "Replayed, but not yet judged with this judge and rubric",
        "outdated": "Only replayed with an older replay method; needs replaying again",
    }.get(state, st.get("why", ""))
    cls = "st-state error" if state == "error" else "st-state"
    return (
        f"<td class='num' data-sort='' title='{esc(why)}'><span class='{cls}'>{label}</span></td>"
    )


def section(status_rows: list[dict], judge_name: str | None) -> str:
    from slash_bad.report import pretty_model

    esc = html.escape
    models = sorted(
        {m for r in status_rows for m in r["models"]},
        key=lambda m: (not m.startswith("claude"), m),
    )
    judged = {
        m: sum(r["models"].get(m, {}).get("state") == "scored" for r in status_rows) for m in models
    }
    head = (
        "<tr><th>Collected</th><th>Complaint</th><th>Harness</th><th>Original model</th>"
        "<th class='num' title='Tokens in the last request the original session sent before "
        "the flagged reply, as counted by the API'>Tokens</th>"
        "<th class='num' title='The judge’s score for the reply you flagged'>Flagged reply</th>"
        + "".join(
            f"<th class='num' title='{judged[m]} of {len(status_rows)} judged'>{esc(pretty_model(m).removeprefix('Claude '))}</th>"
            for m in models
        )
        + "</tr>"
    )
    body = []
    for r in sorted(status_rows, key=lambda r: r["collected"] or "", reverse=True):
        version = r["harnessVersion"] or "?"
        if r["replayHarnessVersion"] and r["replayHarnessVersion"] != r["harnessVersion"]:
            version_title = (
                f"Replayed with Claude Code {r['replayHarnessVersion']}’s system prompt and tools"
            )
            version += f" → {r['replayHarnessVersion']}"
        else:
            version_title = ""
        harness = "Claude Code" if r["harness"] == "claude-code" else "Codex"
        note = " · ".join(
            x for x in ("mined" if r["origin"] == "mined" else "", r.get("behaviour") or "") if x
        )
        note = f"<div class='behaviour-name'>{esc(note)}</div>" if note else ""
        body.append(
            f"<tr><td class='nowrap' data-sort='{esc(r['collected'] or '')}'>{_date(r['collected'])}</td>"
            f"<td class='st-complaint' title='{esc(r['case'])}'><div class='text'>{esc(r['complaint'])}</div>{note}</td>"
            f"<td class='nowrap' title='{esc(version_title)}'>{harness} {esc(version)}</td>"
            f"<td class='nowrap'>{esc(pretty_model(r['model']) or '?')}</td>"
            f"<td class='num' data-sort='{r['tokens'] if r['tokens'] is not None else ''}'>{_tokens(r['tokens'])}</td>"
            + _cell(r["original"])
            + "".join(_cell(r["models"].get(m), control=m == r["model"]) for m in models)
            + "</tr>"
        )
    counts = " · ".join(f"{esc(pretty_model(m))} {judged[m]}" for m in models)
    return (
        f"<style>{_TABLE_CSS}</style>"
        f"<h2 class='section' id='all-transcripts'>All saved transcripts</h2>"
        f"<div class='section-count'>{len(status_rows)} transcripts. Judged per model: {counts}</div>"
        f"<div class='muted'>Scores are out of 10 (0 = problematic, 10 = fixed), from "
        f"{esc(pretty_model(llm.ALIASES.get(judge_name, judge_name or '?')))} with the {esc(judge.RUBRIC)} rubric. A dotted "
        "underline marks the original model asked again. Click a column to sort.</div>"
        "<div class='status-tools'><input id='status-search' type='search' "
        "placeholder='Filter by complaint, model, behaviour…'></div>"
        f"<div class='table-wrap'><table class='transcripts'><thead>{head}</thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def page(judge_name: str | None = None) -> str:
    from slash_bad.report import _CSS

    status_rows, judge_name = rows(judge_name)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>/bad transcripts</title><style>{_CSS} main {{ max-width: 1400px; }}</style></head><body><main>"
        + section(status_rows, judge_name)
        + f"</main>{TABLE_JS}</body></html>"
    )


def write(judge_name: str | None = None):
    slash_bad.HOME.mkdir(parents=True, exist_ok=True)
    path = slash_bad.HOME / PAGE
    path.write_text(page(judge_name))
    return path
