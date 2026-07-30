"""Build a single self-contained HTML viewer for captured ``BAD:`` bundles.

Each bundle in ``DATA_DIR`` becomes a "case": the sidebar lists them with a
slice of your feedback (the complaint), clicking one renders the full chat
scrolled to the bottom — where your ``BAD:`` message sits, just below the
response that prompted it. We never guess which message you flagged; the only
signal is that final ``BAD:`` turn. No server: all data is embedded in the
page and rendered lazily per case.

Tool noise (file dumps, command output, tool inputs) is truncated so the file
stays openable; the prompts and assistant prose — the part you actually read —
are kept whole. ``build_view`` holds the extraction (unit-tested); ``main``
wires up ``CITA_DATA_DIR`` / ``--out`` and opens a browser.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

from claude_in_the_act import DATA_DIR
from claude_in_the_act.transcript import (
    ANNOTATION_TYPE,
    annotation,
    load_bundle,
    user_text,
)

# Caps (chars) so a 19MB transcript doesn't bloat the page. Prose is kept;
# only machine noise (tool I/O, thinking) is clipped.
_TOOL_INPUT_CAP = 2000
_TOOL_RESULT_CAP = 1500
_THINKING_CAP = 6000
_TEXT_CAP = 24000

_CLIP = "\n… [clipped]"


def _clip(s: str, cap: int) -> str:
    return s if len(s) <= cap else s[:cap] + _CLIP


def _stringify(content: object) -> str:
    """Flatten a tool_result's content (str, or list of text blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or json.dumps(b))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _assistant_blocks(rec: dict) -> list[dict]:
    """Renderable blocks of an assistant turn: text / thinking / tool_use."""
    blocks: list[dict] = []
    for b in rec.get("message", {}).get("content", []):
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text" and b.get("text", "").strip():
            blocks.append({"k": "text", "text": _clip(b["text"], _TEXT_CAP)})
        elif kind == "thinking" and b.get("thinking", "").strip():
            blocks.append(
                {"k": "thinking", "text": _clip(b["thinking"], _THINKING_CAP)}
            )
        elif kind == "tool_use":
            try:
                rendered = json.dumps(b.get("input", {}), indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                rendered = str(b.get("input"))
            blocks.append(
                {
                    "k": "tool_use",
                    "name": b.get("name", "tool"),
                    "input": _clip(rendered, _TOOL_INPUT_CAP),
                }
            )
    return blocks


def _messages(recs: list[dict]) -> list[dict]:
    """Records → ordered chat messages, dropping harness bookkeeping.

    Kept: real user prompts, assistant turns (text/thinking/tool_use), tool
    results, and the ``[Request interrupted by user]`` marker. Everything else
    (queue ops, mode changes, snapshots, system records) is skipped.
    """
    msgs: list[dict] = []
    for rec in recs:
        if rec.get("type") == ANNOTATION_TYPE:
            continue
        t = rec.get("type")
        if t == "assistant":
            blocks = _assistant_blocks(rec)
            if blocks:
                msgs.append({"role": "assistant", "blocks": blocks})
        elif t == "user":
            prompt = user_text(rec)
            if prompt is not None:
                msgs.append({"role": "user", "text": _clip(prompt, _TEXT_CAP)})
                continue
            content = rec.get("message", {}).get("content")
            if isinstance(content, list):
                results = [
                    _clip(_stringify(b.get("content")), _TOOL_RESULT_CAP)
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if results:
                    msgs.append({"role": "tool", "results": results})
                    continue
                texts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if any("interrupted" in x.lower() for x in texts):
                    msgs.append({"role": "interrupt", "text": "\n".join(texts)})
    return msgs


def _pretty_ts(captured: str) -> str:
    """``20260618T122734`` → ``2026-06-18 12:27``; pass through anything else."""
    if len(captured) >= 13 and captured[8] == "T":
        d, t = captured[:8], captured[9:]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}"
    return captured


def build_view(data_dir: Path) -> list[dict]:
    """Every bundle in ``data_dir`` as a case dict, newest first."""
    cases: list[dict] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        try:
            recs = load_bundle(path)
        except OSError:
            continue
        if not recs:
            continue
        ann = annotation(recs)
        msgs = _messages(recs)
        cwd = ann.get("cwd") or ""
        cases.append(
            {
                "id": path.stem,
                "file": path.name,
                "feedback": ann.get("feedback", "").strip(),
                "capturedAt": _pretty_ts(ann.get("capturedAt", "")),
                "repo": Path(cwd).name if cwd else "",
                "cwd": cwd,
                "git": ann.get("git") or {},
                "model": _last_model(recs),
                "messages": msgs,
            }
        )
    cases.sort(key=lambda c: c["capturedAt"], reverse=True)
    return cases


def _last_model(recs: list[dict]) -> str:
    for r in reversed(recs):
        if r.get("type") == "assistant":
            m = r.get("message", {}).get("model")
            if m:
                return m
    return ""


def render_html(cases: list[dict]) -> str:
    payload = json.dumps(cases, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("/*__DATA__*/", payload)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = Path(args[0]) if args else DATA_DIR / "viewer.html"
    cases = build_view(DATA_DIR)
    if not cases:
        print(f"no bundles in {DATA_DIR}", file=sys.stderr)
        return 1
    out.write_text(render_html(cases), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(
        f"viewer: {len(cases)} cases → {out.resolve()} ({size_mb:.1f} MB)",
        file=sys.stderr,
    )
    if "--no-open" not in sys.argv:
        webbrowser.open(out.resolve().as_uri())
    return 0


# The HTML is assembled with .replace (not an f-string) so CSS/JS braces stay
# literal; only the /*__DATA__*/ token is substituted.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caught in the act — case file</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Hanken+Grotesk:wght@400;500;600;800&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#f4f0e7; --sidebar:#ece5d6; --card:#fdfbf6; --card-2:#f1ebdd;
  --line:#e2dac8; --hair:#e9e1cf;
  --ink:#2c2a25; --muted:#776f60; --faint:#a89e89;
  --accent:#4c5673; --accent-soft:rgba(76,86,115,.09);
  --flag:#b0442d; --flag-soft:rgba(176,68,45,.06);
  --code:#5d4b27;
  --mono:"Space Mono",ui-monospace,Menlo,Consolas,monospace;
  --body:"Hanken Grotesk",ui-sans-serif,system-ui,sans-serif;
  --sidebar-w:340px;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--body); font-size:15px; line-height:1.58;
  display:grid; grid-template-columns:var(--sidebar-w) 1fr; overflow:hidden;
  -webkit-font-smoothing:antialiased;
}
::selection{background:var(--accent);color:#fff}

/* ---- sidebar : the case index -------------------------------------- */
#index{
  background:var(--sidebar); border-right:1px solid var(--line);
  display:flex; flex-direction:column; min-height:0;
}
.brand{padding:18px 20px 14px; border-bottom:1px solid var(--line)}
.brand h1{
  margin:0; font-family:var(--mono); font-weight:700; font-size:15px;
  letter-spacing:.01em; color:var(--ink);
}
.brand h1 .red{color:var(--flag)}
.brand .sub{
  font-family:var(--mono); font-size:10.5px; color:var(--faint);
  margin-top:5px; letter-spacing:.05em; text-transform:uppercase;
}
.search{padding:12px 14px; border-bottom:1px solid var(--line)}
.search input{
  width:100%; background:var(--card); border:1px solid var(--line);
  color:var(--ink); font-family:var(--mono); font-size:12px;
  padding:9px 11px; border-radius:7px; outline:none;
}
.search input:focus{border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft)}
.search input::placeholder{color:var(--faint)}
#cases{overflow-y:auto; flex:1; min-height:0; padding:8px}
.case{
  display:block; width:100%; text-align:left; cursor:pointer;
  background:transparent; border:1px solid transparent; border-radius:9px;
  padding:12px 14px 13px; margin-bottom:3px; color:inherit; position:relative;
  border-left:2px solid transparent;
}
.case:hover{background:rgba(0,0,0,.035)}
.case:focus-visible{outline:2px solid var(--accent); outline-offset:1px}
.case.active{background:var(--card); border-color:var(--line); border-left-color:var(--accent); box-shadow:0 1px 2px rgba(0,0,0,.04)}
.case .meta{
  display:flex; justify-content:space-between; align-items:baseline; gap:8px;
  font-family:var(--mono); font-size:10px; color:var(--faint);
  letter-spacing:.03em; margin-bottom:6px;
}
.case .repo{color:var(--accent); font-weight:700; text-overflow:ellipsis; overflow:hidden; white-space:nowrap}
.case .complaint{
  font-family:var(--mono); font-size:12px; line-height:1.5; color:var(--ink);
  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;
}
.case .complaint.empty{color:var(--faint); font-style:italic}

/* ---- reading pane -------------------------------------------------- */
#stage{display:flex; flex-direction:column; min-height:0; min-width:0}
#dossier{
  padding:18px 32px 16px; border-bottom:1px solid var(--line); background:var(--sidebar);
}
#dossier .row{
  display:flex; gap:20px; flex-wrap:wrap; font-family:var(--mono);
  font-size:11px; color:var(--muted); letter-spacing:.02em; margin-bottom:11px;
}
#dossier .row .k{color:var(--faint); text-transform:uppercase; letter-spacing:.08em; margin-right:2px}
#chat{overflow-y:auto; flex:1; min-height:0; padding:30px 32px 120px}

/* turns */
.turn{max-width:780px; margin:0 auto 20px; position:relative}
.turn .who{
  font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--faint); margin-bottom:7px;
}
.turn.user .who{color:var(--accent)}
.bubble{
  border:1px solid var(--line); border-radius:11px; padding:14px 18px;
  background:var(--card); white-space:pre-wrap; word-wrap:break-word;
}
.turn.user .bubble{background:var(--accent-soft); border-color:rgba(76,86,115,.22); border-left:2px solid var(--accent)}
.bubble + .bubble{margin-top:8px}
.turn p{margin:0 0 .7em} .turn p:last-child{margin:0}
.turn code{font-family:var(--mono); font-size:.85em; background:var(--card-2); padding:.1em .35em; border-radius:4px; color:var(--code)}
.turn pre{background:var(--card-2); border:1px solid var(--line); border-radius:9px; padding:12px 14px; overflow-x:auto; margin:.6em 0}
.turn pre code{background:none; padding:0; color:#3a382f; font-size:12.5px; line-height:1.55}
.turn h1,.turn h2,.turn h3{font-family:var(--body); line-height:1.25; margin:.9em 0 .4em; font-weight:800}
.turn h1{font-size:1.28em} .turn h2{font-size:1.14em} .turn h3{font-size:1.02em}
.turn ul,.turn ol{margin:.4em 0 .7em; padding-left:1.4em}
.turn a{color:var(--accent)}
.turn blockquote{border-left:2px solid var(--line); margin:.5em 0; padding-left:12px; color:var(--muted)}

/* thinking + tools : the machine layer, quiet + foldable */
details.fold{margin:8px 0; border:1px solid var(--line); border-radius:9px; background:var(--card-2); overflow:hidden}
details.fold>summary{
  cursor:pointer; list-style:none; padding:8px 13px; font-family:var(--mono);
  font-size:10.5px; letter-spacing:.05em; color:var(--muted); display:flex;
  align-items:center; gap:8px; user-select:none;
}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary::before{content:"\25B8"; color:var(--faint)}
details.fold[open]>summary::before{content:"\25BE"}
details.fold>summary:hover{color:var(--ink)}
details.fold .body{padding:11px 13px 12px; font-family:var(--mono); font-size:12px; line-height:1.55; white-space:pre-wrap; color:var(--muted); border-top:1px solid var(--line)}
details.thinking>summary{color:#6f6488}
.toolname{color:var(--accent); font-weight:700}
.turn.tool{max-width:780px}
.interrupt{max-width:780px; margin:0 auto 20px; font-family:var(--mono); font-size:10.5px; color:var(--faint); text-align:center; letter-spacing:.06em}

/* your BAD: turn — the one thing we actually know you flagged. Sits at the
   bottom, just under the response that prompted it, lightly tinted to stand out. */
.turn.bad .who{color:var(--flag)}
.turn.bad .bubble{
  background:var(--flag-soft); border-color:rgba(176,68,45,.28);
  border-left:2px solid var(--flag);
}
.turn.bad .bubble .px{color:var(--flag); font-weight:700; font-family:var(--mono)}

#empty{margin:auto; text-align:center; color:var(--faint); font-family:var(--mono); font-size:13px; padding:40px}
#empty .big{font-size:40px; color:var(--line); margin-bottom:12px}

/* scrollbars */
#cases::-webkit-scrollbar,#chat::-webkit-scrollbar{width:11px}
#cases::-webkit-scrollbar-thumb,#chat::-webkit-scrollbar-thumb{background:#d8cfbb; border-radius:6px; border:3px solid var(--sidebar)}
#chat::-webkit-scrollbar-thumb{border-color:var(--paper)}

@media (max-width:820px){
  body{grid-template-columns:1fr; grid-template-rows:auto 1fr}
  #index{max-height:42vh}
}
</style>
</head>
<body>
<nav id="index">
  <div class="brand">
    <h1>caught <span class="red">in the act</span></h1>
    <div class="sub">case file · <span id="count"></span> exhibits</div>
  </div>
  <div class="search"><input id="q" type="search" placeholder="filter complaints / repos…" autocomplete="off"></div>
  <div id="cases" role="listbox" aria-label="Captured cases"></div>
</nav>

<main id="stage">
  <header id="dossier" hidden>
    <div class="row">
      <span><b><span class="k">case</span> <span id="d-id"></span></b></span>
      <span><b><span class="k">when</span> <span id="d-when"></span></b></span>
      <span><b><span class="k">repo</span> <span id="d-repo"></span></b></span>
      <span><b><span class="k">model</span> <span id="d-model"></span></b></span>
    </div>
  </header>
  <div id="chat"><div id="empty"><div class="big">⚐</div>pick a case from the index<br>to read the transcript</div></div>
</main>

<script id="data" type="application/json">/*__DATA__*/</script>
<script>
const CASES = JSON.parse(document.getElementById('data').textContent);
const elCases = document.getElementById('cases');
const elChat  = document.getElementById('chat');
const elDoss  = document.getElementById('dossier');
document.getElementById('count').textContent = CASES.length;

const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// minimal, safe markdown: fenced code, inline code, headings, bold/italic,
// links, lists, paragraphs. Escapes first, so input can't inject markup.
function md(src){
  const lines=esc(src).split('\n');
  let out=[], list=null, code=null;
  const closeList=()=>{ if(list){out.push('</'+list+'>'); list=null;} };
  const inline=t=>t
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  for(const raw of lines){
    if(/^\s*```/.test(raw)){
      if(code===null){ closeList(); code=[]; }
      else { out.push('<pre><code>'+code.join('\n')+'</code></pre>'); code=null; }
      continue;
    }
    if(code!==null){ code.push(raw); continue; }
    let m;
    if((m=raw.match(/^(#{1,3})\s+(.*)$/))){ closeList(); out.push('<h'+m[1].length+'>'+inline(m[2])+'</h'+m[1].length+'>'); }
    else if((m=raw.match(/^\s*[-*]\s+(.*)$/))){ if(list!=='ul'){closeList();out.push('<ul>');list='ul';} out.push('<li>'+inline(m[1])+'</li>'); }
    else if((m=raw.match(/^\s*\d+\.\s+(.*)$/))){ if(list!=='ol'){closeList();out.push('<ol>');list='ol';} out.push('<li>'+inline(m[1])+'</li>'); }
    else if(raw.trim()===''){ closeList(); }
    else { closeList(); out.push('<p>'+inline(raw)+'</p>'); }
  }
  if(code!==null) out.push('<pre><code>'+code.join('\n')+'</code></pre>');
  closeList();
  return out.join('\n');
}

function fold(cls,label,body,open){
  return '<details class="fold '+cls+'"'+(open?' open':'')+'><summary>'+label+
    '</summary><div class="body">'+body+'</div></details>';
}

function renderTurn(m){
  if(m.role==='interrupt') return '<div class="interrupt">request interrupted by user</div>';
  if(m.role==='tool'){
    const body=m.results.map(r=>esc(r)).join('\n\n<span style="color:var(--faint)">— — —</span>\n\n');
    return '<div class="turn tool">'+fold('','tool result',body,false)+'</div>';
  }
  if(m.role==='user'){
    return '<div class="turn user"><div class="who">you</div><div class="bubble">'+md(m.text)+'</div></div>';
  }
  // assistant
  let inner='';
  for(const b of m.blocks){
    if(b.k==='text')      inner+='<div class="bubble">'+md(b.text)+'</div>';
    else if(b.k==='thinking') inner+=fold('thinking','thinking',esc(b.text),false);
    else if(b.k==='tool_use') inner+=fold('','<span class="toolname">'+esc(b.name)+'</span>',esc(b.input),false);
  }
  return '<div class="turn assistant"><div class="who">claude</div>'+inner+'</div>';
}

let activeBtn=null;
function open(idx){
  const c=CASES[idx];
  if(activeBtn) activeBtn.classList.remove('active');
  activeBtn=elCases.querySelector('[data-i="'+idx+'"]');
  if(activeBtn){ activeBtn.classList.add('active'); activeBtn.scrollIntoView({block:'nearest'}); }

  elDoss.hidden=false;
  document.getElementById('d-id').textContent   = c.id.slice(0,28);
  document.getElementById('d-when').textContent = c.capturedAt||'—';
  document.getElementById('d-repo').textContent = c.repo||'—';
  document.getElementById('d-model').textContent= c.model||'—';

  let html=c.messages.map(renderTurn).join('');
  if(!c.messages.length) html='<div id="empty">this bundle has no readable chat records</div>';
  // Your actual final turn: the BAD: line the hook intercepted. It never
  // reached the transcript, but the feedback is exactly what you typed after
  // "BAD: ", so we rebuild it verbatim and place it last.
  if(c.feedback){
    html+='<div class="turn user bad"><div class="who">you</div><div class="bubble">'+
      '<span class="px">BAD: </span>'+esc(c.feedback)+'</div></div>';
  }
  elChat.innerHTML=html;

  // Land on the bottom = on your BAD: turn, just under the response that
  // prompted it. Reading scrollHeight forces layout so the jump is instant +
  // correct even for huge transcripts; the rAF pass re-pins it after webfonts.
  elChat.scrollTop=elChat.scrollHeight;
  requestAnimationFrame(()=>{ elChat.scrollTop=elChat.scrollHeight; });
  location.hash=idx;
}

function sidebar(filter){
  const f=(filter||'').toLowerCase();
  elCases.innerHTML='';
  CASES.forEach((c,i)=>{
    if(f && !((c.feedback+' '+c.repo+' '+c.cwd).toLowerCase().includes(f))) return;
    const b=document.createElement('button');
    b.className='case';
    b.dataset.i=i; b.setAttribute('role','option');
    const fb=c.feedback?esc(c.feedback):'(no feedback recorded)';
    b.innerHTML=
      '<div class="meta"><span class="repo">'+esc(c.repo||'—')+'</span><span>'+esc(c.capturedAt)+'</span></div>'+
      '<div class="complaint'+(c.feedback?'':' empty')+'">'+fb+'</div>';
    b.addEventListener('click',()=>open(i));
    elCases.appendChild(b);
  });
}

function move(d){
  const vis=[...elCases.querySelectorAll('.case')];
  if(!vis.length) return;
  let pos=activeBtn?vis.indexOf(activeBtn):-1;
  pos=Math.max(0,Math.min(vis.length-1,pos+d));
  vis[pos].click(); vis[pos].focus();
}
document.addEventListener('keydown',e=>{
  if(e.target.id==='q') return;
  if(e.key==='j'||e.key==='ArrowDown'){e.preventDefault();move(1);}
  if(e.key==='k'||e.key==='ArrowUp'){e.preventDefault();move(-1);}
  if(e.key==='/'){e.preventDefault();document.getElementById('q').focus();}
});
document.getElementById('q').addEventListener('input',e=>sidebar(e.target.value));

sidebar('');
const start = Number(location.hash.slice(1));
if(CASES.length) open(Number.isInteger(start)&&start>=0&&start<CASES.length?start:0);
</script>
</body>
</html>
"""
