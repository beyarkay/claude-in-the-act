# claude-in-the-act

Catch Claude giving a bad answer, flag it in one keystroke, and turn that single
moment into a reusable, personal eval — so when a new model ships you can ask the
only question that matters: **does it fix the things _I_ specifically dislike?**

### The name

"Caught in the act" — found out mid-wrongdoing, red-handed. This tool swaps
_caught_ for _Claude_: you catch **Claude in the act** of a bad response, right
there in the live session, the instant it happens. The "act" is also the
performance — a model's turn is its act on stage. You grab the one bad take and
keep it, so later you can re-run the same scene with a different actor and see
who plays it better.

## Why you'd want this

Benchmarks measure what the benchmark authors care about. This measures what
**you** care about — the specific ways your Claude Code sessions go wrong:
over-eager edits, hedging, ignoring your CLAUDE.md, wrong altitude, missing the
point of the prompt. Every time one of those annoys you, you spend one keystroke
to bank it. Over time you accumulate a private regression suite built entirely
from your own pet peeves.

Then a new model drops. Instead of vibes, you run your whole collection through
it and get a scored report: _here's how much `claude-fable-5` fixes the things
that used to bug you._ The replay rebuilds the harness (system prompt + tool
schemas) fresh every time, so you're always testing the **current** Claude Code
against your **old** complaints — and you pick up harness improvements for free.

## The workflow

### 1. Flag a bad response — mid-session, one line

When Claude gives you a response you don't like, start your **next** prompt with
the magic prefix and say what was wrong:

```
BAD: deleted my test instead of fixing it — should have asked before destructive edits
```

That's it. A `UserPromptSubmit` hook intercepts the line, snapshots the whole
session up to and including the bad turn, records the git state, and **blocks the
prompt** so the `BAD:` line never reaches the model — it costs you no turn and no
tokens. You get a one-line confirmation telling you where it saved.

Anything that doesn't start with `BAD: ` passes straight through, untouched.

### 2. (Optional) Write the answer you wanted

The capture leaves a `goodResponse` slot in the saved bundle. If you have the
patience, open the `.jsonl` and replace the placeholder on the last line with the
response you _actually_ wanted. The judge will hold regenerations to that gold
standard. Skip it and the judge scores against your written feedback alone —
still useful, just less precise.

> **Note:** capture is the part that ships as a plugin. The regenerate +
> judge tooling below (steps 3–4) currently runs from a **source checkout** of
> this repo (`uv run …`) and isn't wired into the plugin yet — packaging it as
> one-click slash commands is the next step.

### 3. Baseline: how bad was it, really?

```
uv run cita-eval --mode stored
```

Scores every captured response 0–10 against your own complaint, with no replay,
so it's cheap (~$0.12 a transcript). **10 means the complaint doesn't apply.**
Your captures should mostly score near 0 — that's the judge agreeing with you.
Anything scoring high is worth reading: either the capture grabbed the wrong turn,
or the complaint doesn't survive a second look.

### 4. Regenerate, and see if the new model still does it

```
uv run cita-eval --mode regen --model claude-fable-5
```

Replays every flagged turn under `--model` and scores the new response on the
**same** rubric, then reports the movement: `1.6 → 7.4 (+5.8) · 31 better, 4
same, 3 worse`. That difference is the whole point of the tool.

`uv run cita-judge <bundle.jsonl> [model]` does one bundle and dumps the full
JSON, which is what you want while debugging a single case.

#### Why the judge is blind

Both modes ask the judge one question: here is a request, here is **one**
response, here is a piece of commentary describing a failure — does that
commentary describe this response? The judge is never told whether it's looking
at your original capture or a fresh regeneration, and never sees the other one.

That symmetry is what makes the two numbers subtractable. The obvious design —
"here's the old response, here's the new one, how much better is it?" — can't
produce a baseline at all, because you can't run it against the old response by
itself. It also quietly rewards any change. The judge is likewise told the
commentary *may not apply*, so it can score a response 10 without being
told it's the good one.

Judging runs on `claude-sonnet-5` by default, so a model never grades itself.
Change the rubric and bump `judge.RUBRIC`; scores carry that tag and old ones are
never silently subtracted from new.

### 5. Keep the code alive: `cita-pin`

```
uv run cita-pin          # pin every capture's commit
uv run cita-pin --list   # what's safe, at risk, or lost
```

A capture stores a commit SHA and expects to check it out later. If that turn
happened in a `git worktree` — Claude Code puts them under
`<repo>/.claude/worktrees/<name>` — the directory and its branch usually get
deleted long before you re-run the eval, and an unreachable commit is pruned by
`git gc` within a fortnight. The transcript outlives the code it describes.

`cita-pin` writes a `refs/cita/<capture>` ref in whichever repo holds each
object. No worktree kept, no checkout, no working tree — just reachability. It
lives outside `refs/heads`, so it never shows up in `git branch` or gets pushed
by accident. Run it after a capture session; `--unpin` reverses it.

## How the replay stays honest

A regeneration is only meaningful if the model faces the _real_ situation again.
Two paths, picked automatically:

- **Tool-free turns** — reasoning, judgement, verbosity critiques. The context
  is restored in an isolated temp session and the final prompt re-issued with
  tools off. Fully self-contained.
- **Tool-using turns** — the model touched files. The harness **reconstructs the
  actual repo**: a detached `git worktree` off your local repo, checked out to
  the captured SHA, the working-tree patch re-applied, and the bad turn's own
  edits undone — so the model meets the genuine pre-turn state and real tools
  (git status, `.gitignore`, directory layout) all behave correctly. Paths in
  the resumed context are rewritten to the worktree, so your real repo is never
  touched.

Reconstruction needs git to be reachable (your local repo or a remote with the
SHA). It's working-tree isolation, not a hard sandbox.

## Getting started

Capture ships as a Claude Code **plugin**. The only requirement is `python3`
(≥3.11) on your `PATH` — the runtime is stdlib-only, so there's no `pip`, venv,
`uv`, or `jq` to install. From inside Claude Code:

```
/plugin marketplace add beyarkay/claude-in-the-act
/plugin install claude-in-the-act@claude-in-the-act
```

That registers the `UserPromptSubmit` hook automatically — no `settings.json`
editing. To try it before it's published anywhere, point the marketplace at a
local checkout instead:

```
/plugin marketplace add /path/to/claude-in-the-act
/plugin install claude-in-the-act@claude-in-the-act
```

**Then restart Claude Code** — quit your session and relaunch. Hooks are
registered at session start, so capture won't fire in a session that was already
open when you installed (you'd type `BAD: …` and nothing would happen).

Now go use Claude Code as normal. The next time it disappoints you, type
`BAD: <what was wrong>` and you've started your eval set — you'll get a
`✎ BAD response captured` confirmation and the prompt won't reach the model.
Captures land in `~/.claude/claude-in-the-act/data/` (override with
`CITA_DATA_DIR`).

## A note on your captures

Captures are full session transcripts — they can contain file contents, paths,
and secrets. They're written to `~/.claude/claude-in-the-act/data/`, **outside**
any git repo, so they're never accidentally committed. That directory also
survives `/plugin update` (the plugin's own install dir does not). Treat it as
private; back it up or sync it deliberately, not by accident.
