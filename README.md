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

### 3. Regenerate + judge a single capture

```
uv run cita-judge ~/.claude/claude-in-the-act/data/<timestamp>-<session>.jsonl [model]
```

This is the single entry point. It **regenerates** the flagged turn under the
current harness and then **judges** the result 0–10: how much better is the new
response at addressing _your_ feedback (and matching your ideal answer, if you
wrote one)? The judge runs on a different model (`claude-sonnet-4-6`) to avoid a
model grading itself.

Pass a model id as the second argument to test a specific one —
`claude-fable-5`, say, to check whether a newer model fixes what you flagged.

### 4. Score your whole collection

```
uv run cita-eval [model]
```

Runs the judge over **every** captured bundle in parallel and writes a markdown
report to `data/eval-<timestamp>-<model>.md`: an overall average plus a
per-transcript breakdown, worst first, each with the judge's rationale. This is
the payoff — one command, one number, "how much does this model fix what I
dislike."

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
