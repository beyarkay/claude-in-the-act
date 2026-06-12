# claude-in-the-act

A capture harness + growing dataset of **bad Claude Code responses**, each
paired with a human critique. The long-term goal: replay these as an eval when a
new model ships — prefill the full context, let the new model produce the next
turn, and compare its judgement against the old (bad) one.

## How it works

`flag-bad-response.sh` is a Claude Code **`UserPromptSubmit`** hook — a thin
bash gate that fires on every submit, checks the prefix with `jq` (no Python
boot on the common path), and only on a match hands the payload to the Python
capturer (`cita-capture`). In any session, when you start a prompt with the
magic prefix:

```
BAD: <why this response was bad; what a proper response would look like>
```

the capturer:

1. Copies the live session transcript (`transcript_path` from the hook payload)
   **verbatim** into `data/<timestamp>-<session-id>.jsonl`. At submit time the
   last `assistant` record in that file is the bad response, so the snapshot is
   the full context up to and including it.
2. Captures **git state** (best-effort) so a file/code-dependent context can be
   reconstructed later without assuming the working tree still exists:
   - `git.sha`, `git.branch`, `git.remote` recorded in the annotation;
   - a full working-tree patch vs `HEAD` — **including untracked files**,
     excluding `.gitignore`d ones — written to `data/<...>.patch`. The patch is
     built in a *throwaway index* (`GIT_INDEX_FILE`) so your real staging area
     is never touched. `git apply` restores it.
3. Appends one final record — your critique + the git state — so the jsonl stays
   the self-contained index (the `.patch`, if any, sits beside it):
   ```json
   {"type":"bad_annotation","critique":"...","sessionId":"...","cwd":"...","capturedAt":"...",
    "git":{"sha":"...","branch":"...","remote":null,"patchFile":"<...>.patch"}}
   ```
4. **Blocks** the prompt (`exit 2`) so the `BAD: ...` line never reaches the
   model — it costs no turn and no tokens. You get a one-line confirmation
   showing where it saved.

Any prompt that doesn't start with `BAD: ` passes straight through.

## What the jsonl contains (and doesn't)

Captured — enough to replay the **conversation**:

- Every user prompt; every assistant turn (`thinking` + `text` + `tool_use`
  with full input args); every `tool_result` (what the model actually saw back).
- Per-message `model` id, timestamps, git branch, cwd.
- Partial injected context as `attachment` records (skill listing, MCP
  instructions, deferred-tool name deltas).

**Not** captured — so you cannot reconstruct the byte-exact API request:

- The system prompt (Claude Code preamble + environment + your CLAUDE.md) — sent
  to the API each turn but never written to the transcript.
- Full tool JSON schemas (only names appear).

We store **verbatim** rather than stripping the ~4% throwaway records
(`mode`, `permission-mode`, `ai-title`, `last-prompt`, `queue-operation`): the
eval format isn't locked, lossy-at-capture is irreversible, and filtering is
free at read-time.

## Replaying a captured response

```
uv run cita-recreate data/<...>.jsonl [model]
```

`cita-recreate` regenerates the final assistant turn in a fresh, independent
headless session and prints `{original, recreated, prompt, model}` so you can
diff the stored (bad) response against a fresh one — e.g. pass a newer model id
to see whether it does better.

How it stays faithful and side-effect-free:

- It drops the final assistant turn (the bad one) and the final user turn,
  restores the rest via `claude --resume --fork-session`, and re-issues the
  final user message as the prompt.
- The harness **rebuilds the system prompt + tool schemas fresh at launch**, so
  the replay tracks the *current* harness/model — the bits the transcript
  doesn't store are restored automatically, and the eval picks up harness
  improvements for free.
- It runs from an **isolated temp dir** (canonicalized for the macOS
  `/var`→`/private/var` symlink, or `--resume` can't find the planted session),
  so it never depends on the original working tree still existing.
- **Tools are off by default** (`--allowed-tools ""`): pure-text regeneration,
  no side effects. This is correct for reasoning/judgement critiques. For a turn
  whose quality came from *doing* tool work, set `ALLOW_TOOLS=1` — but then the
  model's calls actually execute against the *current* filesystem, so only do
  that after restoring the captured `.patch` into a reconstructed tree. (A
  faithful record-replay tool shim is a TODO.)

## Install

Requires [`uv`](https://docs.astral.sh/uv/) on `PATH` (the hook invokes
`uv run`) and `jq`.

The hook script lives here; activation is one entry in `~/.claude/settings.json`
under `hooks.UserPromptSubmit`:

```json
{
  "matcher": "",
  "hooks": [
    { "type": "command", "command": "/Users/brk/projects/claude-in-the-act/flag-bad-response.sh" }
  ]
}
```

(Already wired up if this repo was set up by Claude.)

## Layout

```
flag-bad-response.sh         thin UserPromptSubmit gate (bash + jq)
src/claude_in_the_act/
  capture.py                 cita-capture: snapshot transcript + git state
  recreate.py                cita-recreate: replay the final assistant turn
tests/                       pytest suite (unit + marked e2e)
data/                        captured <ts>-<session>.jsonl (+ .patch) snapshots (git-ignored)
pyproject.toml               uv project; console scripts; pytest config
```

## Tests

```
uv run pytest            # fast offline unit tests
uv run pytest -m e2e     # full capture→delete→replay round-trip via real `claude` (costs $$)
```

## Note on `data/`

Snapshots are full transcripts and can contain file contents, paths, and
secrets, so `data/*.jsonl` and `data/*.patch` are git-ignored by default. Remove
the ignore rules if you want to version/share the collection.
