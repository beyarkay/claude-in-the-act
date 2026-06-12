#!/usr/bin/env bash
# flag-bad-response.sh — thin UserPromptSubmit gate for claude-in-the-act.
#
# Fires on EVERY prompt submit, so the common case must be cheap: read the
# payload, check the "BAD: " prefix with jq (~ms, no Python boot), and exit 0
# for anything else. Only on a real flag does it hand the payload to the Python
# capturer, which snapshots the transcript + git state and blocks the prompt.
#
# Wired in via ~/.claude/settings.json under hooks.UserPromptSubmit.
# Capture logic lives in src/claude_in_the_act/capture.py.
set -euo pipefail

DIR="$HOME/projects/claude-in-the-act"
input=$(cat)

# `|| true` so malformed-JSON stdin (jq exits non-zero under `set -e`) degrades
# to a clean passthrough instead of erroring every prompt submit.
prompt=$(printf '%s' "$input" | jq -r '.prompt // ""' 2>/dev/null || true)
[[ "$prompt" == "BAD: "* ]] || exit 0

exec uv run --project "$DIR" python -m claude_in_the_act.capture <<<"$input"
