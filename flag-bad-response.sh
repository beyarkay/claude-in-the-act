#!/usr/bin/env bash
# flag-bad-response.sh — UserPromptSubmit gate for claude-in-the-act.
#
# Fires on EVERY prompt submit, so the common case must be cheap: a tight,
# dependency-free bash check for a prompt that starts with "BAD: " (no jq), and
# exit 0 for everything else. Only a real flag boots python3 — which snapshots
# the transcript + git state, appends the annotation, and exits 2 to block the
# prompt. capture.py re-checks the prefix and is the authoritative decision.
#
# Runtime is stdlib-only, so there is no install/venv/uv step — just python3
# (>=3.11) on PATH. Bundled as a plugin, ${CLAUDE_PLUGIN_ROOT} points at the
# install dir; run standalone (e.g. wired directly into settings.json), it falls
# back to this script's own directory. Either way PYTHONPATH=<root>/src makes
# the package importable without installing it.
set -euo pipefail

ROOT="${CLAUDE_PLUGIN_ROOT:-$(unset CDPATH; cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

input=$(cat)

# Cheap gate: proceed only if the prompt field starts with "BAD: ". A regex
# (not a substring) so a stray "BAD: " elsewhere in the payload can't trigger
# a capture; whitespace around the JSON colon is tolerated.
re='"prompt"[[:space:]]*:[[:space:]]*"BAD: '
[[ "$input" =~ $re ]] || exit 0

# It IS a flag, so we must capture and block — a missing interpreter is a
# blocking error, not a silent passthrough that would leak "BAD: …" to the model.
if ! command -v python3 >/dev/null 2>&1; then
  echo "claude-in-the-act: python3 not found — install Python 3.11+ to capture BAD responses." >&2
  exit 2
fi

exec env PYTHONPATH="$ROOT/src" python3 -m claude_in_the_act.capture <<<"$input"
