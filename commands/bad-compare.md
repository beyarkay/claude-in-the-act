---
description: Re-run your saved /bad cases on a newer model and score whether it does better
argument-hint: <model|latest> [--judge MODEL] [case ...]
allowed-tools: Bash
---
Run the user's personal benchmark with the Bash tool. It re-samples every saved bad response on the new model and has a blind judge score both, so it takes a few minutes and spends model tokens; run it in the background if you can, and give it a timeout of at least 30 minutes.

"${CLAUDE_PLUGIN_ROOT}/bin/slash-bad" compare $ARGUMENTS

If no model was given, use `latest`. If that path doesn't exist (for example in Codex), run `slash-bad compare ...` instead, or `uvx --from git+https://github.com/beyarkay/claude-in-the-act slash-bad compare ...`.

When it finishes, show the scoresheet it printed verbatim in a code block, then the paths of the PNG card and the HTML report. Don't add your own commentary on the results unless the user asks.
