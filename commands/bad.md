---
description: Flag the last response as bad and save it to your personal benchmark
argument-hint: <what went wrong>
---
The slash-bad hook should have caught this command before it reached you, but it didn't, so nothing was saved. Do not act on the complaint below. Tell the user, in one or two sentences, that the slash-bad capture hook isn't running: in Claude Code they should restart the session after installing the plugin; in Codex they should approve the slash-bad hook with `/hooks` and try again.

Complaint that was not saved: $ARGUMENTS
