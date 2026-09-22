---
description: Anonymise your /bad cases and contribute them to the public slash-bad dataset
argument-hint: [case ...]
allowed-tools: Bash
---
First run a dry run with the Bash tool:

"${CLAUDE_PLUGIN_ROOT}/bin/slash-bad" share --dry-run $ARGUMENTS

(If that path doesn't exist, use `slash-bad share ...` or `uvx --from git+https://github.com/beyarkay/claude-in-the-act slash-bad share ...`.)

Show the user what it reports: how many cases, what was redacted, and where the anonymised files were written. Tell them to look through those files, because a transcript can contain code, file contents and names that no automatic redaction will catch. Only if the user then explicitly says to go ahead, run the same command with `--yes` instead of `--dry-run`, and show them the pull request link it prints.
