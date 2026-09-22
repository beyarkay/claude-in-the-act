---
license: cc-by-4.0
pretty_name: slash-bad
language:
  - en
tags:
  - agent-traces
  - claude-code
  - codex
  - llm-evaluation
  - human-feedback
task_categories:
  - text-generation
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*.jsonl
---

# slash-bad

Moments when a coding agent did something its user didn't like, flagged in the
middle of real work with `/bad <what went wrong>`, then re-run on newer models.

Each row is one flagged moment: the conversation up to that point, the response
that annoyed the user, the user's own words about what was wrong with it, and
blind verdicts on how far that complaint applies to the original response and to
re-runs by other models.

The data comes from [slash-bad](https://github.com/beyarkay/claude-in-the-act),
a personal benchmark for Claude Code and Codex. People contribute their cases
with `/bad-share`, which anonymises them and opens a pull request here.

## Fields (`data/*.jsonl`)

| field | what it is |
|---|---|
| `id` | random id for the case |
| `harness` | `claude-code` or `codex` |
| `harness_version` | version of the harness when the case was captured |
| `model` | the model that wrote the flagged response |
| `complaint` | what the user said was wrong, verbatim |
| `flagged_response` | the response they flagged |
| `conversation` | the context before it, as steps: `user`, `assistant`, `tool_call`, `tool_result` |
| `verdicts` | blind judge scores, 0 to 10, where 10 means the complaint doesn't apply; `response` is `flagged` or `replay`, with the replaying model and its text |
| `stats` | counts: user messages, tool calls, and whether the flagged message came straight after a tool result |

`sessions/claude_code/<id>.jsonl` and `sessions/codex/<id>.jsonl` hold the same
context in the harness's own format, so a case can be replayed exactly.

## How a replay works

Only the flagged message is regenerated. The stored conversation is resumed
under the current harness, with its real system prompt and tools, and any tool
call the new model tries is refused with a plain "permission denied" message.
That keeps the comparison fair without copying the repository or mocking tool
output, at the cost of not testing whether a newer model would have taken a
different path earlier in the conversation.

## Privacy

Contributions are anonymised automatically before upload: known secret formats,
credentials in URLs, key/value secrets, email addresses, private IP addresses,
home directory paths, the local username and git identity are replaced, and the
user's CLAUDE.md is dropped. Code and file contents are not removed, and every
contributor is asked to read their files before they upload. If you find
something here that shouldn't be, open a discussion and it will be removed.

## Seed data

The first cases were produced deliberately: sessions of `claude-opus-5` on small
throwaway repositories, in scenarios built to provoke the failure modes described
in [Common LLM failure modes](https://boydkane.com/essays/common-llm-failure-modes),
then flagged and compared against `claude-opus-5-5`.
