from slash_bad.redact import Redactor


def red(extra=None, user="alice"):
    r = Redactor(extra)
    r.user = user
    r.literals = [s for s in (extra or [])]
    import re

    r._user_word = re.compile(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])")
    return r


def test_secrets():
    r = red()
    s = r.text(
        "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123 and hf_abcdefghijklmnopqrstuvwxyzABCDEF "
        "and ghp_abcdefghijklmnopqrstuvwxyz0123456789 and AKIAABCDEFGHIJKLMNOP"
    )
    assert "sk-ant" not in s and "hf_abc" not in s and "ghp_" not in s and "AKIA" not in s
    assert r.counts["anthropic-key"] == 1


def test_assignments_and_url_credentials():
    r = red()
    s = r.text('OPENAI_API_KEY="abc123def456ghi" postgres://bob:hunter22@db.example.com/x')
    assert "abc123def456ghi" not in s and "hunter22" not in s
    assert s.startswith("OPENAI_API_KEY=<redacted>")


def test_home_paths_username_and_emails():
    r = red(user="alice")
    s = r.text(
        "/Users/alice/proj -Users-alice-proj alice@example.com ping alice noreply@anthropic.com"
    )
    assert "alice" not in s
    assert "/Users/user/proj" in s and "-Users-user-proj" in s
    assert "noreply@anthropic.com" in s


def test_git_hashes_survive():
    r = red()
    sha = "35ff778a1b2c3d4e5f60718293a4b5c6d7e8f901"
    assert r.text(f"commit {sha}") == f"commit {sha}"


def test_images_are_removed():
    r = red()
    out = r.value({"content": [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]})
    assert out == {"content": [{"type": "text", "text": "[image removed]"}]}


def test_tokens_inside_words_are_not_secrets():
    r = red()
    s = "/tmp/claude-task-5475f656-a473-48f6-9ef7-a1435f8a4ecb.txt and risk-assessment-matrix-2026-final"
    assert r.text(s) == s


def test_instruction_files_are_dropped(tmp_path):
    import json

    from slash_bad.redact import redact_case_dir

    recs = [
        {
            "type": "attachment",
            "attachment": {"type": "instructions", "files": [{"content": "secret rules"}]},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "# AGENTS.md instructions for /x\nprivate"}
                ],
            },
        },
        {"type": "user", "message": {"role": "user", "content": "keep me"}},
    ]
    (tmp_path / "case.json").write_text(json.dumps({"complaint": "c"}))
    (tmp_path / "context.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = redact_case_dir(tmp_path, red())
    assert [r["type"] for r in out["context"]] == ["user"]
    assert len(redact_case_dir(tmp_path, red(), keep_instructions=True)["context"]) == 3
