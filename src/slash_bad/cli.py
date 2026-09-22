"""``slash-bad``: your personal benchmark for the things you actually care about.

slash-bad list                         your saved cases
slash-bad show 3                       one case in full
slash-bad compare latest               re-run every case on the newest model
slash-bad compare claude-opus-5-5 --judge opus
slash-bad capture SESSION.jsonl "it said the tests pass but never ran them"
slash-bad share --dry-run              preview the anonymised upload
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from slash_bad import CASES_DIR, HOME, claude_code, codex, compare, hook, judge, report
from slash_bad.store import Case, all_cases, find_case

CONFIG = HOME / "config.json"


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def cmd_list(args) -> int:
    print(hook.list_text())
    return 0


def cmd_show(args) -> int:
    case = find_case(args.case)
    mod = compare.HARNESSES[case.harness]
    from slash_bad.view import judge_context

    earlier, turn = judge_context(mod.steps(case.context))
    print(f"case {case.id} · {case.harness} · {case.meta.get('model')}")
    print(f"complaint: {case.complaint}\n")
    print(turn[-4000:])
    print(f"\n<flagged-response>\n{case.meta.get('flagged')}\n</flagged-response>\n")
    for r in case.results():
        if r.get("kind") == "verdict":
            print(
                f"verdict [{r.get('target')}] {r.get('score')}/10 by {r.get('judgeModel')}: {r.get('rationale')}"
            )
        elif r.get("kind") == "replay":
            print(
                f"replay [{r.get('replayId')}] {r.get('model')}: {(r.get('text') or r.get('error') or '')[:300]!r}"
            )
    return 0


def cmd_capture(args) -> int:
    path = Path(args.transcript).expanduser()
    harness = args.harness or hook.detect_harness({"transcript_path": str(path)})
    mod = codex if harness == "codex" else claude_code
    meta, context = mod.build_case(path, args.complaint)
    if args.model_label:
        meta["model"] = args.model_label
    case = Case.create({**meta, "source": str(path)}, context)
    print(f"saved {case.id}: {meta['model']} · {len(context)} context records")
    return 0


def cmd_delete(args) -> int:
    import shutil

    case = find_case(args.case)
    shutil.rmtree(case.path)
    print(f"deleted {case.id}")
    return 0


def _selected(refs: list[str]) -> list[Case]:
    return [find_case(r) for r in refs] if refs else all_cases()


def cmd_compare(args) -> int:
    cfg = load_config()
    judge_model = args.judge or cfg.get("judge") or judge.DEFAULT_JUDGE
    cases = _selected(args.cases)
    if not cases:
        print(
            f"no cases in {CASES_DIR} yet — flag one with /bad <what went wrong>", file=sys.stderr
        )
        return 1
    if not args.no_group:
        from slash_bad import behaviours

        try:
            behaviours.assign(cases)
        except RuntimeError as e:
            print(
                f"slash-bad: couldn't group cases by behaviour ({e}); continuing", file=sys.stderr
            )
    rows = compare.run(
        cases,
        args.model,
        judge_model=judge_model,
        variant=args.variant,
        workers=args.workers,
        fresh=args.fresh,
        samples=args.samples,
        control=not args.no_control,
    )
    if not rows:
        print(f"no cases can run on {args.model} (wrong harness?)", file=sys.stderr)
        return 1
    print(report.terminal(rows, judge_model))
    paths = report.write_all(rows, judge_model, show_complaints=not args.hide_complaints)
    print()
    for kind in ("png", "svg", "html"):
        if kind in paths:
            print(f"{kind:>5}: {paths[kind]}")
    if args.open and "html" in paths:
        webbrowser.open(paths["html"].as_uri())
    return 0


def cmd_baseline(args) -> int:
    cfg = load_config()
    judge_model = args.judge or cfg.get("judge") or judge.DEFAULT_JUDGE
    cases = _selected(args.cases)
    for case in cases:
        v = compare.baseline(case, judge_model)
        print(f"{v.get('score')!s:>4}/10  {case.id}  {' '.join(case.complaint.split())[:70]}")
    return 0


def cmd_behaviours(args) -> int:
    from slash_bad import behaviours

    cases = all_cases()
    labels = behaviours.assign(cases, regroup=args.regroup)
    for b in behaviours.load():
        members = [c for c in cases if labels.get(c.id) == b["name"]]
        if not members:
            continue
        print(f"{b['name']} ({len(members)}): {b.get('description', '')}")
        for c in members:
            print(f"    {c.id}  {' '.join(c.complaint.split())[:90]}")
    return 0


def cmd_config(args) -> int:
    cfg = load_config()
    if args.key and args.value is not None:
        cfg[args.key] = args.value
        HOME.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    return 0


def cmd_setup(args) -> int:
    from slash_bad import setup

    root = Path(__file__).resolve().parents[2]
    written = setup.install(root)
    if not args.quiet:
        for path in written:
            print(f"installed {path}")
        print("Type /bad <what went wrong> in Claude Code, or $bad in Codex.")
    return 0


def cmd_uninstall(args) -> int:
    from slash_bad import setup

    for path in setup.uninstall():
        print(f"removed {path}")
    return 0


def cmd_share(args) -> int:
    from slash_bad import share

    return share.main(args)


def cmd_import_legacy(args) -> int:
    from slash_bad import legacy

    return legacy.main(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="slash-bad", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list saved cases").set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="show one case")
    p.add_argument("case")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("capture", help="save a case from a session transcript file")
    p.add_argument("transcript")
    p.add_argument("complaint")
    p.add_argument("--harness", choices=("claude-code", "codex"))
    p.add_argument("--model-label", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("delete", help="delete one case")
    p.add_argument("case")
    p.set_defaults(fn=cmd_delete)

    p = sub.add_parser("compare", help="re-run your cases on a model and score the change")
    p.add_argument("model", help="e.g. latest, claude-opus-5-5, sonnet, gpt-6-sol")
    p.add_argument("cases", nargs="*", help="case ids or list numbers (default: all)")
    p.add_argument(
        "--judge",
        help=f"judge model (default {judge.DEFAULT_JUDGE}; codex:<model> to judge with Codex)",
    )
    p.add_argument(
        "--variant",
        choices=("deny", "stop"),
        default="deny",
        help="deny: refuse the new model's tool calls and judge what it says next; "
        "stop: judge its first message, tool calls included",
    )
    p.add_argument(
        "--samples", type=int, default=1, help="re-runs per model per case, averaged (default 1)"
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        help="skip re-running the old model; compare with the flagged response instead (cheaper, flatters the new model)",
    )
    p.add_argument("--no-group", action="store_true", help="don't group cases by behaviour")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--fresh", action="store_true", help="re-run replays instead of reusing cached ones"
    )
    p.add_argument(
        "--hide-complaints", action="store_true", help="leave complaint text off the share card"
    )
    p.add_argument("--open", action="store_true", help="open the HTML report when done")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("baseline", help="judge the stored bad responses only (cheap)")
    p.add_argument("cases", nargs="*")
    p.add_argument("--judge")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("behaviours", help="group your cases by the behaviour they complain about")
    p.add_argument("--regroup", action="store_true", help="start the grouping over")
    p.set_defaults(fn=cmd_behaviours)

    p = sub.add_parser("config", help="show or set a config value, e.g. `config judge sonnet`")
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser(
        "share", help="anonymise cases and open a pull request on the shared Hugging Face dataset"
    )
    p.add_argument("cases", nargs="*")
    p.add_argument(
        "--dry-run", action="store_true", help="write the anonymised files locally and stop"
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--repo", default="beyarkay/slash-bad")
    p.add_argument("--redact", default="", help="extra comma-separated strings to redact")
    p.add_argument(
        "--keep-instructions",
        action="store_true",
        help="include your CLAUDE.md / AGENTS.md text (dropped by default)",
    )
    p.add_argument(
        "--direct",
        action="store_true",
        help="commit directly instead of opening a pull request (repo owners)",
    )
    p.set_defaults(fn=cmd_share)

    p = sub.add_parser("import-legacy", help="import captures from claude-in-the-act")
    p.add_argument("dir", nargs="?", default=str(Path.home() / ".claude/claude-in-the-act/data"))
    p.set_defaults(fn=cmd_import_legacy)

    p = sub.add_parser("setup", help="install the short /bad, /bad-compare, … skills")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_setup)

    sub.add_parser("uninstall", help="remove the skills `setup` installed").set_defaults(
        fn=cmd_uninstall
    )

    sub.add_parser("hook", help=argparse.SUPPRESS).set_defaults(fn=lambda a: hook.main())

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except KeyError as e:
        print(f"slash-bad: {e.args[0]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
