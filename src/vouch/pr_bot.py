"""Deterministic decision logic for the AI auto-merge bot.

Pure stdlib — no model dependency, no vouch-runtime imports. The CI workflows
call ``python -m vouch.pr_bot <subcommand>`` for every decision that must be
trustworthy: an author's trust tier, whether a PR touches core/ui paths, whether
a UI PR carries before/after screenshots, and whether a labeled PR may arm
native auto-merge. CodeRabbit runs as a GitHub App and still comments on PRs,
but its verdict no longer gates anything — nothing here reads it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# the review-gate core: writes here are the north star. mirrored verbatim in
# .github/CODEOWNERS (test_pr_bot asserts parity). a PR touching any of these
# needs the owner's review and is never merged by automation.
CORE_GLOBS: tuple[str, ...] = (
    "src/vouch/proposals.py",
    "src/vouch/lifecycle.py",
    "src/vouch/storage.py",
    "src/vouch/audit.py",
    "src/vouch/models.py",
    "src/vouch/capabilities.py",
    "src/vouch/server.py",
    "src/vouch/jsonl_server.py",
    "src/vouch/http_server.py",
    "src/vouch/cli.py",
    "src/vouch/pr_bot.py",
    "src/vouch/migrations/**",
    "migrations/**",
    ".github/**",
)

# ui surfaces: reviewed by before/after screenshot, never by running the app.
UI_GLOBS: tuple[str, ...] = (
    "web/**",
    "src/vouch/web/**",
    "webapp/**",
)

_OWNER_ASSOCIATION = "OWNER"
_BOT_ACTORS = frozenset({"dependabot[bot]"})


def _match(path: str, glob: str) -> bool:
    g = glob.lstrip("/")
    if g.endswith("/**"):
        prefix = g[:-3]
        return path == prefix or path.startswith(prefix + "/")
    return path == g


def _touches(changed: Iterable[str], globs: Iterable[str]) -> bool:
    globs = tuple(globs)
    return any(_match(p, g) for p in changed for g in globs)


def classify(changed: Sequence[str]) -> dict[str, bool]:
    """Classify a changed-file list. Precedence: core > ui > code."""
    is_core = _touches(changed, CORE_GLOBS)
    is_ui = (not is_core) and _touches(changed, UI_GLOBS)
    return {"is_core": is_core, "is_ui": is_ui, "is_code": not is_core and not is_ui}


def klass(changed: Sequence[str]) -> str:
    c = classify(changed)
    return "core" if c["is_core"] else "ui" if c["is_ui"] else "code"


def is_trusted(author_association: str, actor: str) -> bool:
    return author_association == _OWNER_ASSOCIATION or actor in _BOT_ACTORS


_GH_IMAGE = re.compile(
    r"""(?:!\[[^\]]*\]\(\s*|<img\b[^>]*\bsrc\s*=\s*["']?)"""
    r"""(?:https?://(?:user-images\.githubusercontent\.com/"""
    r"""|github\.com/user-attachments/assets/"""
    r"""|github\.com/[^/\s"'>]+/[^/\s"'>]+/assets/))""",
    re.I,
)


def has_before_after_screenshots(body: str | None) -> bool:
    """True when the PR body embeds >=2 GitHub-hosted images (before + after)."""
    if not body:
        return False
    return len(_GH_IMAGE.findall(body)) >= 2


def should_arm_automerge(*, is_core: bool, ci_passing: bool,
                         claude_verdict: str, is_draft: bool) -> bool:
    """Deterministic arm gate. Claude can only veto — it never widens this."""
    if is_draft or is_core or not ci_passing:
        return False
    return claude_verdict == "APPROVE"


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def extract_changed_paths(files_json: str) -> list[str]:
    """flatten a REST ``/pulls/{n}/files`` payload into a path list.

    emits both ``filename`` and, for a renamed entry, ``previous_filename`` —
    a rename that lands a core path under a new name must still classify as
    core. ``gh pr view --json files`` (the GraphQL-backed shortcut) carries no
    previous-filename field and silently drops this; callers must use the
    REST files endpoint (``gh api repos/{o}/{r}/pulls/{n}/files``) instead.
    """
    paths: list[str] = []
    for entry in json.loads(files_json):
        filename = entry.get("filename")
        if filename:
            paths.append(filename)
        previous = entry.get("previous_filename")
        if previous:
            paths.append(previous)
    return paths


# --- diff-coverage comment -------------------------------------------------

# stable marker so the bot upserts one comment per PR instead of piling up.
DIFF_COVERAGE_MARKER = "<!-- vouch-bot: diff-coverage -->"

_DIFF_COVERAGE_MAX_FILES = 20
_DIFF_COVERAGE_MAX_RANGES = 12


def line_ranges(lines: Iterable[int]) -> list[tuple[int, int]]:
    """Collapse sorted line numbers into inclusive (start, end) runs."""
    out: list[tuple[int, int]] = []
    for line in sorted(set(int(x) for x in lines)):
        if out and line == out[-1][1] + 1:
            out[-1] = (out[-1][0], line)
        else:
            out.append((line, line))
    return out


def format_ranges(ranges: Sequence[tuple[int, int]], *, limit: int) -> str:
    shown = ranges[:limit]
    text = ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in shown
    )
    if len(ranges) > limit:
        text += f", +{len(ranges) - limit} more"
    return text


def diff_coverage_comment(report: Mapping[str, Any]) -> str:
    """Render a diff-cover json report as the PR comment body.

    Passing reports get a short resolved note so a stale failure comment is
    replaced rather than left contradicting a green run.
    """
    violations = int(report.get("total_num_violations") or 0)
    total = int(report.get("total_num_lines") or 0)
    percent = report.get("total_percent_covered")
    src_stats = report.get("src_stats") or {}

    if not violations:
        body = [
            DIFF_COVERAGE_MARKER,
            "**diff coverage: 100%** — every python line this PR changes under "
            "`src/vouch/` is executed by a test.",
        ]
        if not total:
            body[1] = (
                "**diff coverage: n/a** — this PR changes no python under "
                "`src/vouch/`, so there is nothing for the gate to measure."
            )
        return "\n\n".join(body)

    pct = f"{float(percent):.0f}%" if percent is not None else "unknown"
    lines = [
        DIFF_COVERAGE_MARKER,
        f"**diff coverage: {pct}** — {violations} of {total} changed "
        f"python line(s) under `src/vouch/` are not executed by any test.",
        "every line this PR adds or changes has to be covered before it can "
        "merge. the uncovered lines:",
    ]

    paths = sorted(src_stats)
    for path in paths[:_DIFF_COVERAGE_MAX_FILES]:
        stats = src_stats.get(path) or {}
        ranges = line_ranges(stats.get("violation_lines") or [])
        if not ranges:
            continue
        where = format_ranges(ranges, limit=_DIFF_COVERAGE_MAX_RANGES)
        lines.append(f"- `{path}` — line(s) {where}")
    if len(paths) > _DIFF_COVERAGE_MAX_FILES:
        lines.append(f"- …and {len(paths) - _DIFF_COVERAGE_MAX_FILES} more file(s)")

    lines.append(
        "reproduce locally: `pytest --cov=vouch --cov-report=xml` then "
        "`diff-cover coverage.xml --compare-branch origin/test "
        "--include 'src/vouch/*' --fail-under 100`."
    )
    return "\n\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vouch.pr_bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify")
    c.add_argument("--files-file", required=True)
    c.add_argument("--print-klass", action="store_true")

    cf = sub.add_parser("changed-files")
    cf.add_argument("--json-file", required=True)

    for name in ("core-touched", "ui-touched"):
        sp = sub.add_parser(name)
        sp.add_argument("--files-file", required=True)

    t = sub.add_parser("trust")
    t.add_argument("--author-association", required=True)
    t.add_argument("--actor", required=True)

    s = sub.add_parser("has-screenshots")
    s.add_argument("--body-file", required=True)

    dc = sub.add_parser("diff-coverage-comment")
    dc.add_argument("--report-file", required=True)

    a = sub.add_parser("should-arm")
    a.add_argument("--files-file", required=True)
    a.add_argument("--ci", required=True, choices=["passing", "failing"])
    a.add_argument("--verdict", required=True)
    a.add_argument("--draft", action="store_true")

    ns = p.parse_args(argv)

    if ns.cmd == "classify":
        changed = _read_lines(ns.files_file)
        sys.stdout.write(klass(changed) if ns.print_klass else json.dumps(classify(changed)))
        return 0
    if ns.cmd == "changed-files":
        with open(ns.json_file, encoding="utf-8") as fh:
            paths = extract_changed_paths(fh.read())
        sys.stdout.write("\n".join(paths))
        return 0
    if ns.cmd == "core-touched":
        return 0 if classify(_read_lines(ns.files_file))["is_core"] else 1
    if ns.cmd == "ui-touched":
        return 0 if _touches(_read_lines(ns.files_file), UI_GLOBS) else 1
    if ns.cmd == "trust":
        return 0 if is_trusted(ns.author_association, ns.actor) else 1
    if ns.cmd == "has-screenshots":
        with open(ns.body_file, encoding="utf-8") as fh:
            return 0 if has_before_after_screenshots(fh.read()) else 1
    if ns.cmd == "diff-coverage-comment":
        with open(ns.report_file, encoding="utf-8") as fh:
            loaded = json.load(fh)
        report = loaded if isinstance(loaded, dict) else {}
        sys.stdout.write(diff_coverage_comment(report))
        return 0
    if ns.cmd == "should-arm":
        c2 = classify(_read_lines(ns.files_file))
        ok = should_arm_automerge(is_core=c2["is_core"], ci_passing=ns.ci == "passing",
                                  claude_verdict=ns.verdict, is_draft=ns.draft)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
