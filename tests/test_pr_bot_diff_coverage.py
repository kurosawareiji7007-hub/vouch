"""The diff-coverage PR comment the bot posts when the gate fails.

The comment is built from diff-cover's json report. File paths in that report
come from the PR's own diff, so an attacker controls them — the renderer must
therefore stay pure string work with no shell or markup escape hatch, and the
workflow posts the result with `--body-file` rather than interpolating it.

The marker is load-bearing: the workflow upserts on it, so one PR gets one
comment instead of a pile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vouch.pr_bot import (
    DIFF_COVERAGE_MARKER,
    diff_coverage_comment,
    format_ranges,
    line_ranges,
    main,
)

# --- line_ranges ---------------------------------------------------------


def test_line_ranges_collapses_a_run() -> None:
    assert line_ranges([1, 2, 3]) == [(1, 3)]


def test_line_ranges_splits_on_a_gap() -> None:
    assert line_ranges([1, 2, 5, 6, 9]) == [(1, 2), (5, 6), (9, 9)]


def test_line_ranges_sorts_and_dedupes() -> None:
    assert line_ranges([5, 1, 2, 2, 1]) == [(1, 2), (5, 5)]


def test_line_ranges_on_empty_input() -> None:
    assert line_ranges([]) == []


def test_line_ranges_accepts_a_single_line() -> None:
    assert line_ranges([42]) == [(42, 42)]


def test_line_ranges_coerces_stringy_numbers() -> None:
    assert line_ranges(["3", "4"]) == [(3, 4)]  # type: ignore[list-item]


# --- format_ranges -------------------------------------------------------


def test_format_ranges_renders_singletons_and_spans() -> None:
    assert format_ranges([(1, 1), (4, 7)], limit=10) == "1, 4-7"


def test_format_ranges_truncates_past_the_limit() -> None:
    ranges = [(n, n) for n in range(1, 6)]
    assert format_ranges(ranges, limit=2) == "1, 2, +3 more"


def test_format_ranges_on_empty_input() -> None:
    assert format_ranges([], limit=5) == ""


# --- diff_coverage_comment: passing ------------------------------------


def test_comment_reports_full_coverage() -> None:
    body = diff_coverage_comment(
        {"total_num_violations": 0, "total_num_lines": 12, "total_percent_covered": 100}
    )
    assert body.startswith(DIFF_COVERAGE_MARKER)
    assert "diff coverage: 100%" in body


def test_comment_reports_nothing_to_measure() -> None:
    # docs-only PRs pass the gate; the comment must not claim a coverage win
    body = diff_coverage_comment({"total_num_violations": 0, "total_num_lines": 0})
    assert "diff coverage: n/a" in body
    assert "no python" in body


def test_comment_on_an_empty_report_does_not_crash() -> None:
    body = diff_coverage_comment({})
    assert body.startswith(DIFF_COVERAGE_MARKER)


# --- diff_coverage_comment: failing -----------------------------------


def _failing() -> dict[str, Any]:
    return {
        "total_num_violations": 6,
        "total_num_lines": 7,
        "total_percent_covered": 14,
        "src_stats": {
            "src/vouch/mod.py": {
                "percent_covered": 33.3,
                "violation_lines": [6, 7],
            },
            "src/vouch/other.py": {
                "percent_covered": 0.0,
                "violation_lines": [1, 2, 3, 4],
            },
        },
    }


def test_comment_names_the_uncovered_lines() -> None:
    body = diff_coverage_comment(_failing())
    assert body.startswith(DIFF_COVERAGE_MARKER)
    assert "diff coverage: 14%" in body
    assert "6 of 7 changed" in body
    assert "`src/vouch/mod.py` — line(s) 6-7" in body
    assert "`src/vouch/other.py` — line(s) 1-4" in body


def test_comment_includes_a_local_reproduction() -> None:
    body = diff_coverage_comment(_failing())
    assert "diff-cover coverage.xml" in body
    assert "--fail-under 100" in body


def test_comment_lists_files_in_a_stable_order() -> None:
    body = diff_coverage_comment(_failing())
    assert body.index("src/vouch/mod.py") < body.index("src/vouch/other.py")


def test_comment_skips_a_file_with_no_violation_lines() -> None:
    report = {
        "total_num_violations": 2,
        "total_num_lines": 4,
        "total_percent_covered": 50,
        "src_stats": {
            "src/vouch/a.py": {"violation_lines": [3, 4]},
            "src/vouch/b.py": {"violation_lines": []},
        },
    }
    body = diff_coverage_comment(report)
    assert "src/vouch/a.py" in body
    assert "src/vouch/b.py" not in body


def test_comment_truncates_a_very_wide_pr() -> None:
    stats = {
        f"src/vouch/f{n:02d}.py": {"violation_lines": [1]} for n in range(30)
    }
    report = {
        "total_num_violations": 30,
        "total_num_lines": 60,
        "total_percent_covered": 50,
        "src_stats": stats,
    }
    body = diff_coverage_comment(report)
    assert "and 10 more file(s)" in body


def test_comment_truncates_a_file_with_many_scattered_lines() -> None:
    scattered = list(range(1, 60, 2))  # 30 non-adjacent lines -> 30 ranges
    report = {
        "total_num_violations": len(scattered),
        "total_num_lines": 100,
        "total_percent_covered": 70,
        "src_stats": {"src/vouch/wide.py": {"violation_lines": scattered}},
    }
    body = diff_coverage_comment(report)
    assert "more" in body


def test_comment_handles_a_missing_percentage() -> None:
    report = {
        "total_num_violations": 1,
        "total_num_lines": 2,
        "src_stats": {"src/vouch/a.py": {"violation_lines": [2]}},
    }
    assert "diff coverage: unknown" in diff_coverage_comment(report)


def test_comment_does_not_execute_or_expand_attacker_paths() -> None:
    # a contributor names the file; the renderer must emit it verbatim inside a
    # code span and never build a shell word from it
    nasty = "src/vouch/$(id).py"
    report = {
        "total_num_violations": 1,
        "total_num_lines": 1,
        "total_percent_covered": 0,
        "src_stats": {nasty: {"violation_lines": [1]}},
    }
    body = diff_coverage_comment(report)
    assert f"`{nasty}`" in body
    assert "uid=" not in body


# --- the CLI surface the workflow calls --------------------------------


def test_cli_emits_the_comment(
    tmp_path: Path, capsys: Any
) -> None:
    report = tmp_path / "dc.json"
    report.write_text(json.dumps(_failing()), encoding="utf-8")
    assert main(["diff-coverage-comment", "--report-file", str(report)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(DIFF_COVERAGE_MARKER)
    assert "src/vouch/mod.py" in out


def test_cli_tolerates_a_non_object_report(
    tmp_path: Path, capsys: Any
) -> None:
    report = tmp_path / "dc.json"
    report.write_text("[]", encoding="utf-8")
    assert main(["diff-coverage-comment", "--report-file", str(report)]) == 0
    assert DIFF_COVERAGE_MARKER in capsys.readouterr().out


def test_cli_emits_the_passing_comment(
    tmp_path: Path, capsys: Any
) -> None:
    report = tmp_path / "dc.json"
    report.write_text(
        json.dumps({"total_num_violations": 0, "total_num_lines": 5}),
        encoding="utf-8",
    )
    assert main(["diff-coverage-comment", "--report-file", str(report)]) == 0
    assert "diff coverage: 100%" in capsys.readouterr().out


# --- the rest of the pr_bot CLI the workflows shell out to ---------------
#
# exit-code contract, not text: every one of these is consumed as `if
# python -m vouch.pr_bot ... ; then` in a workflow, so an inverted code would
# silently flip a gate open.


_seq = iter(range(1, 10_000))


def _files(tmp_path: Path, *paths: str) -> str:
    # a fresh name per call: reusing one path silently clobbers an earlier
    # list, which made a core-touching case look non-core
    f = tmp_path / f"files-{next(_seq)}.txt"
    f.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return str(f)


def test_cli_core_touched_exit_codes(tmp_path: Path) -> None:
    assert main(["core-touched", "--files-file",
                 _files(tmp_path, "src/vouch/proposals.py")]) == 0
    assert main(["core-touched", "--files-file",
                 _files(tmp_path, "README.md")]) == 1


def test_cli_ui_touched_exit_codes(tmp_path: Path) -> None:
    assert main(["ui-touched", "--files-file",
                 _files(tmp_path, "webapp/src/App.tsx")]) == 0
    assert main(["ui-touched", "--files-file",
                 _files(tmp_path, "README.md")]) == 1


def test_cli_has_screenshots_exit_codes(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text(
        "before ![a](https://github.com/x/y/assets/1/aaa)\n"
        "after ![b](https://github.com/x/y/assets/1/bbb)\n",
        encoding="utf-8",
    )
    assert main(["has-screenshots", "--body-file", str(body)]) == 0
    body.write_text("no images here\n", encoding="utf-8")
    assert main(["has-screenshots", "--body-file", str(body)]) == 1


def test_cli_should_arm_approves_a_clean_non_core_pr(tmp_path: Path) -> None:
    assert main([
        "should-arm", "--files-file", _files(tmp_path, "docs/guide.md"),
        "--ci", "passing", "--verdict", "APPROVE",
    ]) == 0


def test_cli_should_arm_refuses_core_failing_ci_and_drafts(
    tmp_path: Path
) -> None:
    core = _files(tmp_path, "src/vouch/proposals.py")
    plain = _files(tmp_path, "docs/guide.md")
    # core is never armed, whatever the verdict
    assert main(["should-arm", "--files-file", core,
                 "--ci", "passing", "--verdict", "APPROVE"]) == 1
    assert main(["should-arm", "--files-file", plain,
                 "--ci", "failing", "--verdict", "APPROVE"]) == 1
    assert main(["should-arm", "--files-file", plain,
                 "--ci", "passing", "--verdict", "REQUEST_CHANGES"]) == 1
    assert main(["should-arm", "--files-file", plain, "--ci", "passing",
                 "--verdict", "APPROVE", "--draft"]) == 1
