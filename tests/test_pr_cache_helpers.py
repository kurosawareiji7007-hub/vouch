"""The `gh` shell-out, LLM analysis, and parsing helpers in `pr_cache`.

Every one of these is an integration edge — a subprocess, a network call, or a
model's free-text output — and every one is documented as best-effort: a
failure must degrade (`None`, `[]`, `""`) rather than abort a `pr-cache build`.
That contract was entirely untested, so a raised exception anywhere in here
would have taken down the whole cache build.

No test touches the network or a real `gh`; the process and urlopen boundaries
are stubbed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from vouch import pr_cache
from vouch.pr_cache import GHError, RepoRef

REPO = RepoRef(owner="acme-example", name="widget")


class _Res:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def gh_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


# --- default_cache_dir ---------------------------------------------------


def test_cache_dir_honours_the_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VOUCH_PR_CACHE_DIR", str(tmp_path / "mine"))
    assert pr_cache.default_cache_dir() == tmp_path / "mine"


def test_cache_dir_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VOUCH_PR_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pr_cache.default_cache_dir() == tmp_path / "xdg" / "vouch" / "pr-cache"


def test_cache_dir_falls_back_to_dot_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VOUCH_PR_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    expected = tmp_path / "home" / ".cache" / "vouch" / "pr-cache"
    assert pr_cache.default_cache_dir() == expected


# --- _run_gh -------------------------------------------------------------


def test_run_gh_requires_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(GHError, match="GitHub CLI"):
        pr_cache._run_gh(["pr", "list"])


def test_run_gh_returns_stdout(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _Res(0, stdout='{"ok":true}')
    )
    assert pr_cache._run_gh(["pr", "list"]) == '{"ok":true}'


def test_run_gh_raises_on_timeout(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _timeout(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="gh", timeout=60)

    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(GHError, match="timed out"):
        pr_cache._run_gh(["pr", "list"])


def test_run_gh_surfaces_stderr_on_failure(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _Res(1, stderr="gh auth required")
    )
    with pytest.raises(GHError, match="gh auth required"):
        pr_cache._run_gh(["pr", "list"])


def test_run_gh_falls_back_to_stdout_when_stderr_is_empty(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _Res(1, stdout="rate limited")
    )
    with pytest.raises(GHError, match="rate limited"):
        pr_cache._run_gh(["pr", "list"])


# --- _gh_pr_files --------------------------------------------------------


def test_pr_files_returns_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"files": [{"path": "a.py"}, {"path": "b.py"}]})
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    assert pr_cache._gh_pr_files(REPO, 1) == ["a.py", "b.py"]


def test_pr_files_skips_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"files": [{"path": "a.py"}, {"no": "path"}, "junk"]})
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    assert pr_cache._gh_pr_files(REPO, 1) == ["a.py"]


def test_pr_files_degrades_on_gh_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> str:
        raise GHError("no auth")

    monkeypatch.setattr(pr_cache, "_run_gh", _boom)
    assert pr_cache._gh_pr_files(REPO, 1) == []


def test_pr_files_handles_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: "")
    assert pr_cache._gh_pr_files(REPO, 1) == []


# --- _gh_pr_review_comments ---------------------------------------------


def test_review_comments_concatenates_comments_and_reviews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({
        "comments": [{"body": "please rebase", "author": {"login": "maintainer"}}],
        "reviews": [
            {"body": "wrong approach", "author": {"login": "reviewer"},
             "state": "CHANGES_REQUESTED"},
        ],
    })
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    out = pr_cache._gh_pr_review_comments(REPO, 1)
    assert "[comment by maintainer]" in out
    assert "please rebase" in out
    assert "[review by reviewer (CHANGES_REQUESTED)]" in out
    assert "wrong approach" in out


def test_review_comments_skips_empty_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({
        "comments": [{"body": "   ", "author": {"login": "x"}}, {"body": None}],
        "reviews": [{"body": "", "author": {"login": "y"}, "state": "APPROVED"}],
    })
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    assert pr_cache._gh_pr_review_comments(REPO, 1) == ""


def test_review_comments_defaults_a_missing_author(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"comments": [{"body": "anon note"}], "reviews": []})
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    assert "[comment by ?]" in pr_cache._gh_pr_review_comments(REPO, 1)


def test_review_comments_tolerates_null_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"comments": [None], "reviews": [None]})
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: payload)
    assert pr_cache._gh_pr_review_comments(REPO, 1) == ""


def test_review_comments_degrades_on_gh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a: Any, **_k: Any) -> str:
        raise GHError("rate limited")

    monkeypatch.setattr(pr_cache, "_run_gh", _boom)
    assert pr_cache._gh_pr_review_comments(REPO, 1) == ""


def test_review_comments_handles_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_cache, "_run_gh", lambda *_a, **_k: "")
    assert pr_cache._gh_pr_review_comments(REPO, 1) == ""


# --- _parse_analysis_json -----------------------------------------------


def test_parse_analysis_json_extracts_a_wrapped_blob() -> None:
    raw = 'Sure! Here you go:\n{"reason": "stale"}\nHope that helps.'
    assert pr_cache._parse_analysis_json(raw) == {"reason": "stale"}


def test_parse_analysis_json_on_plain_json() -> None:
    assert pr_cache._parse_analysis_json('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_parse_analysis_json_rejects_blank(raw: str | None) -> None:
    assert pr_cache._parse_analysis_json(raw) is None  # type: ignore[arg-type]


def test_parse_analysis_json_rejects_text_without_braces() -> None:
    assert pr_cache._parse_analysis_json("no json here") is None


def test_parse_analysis_json_rejects_reversed_braces() -> None:
    assert pr_cache._parse_analysis_json("} not really {") is None


def test_parse_analysis_json_rejects_invalid_json() -> None:
    assert pr_cache._parse_analysis_json("{not: valid, json}") is None


# --- _analyze_via_claude_cli --------------------------------------------


def test_claude_cli_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert pr_cache._analyze_via_claude_cli("prompt", 5.0) is None


def test_claude_cli_returns_stdout(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _Res(0, stdout='{"reason":"stale"}')
    )
    assert pr_cache._analyze_via_claude_cli("prompt", 5.0) == '{"reason":"stale"}'


def test_claude_cli_timeout_returns_none(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _timeout(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="claude", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert pr_cache._analyze_via_claude_cli("prompt", 5.0) is None


def test_claude_cli_nonzero_exit_returns_none(
    gh_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _Res(1, stderr="not logged in")
    )
    assert pr_cache._analyze_via_claude_cli("prompt", 5.0) is None


# --- _analyze_via_anthropic_api -----------------------------------------


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None


def test_anthropic_api_without_a_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) is None


def test_anthropic_api_joins_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    body = json.dumps({
        "content": [
            {"type": "text", "text": '{"reason":'},
            {"type": "thinking", "text": "ignored"},
            {"type": "text", "text": '"stale"}'},
        ]
    }).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Resp(body))
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) == '{"reason":"stale"}'


def test_anthropic_api_honours_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.invalid/")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    seen: dict[str, Any] = {}

    def _urlopen(req: Any, timeout: float | None = None) -> _Resp:
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data)
        return _Resp(json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) == "ok"
    assert seen["url"] == "https://proxy.invalid/v1/messages"
    assert seen["payload"]["model"] == "claude-sonnet-5"


def test_anthropic_api_network_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) is None


def test_anthropic_api_timeout_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise TimeoutError("too slow")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) is None


def test_anthropic_api_non_json_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_a, **_k: _Resp(b"<html>502</html>")
    )
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) is None


def test_anthropic_api_empty_text_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_a, **_k: _Resp(json.dumps({"content": []}).encode()),
    )
    assert pr_cache._analyze_via_anthropic_api("prompt", 5.0) is None


# --- _labels -------------------------------------------------------------


def test_labels_reads_gh_label_objects() -> None:
    assert pr_cache._labels([{"name": "bug"}, {"name": "wontfix"}]) == [
        "bug", "wontfix",
    ]


def test_labels_accepts_bare_strings() -> None:
    assert pr_cache._labels(["bug", "wontfix"]) == ["bug", "wontfix"]


def test_labels_skips_nameless_objects() -> None:
    assert pr_cache._labels([{"name": ""}, {"colour": "red"}, {"name": "keep"}]) == [
        "keep",
    ]


@pytest.mark.parametrize("raw", [None, "bug", 7, {"name": "bug"}])
def test_labels_rejects_a_non_list(raw: Any) -> None:
    assert pr_cache._labels(raw) == []


# --- similarity helpers --------------------------------------------------


def test_jaccard_identical_sets_is_one() -> None:
    assert pr_cache._jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)


def test_jaccard_disjoint_sets_is_zero() -> None:
    assert pr_cache._jaccard({"a"}, {"b"}) == pytest.approx(0.0)


def test_jaccard_partial_overlap() -> None:
    assert pr_cache._jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_jaccard_with_an_empty_set() -> None:
    assert pr_cache._jaccard(set(), {"a"}) == pytest.approx(0.0)
    assert pr_cache._jaccard(set(), set()) == pytest.approx(0.0)


def test_containment_is_one_when_a_is_a_subset() -> None:
    assert pr_cache._containment({"a"}, {"a", "b"}) == pytest.approx(1.0)


def test_containment_partial() -> None:
    # overlap coefficient: |A ∩ B| / min(|A|, |B|), so the smaller side is the
    # denominator -- a subset always scores 1.0 regardless of the other's size
    assert pr_cache._containment({"a", "b", "c"}, {"a", "d"}) == pytest.approx(0.5)
    assert pr_cache._containment({"a", "b"}, {"a"}) == pytest.approx(1.0)


def test_containment_with_an_empty_set() -> None:
    assert pr_cache._containment(set(), {"a"}) == pytest.approx(0.0)
    assert pr_cache._containment({"a"}, set()) == pytest.approx(0.0)


# --- _now_iso ------------------------------------------------------------


def test_now_iso_is_a_utc_timestamp() -> None:
    stamp = pr_cache._now_iso()
    assert stamp.endswith("Z")
    assert len(stamp) == 20
