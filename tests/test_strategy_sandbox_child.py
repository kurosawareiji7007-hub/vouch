"""The sandbox child half of `vouch.strategy`.

These are the functions that run inside `python -I -m vouch.strategy --child`.
They cannot be covered by exercising `run_sandboxed` for real: that child is
spawned with `env={"PATH": ...}`, which strips `COVERAGE_PROCESS_START`, so the
subprocess is deliberately unmeasured.

They also cannot be called naively in-process — `sys.addaudithook` is permanent
for the life of the interpreter, and `_child_main` does `os.dup2(devnull, 1)`,
which would silence pytest's own stdout for every later test. So each hazardous
primitive is intercepted:

* `sys.addaudithook` is swapped for a collector, which both runs
  `_install_audit_hook` to completion and hands back the closure so `_hook`'s
  own branches can be driven directly.
* `resource.setrlimit` is stubbed — the real function would drop this process
  to `RLIMIT_NOFILE = 64`.
* `os.dup2` is stubbed so fd 1 survives.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vouch import strategy as strat
from vouch.strategy import Candidate


def _candidates() -> list[Candidate]:
    return [
        Candidate(kind="claim", id="c1", summary="the review gate", score=0.9),
        Candidate(kind="claim", id="c2", summary="something else", score=0.4),
    ]


@pytest.fixture
def captured_hook(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Run `_install_audit_hook` without arming a permanent audit hook."""
    hooks: list[Any] = []
    monkeypatch.setattr(sys, "addaudithook", hooks.append)
    return hooks


# --- _install_audit_hook / _hook -----------------------------------------


def test_install_audit_hook_registers_exactly_one_hook(
    captured_hook: list[Any]
) -> None:
    strat._install_audit_hook()
    assert len(captured_hook) == 1
    assert callable(captured_hook[0])


def test_hook_blocks_each_exact_blocked_event(captured_hook: list[Any]) -> None:
    strat._install_audit_hook()
    hook = captured_hook[0]
    for event in strat._BLOCKED_EXACT:
        with pytest.raises(PermissionError, match="blocked in strategy sandbox"):
            hook(event, ())


def test_hook_blocks_each_blocked_prefix(captured_hook: list[Any]) -> None:
    strat._install_audit_hook()
    hook = captured_hook[0]
    for prefix in strat._BLOCKED_PREFIXES:
        with pytest.raises(PermissionError, match="blocked in strategy sandbox"):
            hook(f"{prefix}something", ())


def test_hook_allows_an_unrelated_event(captured_hook: list[Any]) -> None:
    strat._install_audit_hook()
    assert captured_hook[0]("object.__getattr__", ()) is None


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "wb"])
def test_hook_blocks_writeish_open_modes(
    captured_hook: list[Any], mode: str
) -> None:
    strat._install_audit_hook()
    hook = captured_hook[0]
    with pytest.raises(PermissionError, match="filesystem writes are blocked"):
        hook("open", ("/tmp/x", mode, 0))


def test_hook_allows_a_read_only_open(captured_hook: list[Any]) -> None:
    # reads must stay allowed: the sandbox has to be able to import numpy
    strat._install_audit_hook()
    assert captured_hook[0]("open", ("/tmp/x", "r", 0)) is None


@pytest.mark.parametrize(
    "flags",
    [os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_APPEND, os.O_WRONLY | os.O_CREAT],
)
def test_hook_blocks_writeish_open_flags(
    captured_hook: list[Any], flags: int
) -> None:
    strat._install_audit_hook()
    hook = captured_hook[0]
    with pytest.raises(PermissionError, match="filesystem writes are blocked"):
        hook("open", ("/tmp/x", None, flags))


def test_hook_allows_read_only_open_flags(captured_hook: list[Any]) -> None:
    strat._install_audit_hook()
    assert captured_hook[0]("open", ("/tmp/x", None, os.O_RDONLY)) is None


def test_hook_tolerates_a_short_open_arg_tuple(captured_hook: list[Any]) -> None:
    strat._install_audit_hook()
    assert captured_hook[0]("open", ()) is None


def test_hook_reads_the_guard_sets_from_closure_cells(
    captured_hook: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # the documented hardening: reassigning the module global must not disarm
    # the installed hook, because it reads a closure cell instead
    strat._install_audit_hook()
    hook = captured_hook[0]
    monkeypatch.setattr(strat, "_BLOCKED_EXACT", frozenset())
    monkeypatch.setattr(strat, "_BLOCKED_PREFIXES", ())
    event = next(iter(strat._BLOCKED_EXACT)) if strat._BLOCKED_EXACT else "socket.connect"
    with pytest.raises(PermissionError):
        hook(event, ())


# --- _apply_rlimits ------------------------------------------------------


def test_apply_rlimits_sets_each_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource

    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(
        resource, "setrlimit", lambda res, pair: calls.append((res, pair))
    )
    strat._apply_rlimits(1024, 5)
    limited = {res for res, _ in calls}
    assert resource.RLIMIT_CPU in limited
    assert resource.RLIMIT_AS in limited
    assert resource.RLIMIT_NOFILE in limited


def test_apply_rlimits_respects_a_finite_hard_limit(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import resource

    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (0, 4))
    monkeypatch.setattr(
        resource, "setrlimit", lambda res, pair: calls.append((res, pair))
    )
    strat._apply_rlimits(1024, 999)
    # never raise above the inherited hard ceiling
    assert all(soft <= 4 for _res, (soft, _hard) in calls)


def test_apply_rlimits_swallows_setrlimit_errors(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import resource

    def _refuse(_res: int, _pair: tuple[int, int]) -> None:
        raise ValueError("not permitted")

    monkeypatch.setattr(resource, "getrlimit", lambda _res: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", _refuse)
    strat._apply_rlimits(1024, 5)  # must not raise


def test_apply_rlimits_is_a_noop_without_the_resource_module(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # windows has no `resource`; the audit hook still applies there
    monkeypatch.setitem(sys.modules, "resource", None)
    strat._apply_rlimits(1024, 5)


# --- _child_main ---------------------------------------------------------


@pytest.fixture
def strategy_file(tmp_path: Path) -> Path:
    path = tmp_path / "reverse_strategy.py"
    path.write_text(
        "def rank(query, candidates, *, limit):\n"
        "    return [c.id for c in reversed(candidates)][:limit]\n",
        encoding="utf-8",
    )
    return path


def _run_child(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> tuple[int, str]:
    """Drive `_child_main` with fd 1 and the audit hook neutralised."""
    written: list[bytes] = []
    monkeypatch.setattr(sys, "addaudithook", lambda _h: None)
    monkeypatch.setattr(strat, "_apply_rlimits", lambda *_a, **_k: None)
    monkeypatch.setattr(os, "dup2", lambda _a, _b: None)
    monkeypatch.setattr(os, "write", lambda _fd, data: written.append(data) or len(data))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    rc = strat._child_main()
    return rc, b"".join(written).decode("utf-8")


def test_child_main_returns_the_strategy_order(
    monkeypatch: pytest.MonkeyPatch, strategy_file: Path
) -> None:
    rc, out = _run_child(
        monkeypatch,
        {
            "path": str(strategy_file),
            "query": "gate",
            "limit": 10,
            "mem_bytes": 1024,
            "cpu_seconds": 5,
            "candidates": [
                {"kind": "claim", "id": "c1", "summary": "one", "score": 0.9},
                {"kind": "claim", "id": "c2", "summary": "two", "score": 0.4},
            ],
        },
    )
    assert rc == 0
    assert json.loads(out)["ordered"] == ["c2", "c1"]


def test_child_main_drops_ids_the_strategy_invented(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a submission must not be able to inject ids that were never candidates
    path = tmp_path / "liar.py"
    path.write_text(
        "def rank(query, candidates, *, limit):\n"
        "    return ['c1', 'not-a-candidate']\n",
        encoding="utf-8",
    )
    _rc, out = _run_child(
        monkeypatch,
        {
            "path": str(path),
            "query": "q",
            "limit": 10,
            "mem_bytes": 1024,
            "cpu_seconds": 5,
            "candidates": [
                {"kind": "claim", "id": "c1", "summary": "one", "score": 0.9}
            ],
        },
    )
    assert json.loads(out)["ordered"] == ["c1"]


def test_child_main_honours_the_limit(
    monkeypatch: pytest.MonkeyPatch, strategy_file: Path
) -> None:
    _rc, out = _run_child(
        monkeypatch,
        {
            "path": str(strategy_file),
            "query": "q",
            "limit": 1,
            "mem_bytes": 1024,
            "cpu_seconds": 5,
            "candidates": [
                {"kind": "claim", "id": "c1", "summary": "one", "score": 0.9},
                {"kind": "claim", "id": "c2", "summary": "two", "score": 0.4},
            ],
        },
    )
    assert json.loads(out)["ordered"] == ["c2"]


# --- main ----------------------------------------------------------------


def test_main_dispatches_to_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strat, "_child_main", lambda: 0)
    assert strat.main(["--child"]) == 0


def test_main_without_child_prints_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert strat.main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_main_reads_sys_argv_when_given_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["vouch.strategy"])
    assert strat.main() == 2
    assert "usage:" in capsys.readouterr().out


# --- run_sandboxed failure modes -----------------------------------------


def _fake_proc(returncode: int, stdout: str) -> Any:
    class _P:
        pass

    p = _P()
    p.returncode = returncode  # type: ignore[attr-defined]
    p.stdout = stdout  # type: ignore[attr-defined]
    return p


def test_run_sandboxed_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    def _timeout(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_returns_none_on_oserror(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no interpreter")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_returns_none_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _fake_proc(1, ""))
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_returns_none_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _fake_proc(0, "not json at all")
    )
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_returns_none_when_ordered_is_missing(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _fake_proc(0, '{"other": 1}')
    )
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_returns_none_when_ordered_is_not_a_list(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _fake_proc(0, '{"ordered": "c1"}')
    )
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) is None


def test_run_sandboxed_coerces_ids_to_strings(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _fake_proc(0, '{"ordered": [1, "c2"]}')
    )
    assert strat.run_sandboxed("s.py", "q", _candidates(), limit=5) == ["1", "c2"]


def test_run_sandboxed_really_runs_a_strategy_file(strategy_file: Path) -> None:
    # one end-to-end pass through the real subprocess, so the wiring is proven
    # even though the child's own lines are measured by the tests above
    out = strat.run_sandboxed(str(strategy_file), "q", _candidates(), limit=5)
    assert out == ["c2", "c1"]


# --- load_from_path / _strategy_from_module ------------------------------


def test_load_from_path_accepts_a_strategy_object(tmp_path: Path) -> None:
    path = tmp_path / "obj_strategy.py"
    path.write_text(
        "class _S:\n"
        "    def rank(self, query, candidates, *, limit):\n"
        "        return [c.id for c in candidates][:limit]\n"
        "STRATEGY = _S()\n",
        encoding="utf-8",
    )
    assert strat.load_from_path(path).rank("q", _candidates(), limit=1) == ["c1"]


def test_load_from_path_rejects_a_module_without_a_strategy(tmp_path: Path) -> None:
    path = tmp_path / "empty_strategy.py"
    path.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define STRATEGY"):
        strat.load_from_path(path)


def test_load_from_path_rejects_an_unloadable_file(tmp_path: Path) -> None:
    with pytest.raises((ValueError, FileNotFoundError, ImportError)):
        strat.load_from_path(tmp_path / "missing.py")


def test_load_from_path_rejects_a_file_with_no_import_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    path = tmp_path / "s.py"
    path.write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_a: None)
    with pytest.raises(ValueError, match="cannot load strategy"):
        strat.load_from_path(path)


def test_load_dotted_imports_a_shipped_strategy() -> None:
    # the trusted path: a merged strategy resolved from config by dotted name
    loaded = strat.load_dotted("vouch.strategies.provenance")
    assert hasattr(loaded, "rank")


# --- apply_ordering ------------------------------------------------------


def _hits() -> list[strat.Hit]:
    return [
        ("claim", "c1", "one", 0.9),
        ("claim", "c2", "two", 0.5),
        ("claim", "c3", "three", 0.1),
    ]


def test_apply_ordering_reorders_the_hits() -> None:
    out = strat.apply_ordering(["c3", "c1", "c2"], _hits())
    assert [h[1] for h in out] == ["c3", "c1", "c2"]


def test_apply_ordering_drops_ids_that_are_not_hits() -> None:
    out = strat.apply_ordering(["c3", "invented"], _hits())
    # the invariant: reorder yes, fabricate no
    assert [h[1] for h in out] == ["c3", "c1", "c2"]


def test_apply_ordering_ignores_a_repeated_id() -> None:
    out = strat.apply_ordering(["c2", "c2"], _hits())
    assert [h[1] for h in out] == ["c2", "c1", "c3"]


def test_apply_ordering_appends_unmentioned_hits_in_original_order() -> None:
    out = strat.apply_ordering(["c2"], _hits())
    assert [h[1] for h in out] == ["c2", "c1", "c3"]


def test_apply_ordering_with_an_empty_ordering_keeps_backend_order() -> None:
    assert strat.apply_ordering([], _hits()) == _hits()


def test_apply_ordering_dedupes_duplicate_hits_by_id() -> None:
    hits = [*_hits(), ("claim", "c1", "one-again", 0.2)]
    out = strat.apply_ordering(["c1"], hits)
    assert [h[1] for h in out] == ["c1", "c2", "c3"]


# --- SandboxProxy --------------------------------------------------------


def test_sandbox_proxy_resolves_the_path(tmp_path: Path) -> None:
    path = tmp_path / "s.py"
    path.write_text("def rank(q, c, *, limit):\n    return []\n", encoding="utf-8")
    proxy = strat.SandboxProxy(path)
    assert proxy.path == str(path.resolve())
    assert proxy.failures == 0


def test_sandbox_proxy_returns_the_child_ordering(strategy_file: Path) -> None:
    proxy = strat.SandboxProxy(strategy_file)
    assert proxy.rank("q", _candidates(), limit=5) == ["c2", "c1"]
    assert proxy.failures == 0


def test_sandbox_proxy_counts_failures_and_returns_empty(
    monkeypatch: pytest.MonkeyPatch, strategy_file: Path
) -> None:
    monkeypatch.setattr(strat, "run_sandboxed", lambda *_a, **_k: None)
    proxy = strat.SandboxProxy(strategy_file)
    assert proxy.rank("q", _candidates(), limit=5) == []
    assert proxy.failures == 1
    # an empty ordering means "keep the backend's order", not an aborted run
    assert strat.apply_ordering([], _hits()) == _hits()


def test_sandbox_proxy_honours_custom_limits(strategy_file: Path) -> None:
    proxy = strat.SandboxProxy(strategy_file, timeout_s=5.0, mem_mb=256, cpu_s=3)
    assert proxy.timeout_s == 5.0
    assert proxy.mem_mb == 256
    assert proxy.cpu_s == 3
    assert proxy.rank("q", _candidates(), limit=5) == ["c2", "c1"]
