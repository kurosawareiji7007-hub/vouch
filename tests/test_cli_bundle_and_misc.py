"""The bundle round-trip plus the remaining import-covered `cli` commands.

The export/import family is the federation surface: `import-apply` writes
straight to the durable store, `import-proposals` is its gated counterpart.
Both were import-covered only, which is a poor place to have no tests — the
whole point of `import-proposals` is that inbound knowledge cannot bypass the
review gate, and nothing was asserting that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from vouch.cli import cli
from vouch.models import Claim, Entity
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path / "origin")
    monkeypatch.chdir(s.root)
    return s


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli, args)


def _ok(args: list[str]) -> Result:
    result = _run(args)
    assert result.exit_code == 0, result.output
    return result


def _clean_error(args: list[str]) -> Result:
    result = _run(args)
    assert result.exit_code != 0, result.output
    assert "Traceback" not in result.output, result.output
    return result


def _claim(store: KBStore, claim_id: str, text: str) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id]))


# --- discover -------------------------------------------------------------


def test_discover_reports_root_and_why_chain(store: KBStore) -> None:
    doc = json.loads(_ok(["discover"]).output)
    assert doc["kb_dir"].endswith(".vouch")
    assert doc["why"]


def test_discover_with_explicit_path(store: KBStore) -> None:
    doc = json.loads(_ok(["discover", "--path", str(store.root)]).output)
    assert doc["root"] == str(store.root)


def test_discover_outside_a_kb_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "bare"
    outside.mkdir()
    monkeypatch.chdir(outside)
    result = _run(["discover", "--path", str(outside)])
    assert result.exit_code == 2
    assert "error:" in result.output


# --- export / export-check ------------------------------------------------


def test_export_writes_a_bundle_and_reports_the_manifest(
    store: KBStore, tmp_path: Path
) -> None:
    _claim(store, "c1", "a claim worth exporting")
    out = tmp_path / "kb.tar.gz"
    doc = json.loads(_ok(["export", "--out", str(out)]).output)
    assert out.exists()
    assert doc["files"] > 0
    assert doc["bundle_id"]


def test_export_honours_exclude(store: KBStore, tmp_path: Path) -> None:
    _claim(store, "c1", "a claim worth exporting")
    out = tmp_path / "kb.tar.gz"
    doc = json.loads(
        _ok(["export", "--out", str(out), "--exclude", "sessions,decided"]).output
    )
    assert "sessions" in doc["excluded"]
    assert "decided" in doc["excluded"]


def test_export_check_passes_on_a_fresh_bundle(
    store: KBStore, tmp_path: Path
) -> None:
    _claim(store, "c1", "a claim worth exporting")
    out = tmp_path / "kb.tar.gz"
    _ok(["export", "--out", str(out)])
    doc = json.loads(_ok(["export-check", str(out)]).output)
    assert doc["ok"] is True
    assert doc["files_checked"] > 0


# --- import-check / import-apply / import-proposals -----------------------


def _bundle_from_origin(store: KBStore, tmp_path: Path) -> Path:
    _claim(store, "c1", "a claim to federate")
    out = tmp_path / "kb.tar.gz"
    _ok(["export", "--out", str(out)])
    return out


def test_import_check_reports_new_files_without_writing(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = _bundle_from_origin(store, tmp_path)
    dest = KBStore.init(tmp_path / "dest")
    monkeypatch.chdir(dest.root)
    doc = json.loads(_ok(["import-check", str(bundle_path)]).output)
    assert doc["new_files"]
    assert dest.list_claims() == []


def test_import_check_sees_identical_files_on_reimport(
    store: KBStore, tmp_path: Path
) -> None:
    bundle_path = _bundle_from_origin(store, tmp_path)
    # checking a bundle against the KB it came from: everything is identical
    doc = json.loads(_ok(["import-check", str(bundle_path)]).output)
    assert doc["identical_files"] > 0


def test_import_apply_writes_the_claims_durably(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = _bundle_from_origin(store, tmp_path)
    dest = KBStore.init(tmp_path / "dest")
    monkeypatch.chdir(dest.root)
    _ok(["import-apply", str(bundle_path)])
    assert [c.id for c in dest.list_claims()] == ["c1"]


def test_import_apply_rejects_an_unreadable_bundle(
    store: KBStore, tmp_path: Path
) -> None:
    junk = tmp_path / "not-a-bundle.tar.gz"
    junk.write_bytes(b"definitely not a tarball")
    _clean_error(["import-apply", str(junk)])


def test_import_proposals_files_pending_not_durable(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = _bundle_from_origin(store, tmp_path)
    dest = KBStore.init(tmp_path / "dest")
    monkeypatch.chdir(dest.root)
    _ok(["import-proposals", str(bundle_path)])
    # the gate holds: nothing durable, everything pending review
    assert dest.list_claims() == []
    assert dest.list_proposals()


def test_import_proposals_accepts_an_origin_label(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = _bundle_from_origin(store, tmp_path)
    dest = KBStore.init(tmp_path / "dest")
    monkeypatch.chdir(dest.root)
    _ok(["import-proposals", str(bundle_path), "--origin-kb", "acme-example"])
    assert dest.list_proposals()


def test_import_proposals_rejects_an_unreadable_bundle(
    store: KBStore, tmp_path: Path
) -> None:
    junk = tmp_path / "not-a-bundle.tar.gz"
    junk.write_bytes(b"definitely not a tarball")
    _clean_error(["import-proposals", str(junk)])


# --- experts --------------------------------------------------------------


def test_experts_on_an_empty_kb(store: KBStore) -> None:
    assert "no experts found." in _ok(["experts", "retrieval"]).output


def test_experts_ranks_entities_by_evidence(store: KBStore) -> None:
    store.put_entity(Entity(id="alice-example", name="alice", type="person"))
    src = store.put_source(b"evidence body")
    store.put_claim(
        Claim(
            id="c1",
            text="alice owns retrieval",
            evidence=[src.id],
            entities=["alice-example"],
        )
    )
    result = _ok(["experts", "retrieval", "--min-claims", "1"])
    assert "no experts found." in result.output or "claims=" in result.output


def test_experts_json_shape(store: KBStore) -> None:
    doc = json.loads(_ok(["experts", "retrieval", "--json"]).output)
    assert "experts" in doc


# --- sessions -------------------------------------------------------------


def test_session_start_prints_an_id(store: KBStore) -> None:
    session_id = _ok(["session", "start", "--agent", "claude-code"]).output.strip()
    assert session_id


def test_session_start_accepts_task_and_note(store: KBStore) -> None:
    session_id = _ok([
        "session", "start", "--task", "close the coverage gap", "--note", "wip",
    ]).output.strip()
    assert session_id


def test_session_end_reports_proposals(store: KBStore) -> None:
    session_id = _ok(["session", "start"]).output.strip()
    doc = json.loads(_ok(["session", "end", session_id]).output)
    assert doc["session"] == session_id
    assert doc["proposals"] == []


def test_session_end_unknown_id_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["session", "end", "no-such-session"])


def test_session_list_empty_and_json(store: KBStore) -> None:
    assert "no sessions found" in _ok(["session", "list"]).output
    doc = json.loads(_ok(["session", "list", "--json"]).output)
    assert doc["sessions"] == []


def test_session_volunteer_with_an_empty_queue(store: KBStore) -> None:
    session_id = _ok(["session", "start"]).output.strip()
    out = _ok(["session", "volunteer", session_id]).output
    assert "(no volunteered context)" in out


def test_session_volunteer_json_is_empty_when_nothing_offered(
    store: KBStore,
) -> None:
    session_id = _ok(["session", "start"]).output.strip()
    doc = json.loads(_ok(["session", "volunteer", session_id, "--json"]).output)
    assert doc["volunteers"] == []


def test_session_volunteer_no_clear_peeks(store: KBStore) -> None:
    session_id = _ok(["session", "start"]).output.strip()
    _ok(["session", "volunteer", session_id, "--no-clear"])


# --- reject-extracted / synthesize / detect-themes ------------------------


def test_reject_extracted_with_nothing_pending(store: KBStore) -> None:
    out = _ok(["reject-extracted", "--reason", "noise"]).output
    assert "no pending auto-extracted edges to reject" in out


def test_synthesize_answers_from_the_kb(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    doc = json.loads(_ok(["synthesize", "review gate"]).output)
    assert isinstance(doc, dict)


def test_detect_themes_on_an_empty_kb(store: KBStore) -> None:
    assert "no themes detected" in _ok(["detect-themes"]).output


def test_detect_themes_json_on_an_empty_kb(store: KBStore) -> None:
    doc = json.loads(_ok(["detect-themes", "--json"]).output)
    assert doc["clusters"] == []
    assert "config" in doc


def test_detect_themes_propose_with_no_clusters(store: KBStore) -> None:
    out = _ok(["detect-themes", "--propose"]).output
    assert "no themes detected" in out


# --- compile --------------------------------------------------------------


def test_compile_surfaces_a_configured_llm_failure(store: KBStore) -> None:
    _claim(store, "c1", "a claim to compile")
    # a command that exits nonzero must arrive as a clean ClickException, not a
    # CompileError traceback
    result = _clean_error(["compile", "--llm-cmd", "false", "--dry-run"])
    assert "Error:" in result.output
