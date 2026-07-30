"""The maintenance / health / index CLI surface.

`lint`, `doctor`, `fsck`, `reindex`, `audit`, `dedup`, `contradict-scan`,
`provenance rebuild`, `embeddings stats`, `list-skills` and `get-skill` were
all import-covered only. They are the commands a user reaches for when the KB
is already suspect, so a traceback here lands at the worst possible moment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner, Result

from vouch.cli import cli
from vouch.embeddings import register
from vouch.embeddings.base import DEFAULT_MODEL_NAME, Embedder
from vouch.models import Claim
from vouch.proposals import propose_claim
from vouch.storage import KBStore


class _HashEmbedder(Embedder):
    name = "mock"
    version = "1"
    dim = 8

    def encode(self, text: str) -> np.ndarray:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        out = np.array([h[i] / 255.0 for i in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path)
    monkeypatch.chdir(s.root)
    return s


@pytest.fixture
def embedder() -> None:
    register(DEFAULT_MODEL_NAME, _HashEmbedder)


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli, args)


def _ok(args: list[str]) -> Result:
    result = _run(args)
    assert result.exit_code == 0, result.output
    return result


def _no_traceback(args: list[str]) -> Result:
    result = _run(args)
    assert "Traceback" not in result.output, result.output
    return result


def _claim(store: KBStore, claim_id: str, text: str) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id]))


# --- status ---------------------------------------------------------------


def test_status_table_lists_counts(store: KBStore) -> None:
    _claim(store, "c1", "a durable claim")
    out = _ok(["status"]).output
    assert "KB at" in out
    assert "durable:" in out
    assert "pending:" in out
    assert "audit:" in out


def test_status_json_is_machine_readable(store: KBStore) -> None:
    _claim(store, "c1", "a durable claim")
    doc = json.loads(_ok(["status", "--json"]).output)
    assert doc["claims"] == 1
    assert doc["pending_proposals"] == 0
    assert "kb_dir" in doc


def test_status_counts_pending_proposals(store: KBStore) -> None:
    src = store.put_source(b"e")
    propose_claim(store, text="pending one", evidence=[src.id], proposed_by="agent")
    doc = json.loads(_ok(["status", "--json"]).output)
    assert doc["pending_proposals"] == 1


# --- lint / doctor / fsck -------------------------------------------------


def test_lint_on_a_fresh_kb_is_clean(store: KBStore) -> None:
    result = _no_traceback(["lint"])
    assert result.exit_code == 0
    assert "clean" in result.output


def test_lint_accepts_a_stale_day_window(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    result = _no_traceback(["lint", "--stale-days", "1"])
    assert result.exit_code in (0, 1)


def test_doctor_prints_a_counts_footer(store: KBStore) -> None:
    result = _no_traceback(["doctor"])
    assert result.exit_code in (0, 1)
    assert "--" in result.output


def test_fsck_flags_a_missing_index(store: KBStore) -> None:
    # a KB that has never been indexed reports index_missing at info severity,
    # which is advisory: the command still exits 0
    result = _no_traceback(["fsck"])
    assert result.exit_code == 0
    assert "index_missing" in result.output


def test_fsck_is_clean_once_indexed(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    _ok(["index"])
    result = _no_traceback(["fsck"])
    assert result.exit_code == 0
    assert "clean" in result.output


# --- index / provenance ---------------------------------------------------


def test_reindex_rebuilds_fts5_by_default(store: KBStore) -> None:
    _claim(store, "c1", "indexed claim")
    assert "reindex: FTS5 rebuilt" in _ok(["reindex"]).output


def test_reindex_backfills_embeddings_when_asked(
    store: KBStore, embedder: None
) -> None:
    _claim(store, "c1", "claim to embed")
    out = _ok(["reindex", "--embeddings"]).output
    assert "reindex: embeddings backfilled" in out


def test_reindex_force_backfill_re_encodes(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "claim to embed")
    out = _ok(["reindex", "--backfill", "--force"]).output
    assert "reindex: embeddings backfilled" in out


def test_provenance_rebuild_reports_edge_count(store: KBStore) -> None:
    _claim(store, "c1", "a claim with evidence")
    assert "provenance: rebuilt" in _ok(["provenance", "rebuild"]).output


def test_provenance_rebuild_json(store: KBStore) -> None:
    _claim(store, "c1", "a claim with evidence")
    doc = json.loads(_ok(["provenance", "rebuild", "--json"]).output)
    assert isinstance(doc["edges"], int)


def test_embeddings_stats_reports_counts_and_cache(
    store: KBStore, embedder: None
) -> None:
    _claim(store, "c1", "a claim to embed")
    out = _ok(["embeddings", "stats"]).output
    assert "query_cache_entries" in out
    assert "query_cache_hits" in out


# --- advisory scans -------------------------------------------------------


def test_dedup_on_a_fresh_kb_finds_nothing(store: KBStore, embedder: None) -> None:
    assert "dedup: no duplicates found" in _ok(["dedup"]).output


def test_dedup_reports_a_near_duplicate_pair(
    store: KBStore, embedder: None
) -> None:
    _claim(store, "c1", "identical duplicated text")
    _claim(store, "c2", "identical duplicated text")
    out = _ok(["dedup"]).output
    assert "cos=" in out
    assert "claim/c2" in out or "claim/c1" in out


def test_contradict_scan_on_a_fresh_kb_finds_nothing(store: KBStore) -> None:
    out = _ok(["contradict-scan"]).output
    assert "contradict-scan: no candidates found" in out


def test_contradict_scan_dry_run_writes_no_proposals(store: KBStore) -> None:
    _claim(store, "c1", "the gate is enabled")
    _claim(store, "c2", "the gate is not enabled")
    _no_traceback(["contradict-scan", "--dry-run"])
    assert store.list_proposals() == []


# --- audit ----------------------------------------------------------------


def test_audit_tail_lists_events(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    _ok(["approve", pr.id])
    out = _ok(["audit"]).output
    assert "by " in out
    assert "objects=" in out


def test_audit_json_includes_viewer_and_events(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    _ok(["approve", pr.id])
    doc = json.loads(_ok(["audit", "--json"]).output)
    assert "viewer" in doc
    assert doc["events"]


def test_audit_tail_caps_the_event_count(store: KBStore) -> None:
    src = store.put_source(b"e")
    for i in range(3):
        pr = propose_claim(
            store, text=f"claim {i}", evidence=[src.id], proposed_by="agent"
        )
        _ok(["approve", pr.id])
    doc = json.loads(_ok(["audit", "--json", "--tail", "2"]).output)
    assert len(doc["events"]) == 2


def test_audit_echoes_the_viewer_when_scoped(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    _ok(["approve", pr.id])
    result = _ok(["audit", "--project", "acme-example", "--agent", "claude-code"])
    assert "viewer:" in result.output


# --- skills ---------------------------------------------------------------


def test_list_skills_on_a_bare_kb(store: KBStore) -> None:
    out = _ok(["list-skills"]).output
    assert "no skills published" in out or "[" in out


def test_list_skills_json_is_a_list(store: KBStore) -> None:
    doc = json.loads(_ok(["list-skills", "--json"]).output)
    assert isinstance(doc, list)


def test_get_skill_unknown_name_is_a_clean_error(store: KBStore) -> None:
    result = _run(["get-skill", "no-such-skill"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Error:" in result.output
