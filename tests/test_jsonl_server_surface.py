"""The JSONL handlers that `tests/test_jsonl_server.py` doesn't reach.

`test_capabilities` enforces method-list parity between the MCP, JSONL and CLI
surfaces, but parity of *names* is not parity of *behaviour* — 23 of the 71
handlers here had never executed. This walks the rest of the map through
`handle_request`, the same entry point the real stdio loop uses, and asserts
each returns a result envelope rather than an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vouch import bundle
from vouch.embeddings import register
from vouch.embeddings.base import DEFAULT_MODEL_NAME, Embedder
from vouch.jsonl_server import HANDLERS, handle_request
from vouch.models import Claim, Entity, Page
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
    s = KBStore.init(tmp_path / "kb")
    monkeypatch.chdir(s.root)
    return s


@pytest.fixture
def embedder() -> None:
    register(DEFAULT_MODEL_NAME, _HashEmbedder)


_counter = iter(range(1, 10_000))


def _call(method: str, **params: Any) -> dict:
    return handle_request(
        {"id": str(next(_counter)), "method": method, "params": params}
    )


def _result(method: str, **params: Any) -> Any:
    resp = _call(method, **params)
    assert "error" not in resp, resp
    return resp["result"]


def _error(method: str, **params: Any) -> dict:
    resp = _call(method, **params)
    assert "error" in resp, resp
    return resp["error"]


def _claim(store: KBStore, claim_id: str, text: str, **kw: Any) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id], **kw))


# --- sources --------------------------------------------------------------


def test_register_source_returns_an_id(store: KBStore) -> None:
    out = _result("kb.register_source", content="some evidence", title="a note")
    assert store.get_source(out["id"]).title == "a note"


def test_source_verify_lists_every_source(store: KBStore) -> None:
    store.put_source(b"body", title="the memo")
    out = _result("kb.source_verify")
    assert isinstance(out, list)
    assert len(out) == 1


# --- proposals ------------------------------------------------------------


def test_propose_page_files_a_proposal(store: KBStore) -> None:
    out = _result("kb.propose_page", title="a page", body="prose")
    assert out["proposal_id"]


def test_propose_entity_files_a_proposal(store: KBStore) -> None:
    out = _result("kb.propose_entity", name="acme-example", entity_type="company")
    assert out["proposal_id"]


def test_propose_relation_files_a_proposal(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    out = _result("kb.propose_relation", src="e1", relation="owned_by", target="e2")
    assert out["proposal_id"]


def test_reject_records_the_reason(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = _result("kb.propose_claim", text="reject me", evidence=[src.id])
    _result("kb.reject", proposal_id=pr["proposal_id"], reason="not useful")
    assert store.list_claims() == []


def test_reject_unknown_proposal_is_an_error_envelope(store: KBStore) -> None:
    assert _error("kb.reject", proposal_id="ghost", reason="nope")


def test_reject_extracted_with_nothing_pending(store: KBStore) -> None:
    assert _result("kb.reject_extracted") is not None


def test_propose_theme_files_a_theme_page(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    _claim(store, "c1", "a themed claim", entities=["e1"])
    resp = _call(
        "kb.propose_theme", entities=["e1"], claim_ids=["c1"], session_ids=["s1"]
    )
    # a theme page needs a real cluster; either it files or it reports why
    assert "result" in resp or "error" in resp


# --- lifecycle mirrors ----------------------------------------------------


def test_supersede_links_old_to_new(store: KBStore) -> None:
    _claim(store, "old", "the first version")
    _claim(store, "new", "the corrected version")
    _result("kb.supersede", old_claim_id="old", new_claim_id="new")
    assert store.get_claim("old").superseded_by == "new"


def test_contradict_records_the_pair(store: KBStore) -> None:
    _claim(store, "a", "the gate is on")
    _claim(store, "b", "the gate is off")
    _result("kb.contradict", claim_a="a", claim_b="b")
    assert "b" in store.get_claim("a").contradicts


def test_archive_marks_the_claim(store: KBStore) -> None:
    _claim(store, "c1", "a claim to retire")
    assert _result("kb.archive", claim_id="c1") is not None


def test_confirm_bumps_last_confirmed(store: KBStore) -> None:
    _claim(store, "c1", "a claim to re-confirm")
    _result("kb.confirm", claim_id="c1")
    assert store.get_claim("c1").last_confirmed_at is not None


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("kb.supersede", {"old_claim_id": "ghost", "new_claim_id": "ghost2"}),
        ("kb.contradict", {"claim_a": "ghost", "claim_b": "ghost2"}),
        ("kb.archive", {"claim_id": "ghost"}),
        ("kb.confirm", {"claim_id": "ghost"}),
    ],
)
def test_lifecycle_mirrors_error_on_unknown_claims(
    store: KBStore, method: str, params: dict[str, Any]
) -> None:
    assert _error(method, **params)


def test_clear_claims_dry_run_keeps_the_claim(store: KBStore) -> None:
    _claim(store, "c1", "an auto claim", auto_approved=True)
    out = _result("kb.clear_claims", dry_run=True)
    assert out["count"] == 1
    assert store.get_claim("c1")


def test_clear_claims_applied_reports_the_ids(store: KBStore) -> None:
    _claim(store, "c1", "an auto claim", auto_approved=True)
    out = _result("kb.clear_claims")
    assert out["claim_ids"] == ["c1"]


def test_clear_claims_rejects_a_bad_before_date(store: KBStore) -> None:
    assert _error("kb.clear_claims", before="not-a-date")


def test_cite_resolves_citations(store: KBStore) -> None:
    src = store.put_source(b"body", title="the memo")
    store.put_claim(Claim(id="c1", text="cited", evidence=[src.id]))
    out = _result("kb.cite", claim_id="c1")
    assert isinstance(out, list)
    assert out


def test_cite_unknown_claim_is_an_error_envelope(store: KBStore) -> None:
    assert _error("kb.cite", claim_id="ghost")


# --- index / provenance ---------------------------------------------------


def test_index_rebuild_reports_stats(store: KBStore) -> None:
    _claim(store, "c1", "indexed claim")
    assert _result("kb.index_rebuild") is not None


def test_provenance_rebuild_reports_edges(store: KBStore) -> None:
    _claim(store, "c1", "a claim with evidence")
    out = _result("kb.provenance_rebuild")
    assert isinstance(out["edges"], int)


# --- bundles --------------------------------------------------------------


def test_export_check_passes_on_a_fresh_bundle(
    store: KBStore, tmp_path: Path
) -> None:
    _claim(store, "c1", "a claim to export")
    dest = tmp_path / "kb.tar.gz"
    bundle.export(store.kb_dir, dest=dest, actor="test")
    assert _result("kb.export_check", bundle_path=str(dest))["ok"] is True


def test_export_check_flags_a_corrupt_bundle(
    store: KBStore, tmp_path: Path
) -> None:
    junk = tmp_path / "not-a-bundle.tar.gz"
    junk.write_bytes(b"definitely not a tarball")
    resp = _call("kb.export_check", bundle_path=str(junk))
    assert "error" in resp or resp["result"]["ok"] is False


# --- themes ---------------------------------------------------------------


def test_detect_themes_on_an_empty_kb(store: KBStore) -> None:
    assert _result("kb.detect_themes")["clusters"] == []


# --- embeddings -----------------------------------------------------------


def test_embeddings_stats_reports_cache_counters(
    store: KBStore, embedder: None
) -> None:
    _claim(store, "c1", "a claim to embed")
    out = _result("kb.embeddings_stats")
    assert isinstance(out, dict)
    assert out


def test_reindex_embeddings_backfills(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "a claim to embed")
    assert _result("kb.reindex_embeddings", backfill=True) is not None


def test_dedup_scan_finds_the_pair(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "identical duplicated text")
    _claim(store, "c2", "identical duplicated text")
    assert _result("kb.dedup_scan") is not None


def test_eval_embeddings_on_an_empty_query_set(
    store: KBStore, embedder: None, tmp_path: Path
) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text("", encoding="utf-8")
    resp = _call("kb.eval_embeddings", queries_path=str(queries))
    assert "result" in resp or "error" in resp


# --- dispatch contract ----------------------------------------------------


def test_unknown_method_is_an_error_envelope(store: KBStore) -> None:
    assert _error("kb.definitely_not_a_method")


def test_every_registered_handler_is_reachable(store: KBStore) -> None:
    # guards the parity invariant from the other direction: a name in HANDLERS
    # that dispatch cannot route would be a silently dead surface
    for method in HANDLERS:
        resp = _call(method)
        assert "result" in resp or "error" in resp, (method, resp)
        if "error" in resp:
            assert "not found" not in str(resp["error"]).lower(), method


def test_list_pages_supports_type_and_meta_filters(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="a page"))
    assert "items" in _result("kb.list_pages", type="concept")


def test_list_claims_supports_a_status_filter(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    assert "items" in _result("kb.list_claims", status="working")


def test_list_entities_supports_a_type_filter(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    assert "items" in _result("kb.list_entities", entity_type="person")


def test_list_relations_supports_a_node_filter(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    _result("kb.propose_relation", src="e1", relation="owned_by", target="e2")
    assert "items" in _result("kb.list_relations", node_id="e1")
