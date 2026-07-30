"""Propose-time similarity warnings.

`find_similar_on_propose` is reached from `proposals.propose_claim` (the lazy
import at proposals.py:258) and is advisory only — every failure mode must
degrade to an empty warning list rather than block the proposal. These tests
pin that contract plus the two warning codes and their per-code cap.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vouch.embeddings import register
from vouch.embeddings.base import DEFAULT_MODEL_NAME, Embedder
from vouch.embeddings.similarity import (
    DEFAULT_THRESHOLD,
    _similar_pending,
    find_similar_on_propose,
    similarity_threshold,
)
from vouch.models import Claim, Proposal, ProposalKind, ProposalStatus
from vouch.storage import KBStore

_ZERO_MARK = "zero-vector-please"
_BOOM_MARK = "raise-on-encode-please"


class _HashEmbedder(Embedder):
    """Deterministic unit-norm embedder — identical text gives cosine 1.0.

    Mirrors tests/embeddings/test_dedup.py: bytes scaled to 0..1 keep the
    float32 dot products well away from overflow.
    """

    name = "mock"
    version = "1"
    dim = 8

    def encode(self, text: str) -> np.ndarray:
        if _BOOM_MARK in text:
            raise RuntimeError("embedder refused this text")
        if _ZERO_MARK in text:
            return np.zeros(self.dim, dtype=np.float32)
        h = hashlib.sha256(text.encode()).digest()
        out = np.array([h[i] / 255.0 for i in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out


@pytest.fixture(autouse=True)
def _register_default() -> None:
    register(DEFAULT_MODEL_NAME, _HashEmbedder)


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def _approved(store: KBStore, claim_id: str, text: str) -> Claim:
    src = store.put_source(b"evidence")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id]))


def _pending(
    store: KBStore,
    proposal_id: str,
    *,
    text: str | None,
    kind: ProposalKind = ProposalKind.CLAIM,
) -> Proposal:
    payload: dict[str, Any] = {} if text is None else {"text": text}
    return store.put_proposal(
        Proposal(
            id=proposal_id,
            kind=kind,
            proposed_by="agent",
            payload=payload,
            status=ProposalStatus.PENDING,
        )
    )


def _codes(warnings: list[dict[str, Any]], code: str) -> list[dict[str, Any]]:
    return [w for w in warnings if w["code"] == code]


# --- similarity_threshold -------------------------------------------------


def test_threshold_falls_back_to_dedup_default(store: KBStore) -> None:
    assert similarity_threshold(store) == DEFAULT_THRESHOLD


def test_threshold_reads_review_config(store: KBStore) -> None:
    cfg = yaml.safe_load(store.config_path.read_text(encoding="utf-8")) or {}
    cfg["review"] = {"similarity_threshold": 0.5}
    store.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert similarity_threshold(store) == 0.5


def test_threshold_ignores_non_mapping_review_block(store: KBStore) -> None:
    cfg = yaml.safe_load(store.config_path.read_text(encoding="utf-8")) or {}
    cfg["review"] = "not-a-mapping"
    store.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert similarity_threshold(store) == DEFAULT_THRESHOLD


def test_threshold_ignores_null_similarity_threshold(store: KBStore) -> None:
    cfg = yaml.safe_load(store.config_path.read_text(encoding="utf-8")) or {}
    cfg["review"] = {"similarity_threshold": None}
    store.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert similarity_threshold(store) == DEFAULT_THRESHOLD


def test_threshold_ignores_scalar_config_document(store: KBStore) -> None:
    store.config_path.write_text("just-a-string\n", encoding="utf-8")
    assert similarity_threshold(store) == DEFAULT_THRESHOLD


def test_threshold_survives_unreadable_config(store: KBStore) -> None:
    # unparseable yaml must not propagate out of an advisory helper
    store.config_path.write_text("review: [unclosed\n", encoding="utf-8")
    assert similarity_threshold(store) == DEFAULT_THRESHOLD


# --- degradation paths ----------------------------------------------------


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_blank_text_yields_no_warnings(store: KBStore, text: str) -> None:
    _approved(store, "c1", "some approved claim")
    assert find_similar_on_propose(store, text) == []


def test_missing_embedder_degrades_to_empty(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vouch.embeddings as emb

    # seed first — storage indexes through the same get_embedder()
    _approved(store, "c1", "duplicate me")

    def _no_embedder() -> Embedder:
        raise RuntimeError("no embedding backend configured")

    monkeypatch.setattr(emb, "get_embedder", _no_embedder)
    assert find_similar_on_propose(store, "duplicate me") == []


def test_pending_scan_degrades_when_numpy_missing(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_similar_pending` imports numpy lazily; a None entry in sys.modules makes
    # that import raise ImportError, which must yield no pending warnings. called
    # directly because the approved leg reaches index_db, which needs real numpy.
    _pending(store, "p1", text="duplicate me")
    query_vec = _HashEmbedder().encode("duplicate me")
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert (
        _similar_pending(
            store,
            query_vec=query_vec,
            embedder=_HashEmbedder(),
            threshold=DEFAULT_THRESHOLD,
        )
        == []
    )


# --- similar_approved -----------------------------------------------------


def test_identical_approved_claim_warns(store: KBStore) -> None:
    _approved(store, "c1", "vouch gates every write behind review")
    warnings = _codes(
        find_similar_on_propose(store, "vouch gates every write behind review"),
        "similar_approved",
    )
    assert [w["artifact_id"] for w in warnings] == ["c1"]
    hit = warnings[0]
    assert hit["artifact_kind"] == "claim"
    assert hit["cosine"] == pytest.approx(1.0, abs=1e-4)
    assert hit["snippet"] == "vouch gates every write behind review"


def test_disjoint_approved_claim_does_not_warn(store: KBStore) -> None:
    _approved(store, "c1", "apples")
    assert _codes(find_similar_on_propose(store, "zebras"), "similar_approved") == []


def test_exclude_claim_id_filters_itself_out(store: KBStore) -> None:
    text = "self-edit should not warn about itself"
    _approved(store, "c1", text)
    assert (
        _codes(
            find_similar_on_propose(store, text, exclude_claim_id="c1"),
            "similar_approved",
        )
        == []
    )


def test_approved_warnings_capped_at_three(store: KBStore) -> None:
    text = "the same claim filed five times"
    for i in range(5):
        _approved(store, f"c{i}", text)
    warnings = _codes(find_similar_on_propose(store, text), "similar_approved")
    assert len(warnings) == 3


def test_explicit_threshold_overrides_config(store: KBStore) -> None:
    _approved(store, "c1", "apples")
    # a floor of 0.0 admits even an unrelated claim
    warnings = _codes(
        find_similar_on_propose(store, "zebras", threshold=0.0), "similar_approved"
    )
    assert [w["artifact_id"] for w in warnings] == ["c1"]


def test_snippet_of_long_claim_is_truncated(store: KBStore) -> None:
    # no trailing space: find_similar_on_propose strips the query, and this
    # embedder hashes the exact string, so a mismatch would drop the cosine
    long_text = ("alpha " * 40).strip()
    _approved(store, "c1", long_text)
    warnings = _codes(find_similar_on_propose(store, long_text), "similar_approved")
    snippet = warnings[0]["snippet"]
    assert len(snippet) == 120
    assert snippet.endswith("…")
    # newlines and runs of whitespace collapse to single spaces
    assert "  " not in snippet


def test_snippet_falls_back_to_id_when_claim_file_gone(store: KBStore) -> None:
    text = "claim indexed then deleted from disk"
    claim = _approved(store, "c1", text)
    # leave the embedding index intact but remove the artifact the snippet reads
    store._claim_path(claim.id).unlink()
    warnings = _codes(find_similar_on_propose(store, text), "similar_approved")
    assert [w["snippet"] for w in warnings] == ["c1"]


# --- similar_pending ------------------------------------------------------


def test_identical_pending_proposal_warns(store: KBStore) -> None:
    _pending(store, "p1", text="a pending duplicate")
    warnings = _codes(
        find_similar_on_propose(store, "a pending duplicate"), "similar_pending"
    )
    assert [w["artifact_id"] for w in warnings] == ["p1"]
    hit = warnings[0]
    assert hit["artifact_kind"] == "proposal"
    assert hit["cosine"] == pytest.approx(1.0, abs=1e-4)
    assert hit["snippet"] == "a pending duplicate"


def test_disjoint_pending_proposal_does_not_warn(store: KBStore) -> None:
    _pending(store, "p1", text="apples")
    assert _codes(find_similar_on_propose(store, "zebras"), "similar_pending") == []


def test_non_claim_pending_proposals_are_skipped(store: KBStore) -> None:
    _pending(store, "p1", text="a pending duplicate", kind=ProposalKind.PAGE)
    assert (
        _codes(find_similar_on_propose(store, "a pending duplicate"), "similar_pending")
        == []
    )


def test_pending_proposal_without_text_is_skipped(store: KBStore) -> None:
    _pending(store, "p1", text=None)
    _pending(store, "p2", text="   ")
    assert _codes(find_similar_on_propose(store, "anything"), "similar_pending") == []


def test_pending_proposal_that_fails_to_encode_is_skipped(store: KBStore) -> None:
    _pending(store, "p1", text=f"a pending duplicate {_BOOM_MARK}")
    _pending(store, "p2", text="a pending duplicate")
    warnings = _codes(
        find_similar_on_propose(store, "a pending duplicate"), "similar_pending"
    )
    assert [w["artifact_id"] for w in warnings] == ["p2"]


def test_pending_proposal_with_zero_vector_is_skipped(store: KBStore) -> None:
    _pending(store, "p1", text=_ZERO_MARK)
    assert _codes(find_similar_on_propose(store, "anything", threshold=0.0),
                  "similar_pending") == []


def test_zero_norm_query_still_scans_pending(store: KBStore) -> None:
    # a degenerate query vector must not raise; cosine collapses to 0.0
    _pending(store, "p1", text="a pending claim")
    warnings = _codes(
        find_similar_on_propose(store, _ZERO_MARK, threshold=0.0), "similar_pending"
    )
    assert [w["artifact_id"] for w in warnings] == ["p1"]
    assert warnings[0]["cosine"] == pytest.approx(0.0)


def test_pending_warnings_capped_at_three_and_ranked(store: KBStore) -> None:
    text = "the same pending claim filed five times"
    for i in range(5):
        _pending(store, f"p{i}", text=text)
    warnings = _codes(find_similar_on_propose(store, text), "similar_pending")
    assert len(warnings) == 3
    cosines = [w["cosine"] for w in warnings]
    assert cosines == sorted(cosines, reverse=True)


def test_approved_and_pending_warnings_both_surface(store: KBStore) -> None:
    text = "one text, two warning codes"
    _approved(store, "c1", text)
    _pending(store, "p1", text=text)
    warnings = find_similar_on_propose(store, text)
    assert {w["code"] for w in warnings} == {"similar_approved", "similar_pending"}
