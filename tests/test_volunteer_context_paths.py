"""Config, scoring filters, and the MCP push / watch-thread paths.

`volunteer_context` proactively offers approved claims into a live session, so
its filters are the difference between a useful nudge and a stream of noise:
retracted claims must never be offered, an already-offered claim must not
repeat, and the per-session cap and throttle must hold. The MCP push side is
best-effort by design — a dead notification channel must log, never raise into
the caller.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from vouch import hot_memory
from vouch import volunteer_context as vc
from vouch.models import Claim, ClaimStatus, Session
from vouch.storage import KBStore
from vouch.volunteer_context import VolunteerConfig, VolunteerOffer


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


@pytest.fixture(autouse=True)
def _clean_module_state() -> Any:
    """Both modules keep process-global registries; isolate every test."""

    def _reset() -> None:
        with vc._state_lock:
            vc._pending.clear()
            vc._mcp_push.clear()
        for sid in list(vc._watch_threads):
            hot_memory.unregister(sid)
        vc._watch_threads.clear()
        with hot_memory._lock:
            hot_memory._registry.clear()
        hot_memory._SIDEBAR_CACHE.clear()

    _reset()
    yield
    _reset()


def _session(session_id: str = "s1", *, task: str | None = "the review gate") -> Session:
    return Session(id=session_id, agent="claude-code", task=task)


def _claim(store: KBStore, claim_id: str, text: str, **kw: Any) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id], **kw))


# --- load_config ---------------------------------------------------------


def test_config_defaults(store: KBStore) -> None:
    assert vc.load_config(store) == VolunteerConfig()


def test_config_unreadable_file_falls_back(store: KBStore) -> None:
    store.config_path.unlink()
    assert vc.load_config(store) == VolunteerConfig()


def test_config_malformed_yaml_falls_back(store: KBStore) -> None:
    store.config_path.write_text("volunteer: [unclosed\n", encoding="utf-8")
    assert vc.load_config(store) == VolunteerConfig()


def test_config_scalar_document_falls_back(store: KBStore) -> None:
    store.config_path.write_text("just-a-string\n", encoding="utf-8")
    assert vc.load_config(store) == VolunteerConfig()


def test_config_non_mapping_volunteer_block_falls_back(store: KBStore) -> None:
    store.config_path.write_text("volunteer: not-a-mapping\n", encoding="utf-8")
    assert vc.load_config(store) == VolunteerConfig()


def test_config_reads_every_field(store: KBStore) -> None:
    store.config_path.write_text(
        "volunteer:\n"
        "  enabled: false\n"
        "  threshold: 0.75\n"
        "  throttle_seconds: 9\n"
        "  poll_interval_seconds: 3\n"
        "  max_per_session: 2\n",
        encoding="utf-8",
    )
    cfg = vc.load_config(store)
    assert cfg.enabled is False
    assert cfg.threshold == 0.75
    assert cfg.throttle_seconds == 9.0
    assert cfg.poll_interval_seconds == 3.0
    assert cfg.max_per_session == 2


# --- session_query / normalize_relevance ---------------------------------


def test_session_query_joins_task_and_note() -> None:
    sess = Session(id="s1", agent="a", task=" the gate ", note=" and a note ")
    assert vc.session_query(sess) == "the gate and a note"


def test_session_query_is_none_without_task_or_note() -> None:
    assert vc.session_query(Session(id="s1", agent="a")) is None


def test_normalize_relevance_clamps_embedding_scores() -> None:
    assert vc.normalize_relevance(1.4, "embedding", batch_max=1.0) == 1.0
    assert vc.normalize_relevance(-0.2, "embedding", batch_max=1.0) == 0.0
    assert vc.normalize_relevance(0.5, "embedding", batch_max=1.0) == 0.5


def test_normalize_relevance_zero_batch_max_is_zero() -> None:
    # fts5/substring scores are unbounded, so they are scaled by the batch max;
    # a zero max would divide by zero
    assert vc.normalize_relevance(3.0, "fts5", batch_max=0.0) == 0.0


def test_normalize_relevance_scales_by_batch_max() -> None:
    assert vc.normalize_relevance(2.0, "fts5", batch_max=4.0) == 0.5


# --- _retrieve_claim_scores ---------------------------------------------


def _viewer(store: KBStore) -> Any:
    from vouch.scoping import viewer_from

    return viewer_from(config_path=store.config_path, project=None, agent=None)


def test_retrieve_scores_returns_nothing_on_an_empty_kb(store: KBStore) -> None:
    assert vc._retrieve_claim_scores(store, "gate", _viewer(store)) == []


def test_retrieve_scores_finds_a_matching_claim(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    out = vc._retrieve_claim_scores(store, "review gate", _viewer(store))
    assert [row[0] for row in out] == ["c1"]
    assert 0.0 <= out[0][1] <= 1.0


@pytest.mark.parametrize(
    "status",
    [ClaimStatus.SUPERSEDED, ClaimStatus.REDACTED, ClaimStatus.ARCHIVED],
)
def test_retrieve_scores_skips_retracted_claims(
    store: KBStore, status: ClaimStatus
) -> None:
    _claim(store, "c1", "the review gate is load-bearing", status=status)
    assert vc._retrieve_claim_scores(store, "review gate", _viewer(store)) == []


def test_retrieve_scores_skips_a_claim_whose_file_vanished(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # index says the claim exists, disk says otherwise -- must skip, not raise
    from vouch.storage import ArtifactNotFoundError

    _claim(store, "c1", "the review gate is load-bearing")

    def _gone(_claim_id: str) -> Claim:
        raise ArtifactNotFoundError("claim c1")

    monkeypatch.setattr(store, "get_claim", _gone)
    assert vc._retrieve_claim_scores(store, "review gate", _viewer(store)) == []


def test_retrieve_scores_sorts_by_relevance(store: KBStore) -> None:
    _claim(store, "c1", "review gate review gate review gate")
    _claim(store, "c2", "review gate mentioned once")
    out = vc._retrieve_claim_scores(store, "review gate", _viewer(store))
    rels = [row[1] for row in out]
    assert rels == sorted(rels, reverse=True)


# --- _build_why ----------------------------------------------------------


def test_build_why_includes_a_snippet_preview() -> None:
    why = vc._build_why(
        claim_id="c1", query="gate", relevance=0.9, backend="fts5",
        snippet="the «review» gate matters",
    )
    assert "the review gate matters" in why
    assert "fts5 relevance 0.90" in why


def test_build_why_without_a_snippet() -> None:
    why = vc._build_why(
        claim_id="c1", query="gate", relevance=0.9, backend="fts5", snippet="  ",
    )
    assert "matches with fts5 relevance 0.90" in why


# --- evaluate_session ----------------------------------------------------


def test_evaluate_returns_none_when_disabled(store: KBStore) -> None:
    out = vc.evaluate_session(
        store, _session(), config=VolunteerConfig(enabled=False)
    )
    assert out is None


def test_evaluate_returns_none_without_a_query(store: KBStore) -> None:
    assert vc.evaluate_session(store, _session(task=None)) is None


def test_evaluate_returns_none_without_hot_memory(store: KBStore) -> None:
    # no register() call: nothing is tracking this session
    assert vc.evaluate_session(store, _session()) is None


def test_evaluate_returns_none_at_the_per_session_cap(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    hot_memory.mark_volunteered("s1", "c1", pushed_at=0.0)
    out = vc.evaluate_session(
        store, _session(), config=VolunteerConfig(max_per_session=1)
    )
    assert out is None


def test_evaluate_offers_a_matching_claim(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    out = vc.evaluate_session(
        store, _session(), config=VolunteerConfig(threshold=0.0, throttle_seconds=0.0)
    )
    assert out is not None
    assert out.claim_id == "c1"
    assert out.session_id == "s1"


def test_evaluate_skips_an_already_offered_claim(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    hot_memory.mark_volunteered("s1", "c1", pushed_at=0.0)
    out = vc.evaluate_session(
        store,
        _session(),
        config=VolunteerConfig(
            threshold=0.0, throttle_seconds=0.0, max_per_session=99
        ),
    )
    assert out is None


def test_evaluate_respects_the_throttle(store: KBStore) -> None:
    import time

    _claim(store, "c1", "the review gate is load-bearing")
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    hot_memory.mark_volunteered("s1", "other", pushed_at=time.monotonic())
    out = vc.evaluate_session(
        store,
        _session(),
        config=VolunteerConfig(
            threshold=0.0, throttle_seconds=9999.0, max_per_session=99
        ),
    )
    assert out is None


def test_evaluate_returns_none_below_the_threshold(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    out = vc.evaluate_session(
        store,
        _session(),
        config=VolunteerConfig(threshold=1.01, throttle_seconds=0.0),
    )
    assert out is None


def test_evaluate_returns_none_when_nothing_matches(store: KBStore) -> None:
    _claim(store, "c1", "totally unrelated content")
    hot_memory.register(session_id="s1", query="zebras", agent="claude-code")
    out = vc.evaluate_session(
        store,
        Session(id="s1", agent="claude-code", task="zebras"),
        config=VolunteerConfig(threshold=0.0, throttle_seconds=0.0),
    )
    assert out is None


# --- drain_pending / enqueue --------------------------------------------


def _offer(session_id: str = "s1", claim_id: str = "c1") -> VolunteerOffer:
    return VolunteerOffer(
        claim_id=claim_id, relevance=0.9, why="because", session_id=session_id
    )


def test_enqueue_then_drain_clears_the_queue() -> None:
    vc.enqueue_offer(_offer())
    assert [o.claim_id for o in vc.drain_pending("s1")] == ["c1"]
    assert vc.drain_pending("s1") == []


def test_drain_with_no_clear_peeks() -> None:
    vc.enqueue_offer(_offer())
    assert len(vc.drain_pending("s1", clear=False)) == 1
    assert len(vc.drain_pending("s1", clear=False)) == 1


def test_offer_to_dict_shape() -> None:
    assert _offer().to_dict() == {
        "claim_id": "c1",
        "relevance": 0.9,
        "why": "because",
        "session_id": "s1",
    }


# --- MCP push -----------------------------------------------------------


class _FakeSession:
    def __init__(self, *, boom: bool = False) -> None:
        self.sent: list[Any] = []
        self.boom = boom
        self.done = threading.Event()

    async def send_notification(self, note: Any) -> None:
        try:
            if self.boom:
                raise RuntimeError("channel closed")
            self.sent.append(note)
        finally:
            self.done.set()


@pytest.fixture
def running_loop() -> Any:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def test_mcp_push_is_skipped_without_a_registered_channel() -> None:
    vc._maybe_mcp_push(_offer())  # must not raise


def test_mcp_push_sends_a_notification(running_loop: Any) -> None:
    session = _FakeSession()
    vc.register_mcp_push("s1", session, running_loop)  # type: ignore[arg-type]
    vc._maybe_mcp_push(_offer())
    assert session.done.wait(timeout=5)
    assert len(session.sent) == 1
    assert session.sent[0].method == "kb.volunteer_context"
    assert session.sent[0].params is not None


def _wait_for_log(caplog: pytest.LogCaptureFixture, needle: str) -> bool:
    """The push runs on another loop; the log lands after our event fires."""
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if any(needle in r.getMessage() for r in caplog.records):
            return True
        time.sleep(0.02)
    return False


def test_mcp_push_logs_and_swallows_a_send_failure(
    running_loop: Any, caplog: pytest.LogCaptureFixture
) -> None:
    session = _FakeSession(boom=True)
    vc.register_mcp_push("s1", session, running_loop)  # type: ignore[arg-type]
    with caplog.at_level("ERROR"):
        vc._maybe_mcp_push(_offer())
        assert session.done.wait(timeout=5)
        # best-effort: a dead channel must not propagate into the caller
        assert _wait_for_log(caplog, "push failed")


def test_mcp_push_logs_when_the_loop_is_gone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dead = asyncio.new_event_loop()
    dead.close()
    vc.register_mcp_push("s1", _FakeSession(), dead)  # type: ignore[arg-type]
    with caplog.at_level("ERROR"):
        vc._maybe_mcp_push(_offer())
    assert _wait_for_log(caplog, "no event loop")


# --- session lifecycle --------------------------------------------------


def test_on_session_start_is_a_noop_when_disabled(store: KBStore) -> None:
    store.config_path.write_text(
        "volunteer:\n  enabled: false\n", encoding="utf-8"
    )
    vc.on_session_start(store, _session())
    assert hot_memory.get("s1") is None


def test_on_session_start_is_a_noop_without_a_task(store: KBStore) -> None:
    vc.on_session_start(store, _session(task=None))
    assert hot_memory.get("s1") is None


def test_on_session_start_registers_and_watches(store: KBStore) -> None:
    store.config_path.write_text(
        "volunteer:\n  poll_interval_seconds: 30\n", encoding="utf-8"
    )
    _claim(store, "c1", "the review gate is load-bearing")
    store.put_session(_session())
    vc.on_session_start(store, _session())
    assert hot_memory.get("s1") is not None
    assert "s1" in vc._watch_threads
    vc.on_session_end("s1")


def test_on_session_start_logs_an_evaluation_failure(
    store: KBStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("retrieval exploded")

    store.config_path.write_text(
        "volunteer:\n  poll_interval_seconds: 30\n", encoding="utf-8"
    )
    monkeypatch.setattr(vc, "evaluate_session", _boom)
    with caplog.at_level("ERROR"):
        vc.on_session_start(store, _session())
    assert _wait_for_log(caplog, "initial volunteer evaluation failed")
    vc.on_session_end("s1")


def test_on_session_end_clears_everything(store: KBStore) -> None:
    hot_memory.register(session_id="s1", query="q", agent="a")
    vc.enqueue_offer(_offer())
    vc.register_mcp_push("s1", _FakeSession(), asyncio.new_event_loop())  # type: ignore[arg-type]
    vc.on_session_end("s1")
    assert hot_memory.get("s1") is None
    assert vc.drain_pending("s1", clear=False) == []
    with vc._state_lock:
        assert "s1" not in vc._mcp_push


def test_start_watch_does_not_start_a_second_thread(store: KBStore) -> None:
    cfg = VolunteerConfig(poll_interval_seconds=30.0)
    hot_memory.register(session_id="s1", query="q", agent="a")
    store.put_session(_session())
    vc._start_watch(store, "s1", cfg)
    first = vc._watch_threads["s1"]
    vc._start_watch(store, "s1", cfg)
    assert vc._watch_threads["s1"] is first
    vc.on_session_end("s1")


def test_watch_loop_logs_an_evaluation_failure(
    store: KBStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen = threading.Event()

    def _boom(*_a: Any, **_k: Any) -> Any:
        seen.set()
        raise RuntimeError("watch exploded")

    monkeypatch.setattr(vc, "evaluate_session", _boom)
    hot_memory.register(session_id="s1", query="q", agent="a")
    store.put_session(_session())
    with caplog.at_level("ERROR"):
        vc._start_watch(store, "s1", VolunteerConfig(poll_interval_seconds=30.0))
        assert seen.wait(timeout=5)
        hot_memory.unregister("s1")
        vc._watch_threads["s1"].join(timeout=5)
    assert _wait_for_log(caplog, "volunteer watch failed")
    vc._watch_threads.pop("s1", None)


def test_watch_loop_enqueues_an_offer_it_finds(store: KBStore) -> None:
    import time

    _claim(store, "c1", "the review gate is load-bearing")
    store.put_session(_session())
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    vc._start_watch(
        store,
        "s1",
        VolunteerConfig(
            threshold=0.0, throttle_seconds=0.0, poll_interval_seconds=30.0
        ),
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if vc.drain_pending("s1", clear=False):
            break
        time.sleep(0.02)
    assert [o.claim_id for o in vc.drain_pending("s1")] == ["c1"]
    vc.on_session_end("s1")


def test_watch_loop_exits_immediately_without_hot_memory(store: KBStore) -> None:
    vc._start_watch(store, "no-such-session", VolunteerConfig())
    thread = vc._watch_threads["no-such-session"]
    thread.join(timeout=5)
    assert not thread.is_alive()
    vc._watch_threads.pop("no-such-session", None)


# --- evaluate_now -------------------------------------------------------


def test_evaluate_now_enqueues_the_offer(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    store.put_session(_session())
    hot_memory.register(session_id="s1", query="review gate", agent="claude-code")
    store.config_path.write_text(
        "volunteer:\n  threshold: 0.0\n  throttle_seconds: 0\n", encoding="utf-8"
    )
    offer = vc.evaluate_now(store, "s1")
    assert offer is not None
    assert [o.claim_id for o in vc.drain_pending("s1")] == ["c1"]


def test_evaluate_now_returns_none_when_nothing_qualifies(store: KBStore) -> None:
    store.put_session(_session())
    assert vc.evaluate_now(store, "s1") is None
