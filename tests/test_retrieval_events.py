"""Retrieval-event log: config, masking, rotation, never-load-bearing wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from vouch.context import build_context_pack
from vouch.retrieval_events import (
    FILENAME,
    ROTATED_FILENAME,
    EventsConfig,
    load_events_config,
    log_event,
    read_events,
)
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def _items() -> list[dict[str, object]]:
    return [{"type": "claim", "id": "c1", "score": 0.5, "summary": "ignored"}]


def test_config_defaults_on(store: KBStore) -> None:
    cfg = load_events_config(store)
    assert cfg == EventsConfig()
    assert cfg.enabled is True


def test_config_disable(store: KBStore) -> None:
    store.config_path.write_text(
        "retrieval:\n  events:\n    enabled: false\n", encoding="utf-8"
    )
    assert load_events_config(store).enabled is False
    assert log_event(
        store, query="q", backend="fts5", limit=5, budget_chars=None, items=[],
    ) is False
    assert not (store.kb_dir / FILENAME).exists()


def test_config_quoted_false_does_not_enable(store: KBStore) -> None:
    """Regression (#558 residual): bool(\"false\") is True, so a quoted
    enabled: \"false\" used to leave retrieval.events on."""
    store.config_path.write_text(
        'retrieval:\n  events:\n    enabled: "false"\n', encoding="utf-8"
    )
    assert load_events_config(store).enabled is False
    assert log_event(
        store, query="q", backend="fts5", limit=5, budget_chars=None, items=[],
    ) is False
    assert not (store.kb_dir / FILENAME).exists()


def test_log_event_writes_masked_record(store: KBStore) -> None:
    tok = "ghp_" + "a" * 36  # same synthetic github-token shape test_secrets uses
    ok = log_event(
        store,
        query=f"why does token={tok} fail",
        backend="fts5", limit=5, budget_chars=800, items=_items(),
    )
    assert ok is True
    events = read_events(store)
    assert len(events) == 1
    ev = events[0]
    assert tok not in ev["query"]
    assert ev["backend"] == "fts5"
    assert ev["budget_chars"] == 800
    # summaries are dropped: only type/id/score are telemetry
    assert ev["items"] == [{"type": "claim", "id": "c1", "score": 0.5}]


def test_log_rotates_at_cap(store: KBStore) -> None:
    cfg = EventsConfig(max_bytes=1)
    for _ in range(2):
        log_event(
            store, query="q", backend="fts5", limit=5, budget_chars=None,
            items=[], config=cfg,
        )
    assert (store.kb_dir / ROTATED_FILENAME).exists()
    # after rotation the live file holds only the newest record
    assert len(read_events(store)) == 1


def test_read_events_tolerates_garbage_and_limits(store: KBStore) -> None:
    path = store.kb_dir / FILENAME
    path.write_text('not json\n{"ts": "t1"}\n{"ts": "t2"}\n', encoding="utf-8")
    assert [e["ts"] for e in read_events(store)] == ["t1", "t2"]
    assert [e["ts"] for e in read_events(store, limit=1)] == ["t2"]


def test_gitignore_backfilled_for_existing_kb(store: KBStore) -> None:
    gi = store.kb_dir / ".gitignore"
    gi.write_text("state.db\n", encoding="utf-8")  # pre-events KB
    log_event(
        store, query="q", backend="fts5", limit=5, budget_chars=None, items=[],
    )
    assert FILENAME in gi.read_text(encoding="utf-8")


def test_init_template_ignores_events_log(store: KBStore) -> None:
    text = (store.kb_dir / ".gitignore").read_text(encoding="utf-8")
    assert "retrieval_events.jsonl*" in text


def test_build_context_pack_logs_event(store: KBStore) -> None:
    pack = build_context_pack(store, query="anything at all", limit=3)
    events = read_events(store)
    assert len(events) == 1
    assert events[0]["query"] == "anything at all"
    assert events[0]["backend"] == pack["backend"]


def test_build_context_pack_survives_broken_log(store: KBStore) -> None:
    # a directory where the log file should be makes every write fail
    (store.kb_dir / FILENAME).mkdir()
    pack = build_context_pack(store, query="still works", limit=3)
    assert pack["query"] == "still works"
