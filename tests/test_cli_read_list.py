"""The read-*/list-* CLI surface plus the small read-only commands.

These are the human mirror of the `kb_read_*` MCP tools. Every one of them
was import-covered only — the decorator ran, the body never did — so a
regression in any of them (wrong yaml shape, a crash on an empty KB, a
traceback instead of a clean `Error:` line) would have shipped silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from vouch.cli import cli
from vouch.models import Claim, Entity, Evidence, Page, Relation
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path)
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
    assert "Error:" in result.output, result.output
    return result


# --- read-* ---------------------------------------------------------------


def test_read_claim_emits_yaml(store: KBStore) -> None:
    src = store.put_source(b"evidence")
    store.put_claim(Claim(id="c1", text="the gate stays", evidence=[src.id]))
    doc = yaml.safe_load(_ok(["read-claim", "c1"]).output)
    assert doc["id"] == "c1"
    assert doc["text"] == "the gate stays"


def test_read_page_emits_yaml(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="review gate"))
    doc = yaml.safe_load(_ok(["read-page", "p1"]).output)
    assert doc["id"] == "p1"
    assert doc["title"] == "review gate"


def test_read_entity_emits_yaml(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="acme-example", type="company"))
    doc = yaml.safe_load(_ok(["read-entity", "e1"]).output)
    assert doc["name"] == "acme-example"
    assert doc["type"] == "company"


def test_read_relation_emits_yaml(store: KBStore) -> None:
    # both endpoints must already exist -- storage validates relation refs
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    store.put_relation(
        Relation(id="r1", source="e1", relation="owned_by", target="e2")
    )
    doc = yaml.safe_load(_ok(["read-relation", "r1"]).output)
    assert doc["source"] == "e1"
    assert doc["relation"] == "owned_by"
    assert doc["target"] == "e2"


def test_read_source_emits_yaml(store: KBStore) -> None:
    src = store.put_source(b"body", title="a note")
    doc = yaml.safe_load(_ok(["read-source", src.id]).output)
    assert doc["id"] == src.id
    assert doc["title"] == "a note"


def test_read_evidence_emits_yaml(store: KBStore) -> None:
    src = store.put_source(b"body")
    store.put_evidence(Evidence(id="ev1", source_id=src.id, locator="p1"))
    doc = yaml.safe_load(_ok(["read-evidence", "ev1"]).output)
    assert doc["id"] == "ev1"
    assert doc["source_id"] == src.id


@pytest.mark.parametrize(
    "command",
    [
        "read-claim",
        "read-page",
        "read-entity",
        "read-relation",
        "read-evidence",
        "read-source",
    ],
)
def test_read_missing_id_is_a_clean_error(store: KBStore, command: str) -> None:
    _clean_error([command, "does-not-exist"])


# --- list-* ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "empty_line"),
    [
        ("list-claims", "no claims found"),
        ("list-pages", "no pages found"),
        ("list-entities", "no entities found"),
        ("list-relations", "no relations found"),
    ],
)
def test_list_on_empty_kb_says_so(
    store: KBStore, command: str, empty_line: str
) -> None:
    assert empty_line in _ok([command]).output


def test_list_claims_shows_id_and_text(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="first claim", evidence=[src.id]))
    store.put_claim(Claim(id="c2", text="second claim", evidence=[src.id]))
    out = _ok(["list-claims"]).output
    assert "c1" in out and "first claim" in out
    assert "c2" in out and "second claim" in out


def test_list_pages_shows_id_and_title(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="the review gate"))
    out = _ok(["list-pages"]).output
    assert "p1" in out and "the review gate" in out


def test_list_entities_shows_name_and_type(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    out = _ok(["list-entities"]).output
    assert "alice-example" in out
    assert "(person)" in out


def test_list_relations_shows_the_triple(store: KBStore) -> None:
    store.put_entity(Entity(id="alice-example", name="alice", type="person"))
    store.put_entity(Entity(id="acme-example", name="acme", type="company"))
    store.put_relation(
        Relation(id="r1", source="alice-example", relation="owned_by",
                 target="acme-example")
    )
    out = _ok(["list-relations"]).output
    assert "alice-example -> owned_by -> acme-example" in out


# --- small read-only commands --------------------------------------------


def test_capabilities_emits_the_method_list(store: KBStore) -> None:
    doc = json.loads(_ok(["capabilities"]).output)
    assert "kb.search" in doc["methods"]


def test_capabilities_outside_a_kb_still_asks_for_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # documents current behaviour, which contradicts the comment above the
    # try/except in cli.capabilities: `_load_store()` exits via SystemExit,
    # which `except Exception` cannot catch, so the no-KB fallback is dead.
    outside = tmp_path / "not-a-kb"
    outside.mkdir()
    monkeypatch.chdir(outside)
    result = _run(["capabilities"])
    assert result.exit_code == 2
    assert "No .vouch/ directory found" in result.output


def test_capabilities_falls_back_when_skills_lookup_raises(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import skills as skills_mod

    def _boom(_store: KBStore) -> bool:
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(skills_mod, "publish_skills_enabled", _boom)
    doc = json.loads(_ok(["capabilities"]).output)
    assert doc["methods"]


def test_index_rebuilds_state_db(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="indexed claim", evidence=[src.id]))
    assert "indexed:" in _ok(["index"]).output


def test_context_emits_a_pack_for_the_task(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(
        Claim(id="c1", text="the review gate is load-bearing", evidence=[src.id])
    )
    doc = json.loads(_ok(["context", "review gate"]).output)
    assert "items" in doc


def test_context_respects_limit(store: KBStore) -> None:
    src = store.put_source(b"e")
    for i in range(5):
        store.put_claim(Claim(id=f"c{i}", text=f"gate claim {i}", evidence=[src.id]))
    doc = json.loads(_ok(["context", "gate", "--limit", "2"]).output)
    assert len(doc["items"]) <= 2


def test_neighbors_emits_json_for_a_known_node(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="a claim with evidence", evidence=[src.id]))
    doc = json.loads(_ok(["neighbors", "c1"]).output)
    assert doc["node_id"] == "c1"
    assert doc["kind"] == "claim"
    assert isinstance(doc["nodes"], list)
    assert isinstance(doc["edges"], list)


def test_neighbors_unknown_node_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["neighbors", "no-such-node"])
