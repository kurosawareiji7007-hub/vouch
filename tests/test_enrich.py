"""Session enrichment: config, prompt, defensive parsing, never-block runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from vouch.enrich import (
    EnrichConfig,
    Enrichment,
    Subject,
    build_enrich_prompt,
    enrich_session,
    load_enrich_config,
    parse_enrichment,
    subject_tags,
    subjects_metadata,
)
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


EXTRACTION = (
    '{"summary": "Ported the dream-style subject extraction into capture.",'
    ' "subjects": [{"name": "Session enrichment", "description": "LLM pass'
    ' decorating session pages", "type": "project"},'
    ' {"name": "Ditto comparison", "description": "competitive review",'
    ' "type": "topic"}]}'
)


def _stub_cmd(tmp_path: Path, output: str, *, exit_code: int = 0) -> str:
    script = tmp_path / "stub-llm.sh"
    script.write_text(
        f"#!/bin/sh\ncat > /dev/null\ncat <<'EOF'\n{output}\nEOF\nexit {exit_code}\n",
        encoding="utf-8",
    )
    return f"sh {script}"


def test_enrich_config_defaults(store: KBStore) -> None:
    cfg = load_enrich_config(store)
    assert cfg == EnrichConfig()
    assert cfg.enabled is True
    assert cfg.llm_cmd is None
    assert cfg.max_subjects == 5


def test_enrich_config_reads_override(store: KBStore) -> None:
    store.config_path.write_text(
        "capture:\n  enrich:\n    llm_cmd: \"cat /dev/null\"\n"
        "    max_subjects: 3\n    timeout_seconds: 10\n",
        encoding="utf-8",
    )
    cfg = load_enrich_config(store)
    assert cfg.llm_cmd == "cat /dev/null"
    assert cfg.max_subjects == 3
    assert cfg.timeout_seconds == 10.0


def test_enrich_config_quoted_false_does_not_enable(store: KBStore) -> None:
    """Regression (#558 residual): bool(\"false\") is True, so a quoted
    enabled: \"false\" used to leave capture.enrich on."""
    store.config_path.write_text(
        'capture:\n  enrich:\n    enabled: "false"\n', encoding="utf-8"
    )
    assert load_enrich_config(store).enabled is False


def test_enrich_config_malformed_yaml_falls_back(store: KBStore) -> None:
    store.config_path.write_text("capture:\n  enrich:\n  - nope\n", encoding="utf-8")
    assert load_enrich_config(store) == EnrichConfig()


def test_parse_enrichment_strips_fences_and_prose() -> None:
    fenced = f"Sure! Here you go:\n```json\n{EXTRACTION}\n```\nHope that helps."
    parsed = parse_enrichment(fenced)
    assert parsed is not None
    assert parsed.summary.startswith("Ported the dream-style")
    assert [s.name for s in parsed.subjects] == ["Session enrichment", "Ditto comparison"]


def test_parse_enrichment_skips_garbage_entries() -> None:
    messy = (
        '{"summary": 7, "subjects": [42, {"description": "no name"},'
        ' {"text": "Rust", "description": "lang", "type": "weird-type"},'
        ' {"name": "   "}]}'
    )
    parsed = parse_enrichment(messy)
    assert parsed is not None
    assert parsed.summary == ""
    assert len(parsed.subjects) == 1
    # "text" alias accepted; unknown type coerced to topic, not dropped.
    assert parsed.subjects[0] == Subject(name="Rust", description="lang", type="topic")


def test_parse_enrichment_caps_subjects() -> None:
    many = ", ".join(
        f'{{"name": "S{i}", "description": "d", "type": "topic"}}' for i in range(9)
    )
    parsed = parse_enrichment(f'{{"summary": "s", "subjects": [{many}]}}', max_subjects=5)
    assert parsed is not None
    assert len(parsed.subjects) == 5


def test_parse_enrichment_unrecoverable_returns_none() -> None:
    assert parse_enrichment("no json here at all") is None
    assert parse_enrichment("{not valid json}") is None
    # parseable but empty extraction means "nothing durable"
    assert parse_enrichment('{"summary": "", "subjects": []}') is None
    assert parse_enrichment('["an", "array"]') is None


def test_build_enrich_prompt_includes_record_and_caps() -> None:
    prompt = build_enrich_prompt(
        [{"summary": "Edit src/vouch/enrich.py"}],
        ["src/vouch/enrich.py"],
        "1 file changed",
        intent="port the dream pipeline",
        max_subjects=5,
        max_input_chars=200,
    )
    assert "STRICT JSON" in prompt
    assert "port the dream pipeline" in prompt
    assert "0 to 5 subjects" in prompt
    # the record section is char-capped, the instructions are not
    record = prompt.split("SESSION RECORD:\n", 1)[1]
    assert len(record) <= 200


def test_subject_tags_slugified_and_deduped() -> None:
    e = Enrichment(
        summary="s",
        subjects=[
            Subject("Session Enrichment", "", "project"),
            Subject("session enrichment", "", "topic"),
            Subject("Audit Log", "", "topic"),
        ],
    )
    assert subject_tags(e) == ["session-enrichment", "audit-log"]
    assert subject_tags(None) == []


def test_subjects_metadata_shape() -> None:
    e = Enrichment(summary="s", subjects=[Subject("A", "d", "topic")])
    assert subjects_metadata(e) == [{"name": "A", "description": "d", "type": "topic"}]
    assert subjects_metadata(None) == []


def test_enrich_session_without_cmd_returns_none(store: KBStore) -> None:
    # no capture.enrich.llm_cmd and no compile.llm_cmd: base install is inert
    assert (
        enrich_session(store, "s1", [], [], "", intent=None) is None
    )


def test_enrich_session_disabled_returns_none(store: KBStore, tmp_path: Path) -> None:
    cfg = EnrichConfig(enabled=False, llm_cmd=_stub_cmd(tmp_path, EXTRACTION))
    assert enrich_session(store, "s1", [], [], "", intent=None, config=cfg) is None


def test_enrich_session_happy_path(store: KBStore, tmp_path: Path) -> None:
    cfg = EnrichConfig(llm_cmd=_stub_cmd(tmp_path, EXTRACTION))
    got = enrich_session(
        store, "s1", [{"summary": "Edit enrich.py"}], ["src/vouch/enrich.py"], "",
        intent="port the dream pipeline", config=cfg,
    )
    assert got is not None
    assert got.summary.startswith("Ported")
    assert len(got.subjects) == 2


def test_enrich_session_falls_back_to_compile_cmd(store: KBStore, tmp_path: Path) -> None:
    cmd = _stub_cmd(tmp_path, EXTRACTION)
    store.config_path.write_text(f'compile:\n  llm_cmd: "{cmd}"\n', encoding="utf-8")
    got = enrich_session(store, "s1", [{"summary": "x"}], [], "", intent=None)
    assert got is not None


def test_enrich_session_failing_cmd_returns_none(store: KBStore, tmp_path: Path) -> None:
    cfg = EnrichConfig(llm_cmd=_stub_cmd(tmp_path, "boom", exit_code=3))
    assert enrich_session(store, "s1", [], [], "", intent=None, config=cfg) is None


def test_enrich_session_unparseable_output_returns_none(
    store: KBStore, tmp_path: Path
) -> None:
    cfg = EnrichConfig(llm_cmd=_stub_cmd(tmp_path, "sorry, no json today"))
    assert enrich_session(store, "s1", [], [], "", intent=None, config=cfg) is None
