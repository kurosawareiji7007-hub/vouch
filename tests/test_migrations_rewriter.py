"""Atomic writes and the manifest transform verbs.

`atomic_write_text` is the single mutation path every schema migration goes
through, so its temp-file cleanup on failure is the difference between a
crashed migration and a `.vouch/` littered with `.mig-*.tmp` files. The
`split`/`merge` verbs and the markdown-frontmatter branch of `transform_text`
had never executed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vouch.migrations.rewriter import (
    ARTIFACT_KINDS,
    apply_transforms,
    artifact_files,
    atomic_write_text,
    transform_text,
)

# --- atomic_write_text ---------------------------------------------------


def test_atomic_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "claim.yaml"
    atomic_write_text(target, "id: c1\n")
    assert target.read_text(encoding="utf-8") == "id: c1\n"


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "claim.yaml"
    target.write_text("id: old\n", encoding="utf-8")
    atomic_write_text(target, "id: new\n")
    assert target.read_text(encoding="utf-8") == "id: new\n"


def test_atomic_write_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    target = tmp_path / "claim.yaml"
    atomic_write_text(target, "id: c1\n")
    assert [p.name for p in tmp_path.iterdir()] == ["claim.yaml"]


def test_atomic_write_cleans_up_the_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "claim.yaml"

    def _boom(_src: str, _dst: str) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="rename failed"):
        atomic_write_text(target, "id: c1\n")
    # the whole point: a failed migration must not strand .mig-*.tmp files
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_cleans_up_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "claim.yaml"

    def _interrupt(_src: str, _dst: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_text(target, "id: c1\n")
    assert list(tmp_path.iterdir()) == []


# --- transform verbs -----------------------------------------------------


def test_rename_moves_the_value() -> None:
    out = apply_transforms({"old": 1}, [{"rename": {"from": "old", "to": "new"}}])
    assert out == {"new": 1}


def test_rename_is_a_noop_when_the_field_is_absent() -> None:
    assert apply_transforms({"a": 1}, [{"rename": {"from": "x", "to": "y"}}]) == {"a": 1}


def test_default_fills_only_a_missing_field() -> None:
    assert apply_transforms({}, [{"default": {"field": "f", "value": 7}}]) == {"f": 7}
    assert apply_transforms(
        {"f": 1}, [{"default": {"field": "f", "value": 7}}]
    ) == {"f": 1}


def test_drop_removes_the_field_and_tolerates_absence() -> None:
    assert apply_transforms({"a": 1, "b": 2}, [{"drop": {"field": "b"}}]) == {"a": 1}
    assert apply_transforms({"a": 1}, [{"drop": {"field": "zz"}}]) == {"a": 1}


def test_split_fans_a_field_into_parts() -> None:
    out = apply_transforms(
        {"name": "alice example"},
        [{"split": {"field": "name", "into": ["first", "last"]}}],
    )
    assert out == {"first": "alice", "last": "example"}


def test_split_honours_a_custom_separator() -> None:
    out = apply_transforms(
        {"path": "a/b"}, [{"split": {"field": "path", "into": ["x", "y"], "on": "/"}}]
    )
    assert out == {"x": "a", "y": "b"}


def test_split_pads_missing_parts_with_empty_strings() -> None:
    out = apply_transforms(
        {"name": "alice"}, [{"split": {"field": "name", "into": ["first", "last"]}}]
    )
    assert out == {"first": "alice", "last": ""}


def test_split_keeps_the_source_field_when_it_is_a_target() -> None:
    out = apply_transforms(
        {"name": "alice example"},
        [{"split": {"field": "name", "into": ["name", "last"]}}],
    )
    assert out == {"name": "alice", "last": "example"}


def test_split_is_a_noop_when_the_field_is_absent() -> None:
    out = apply_transforms(
        {"a": 1}, [{"split": {"field": "missing", "into": ["x", "y"]}}]
    )
    assert out == {"a": 1}


def test_merge_joins_fields_and_drops_the_sources() -> None:
    out = apply_transforms(
        {"first": "alice", "last": "example"},
        [{"merge": {"fields": ["first", "last"], "into": "name"}}],
    )
    assert out == {"name": "alice example"}


def test_merge_honours_a_custom_joiner() -> None:
    out = apply_transforms(
        {"a": "x", "b": "y"},
        [{"merge": {"fields": ["a", "b"], "into": "c", "with": "-"}}],
    )
    assert out == {"c": "x-y"}


def test_merge_treats_missing_sources_as_empty() -> None:
    out = apply_transforms(
        {"first": "alice"},
        [{"merge": {"fields": ["first", "last"], "into": "name"}}],
    )
    assert out == {"name": "alice "}


def test_merge_keeps_a_source_that_is_also_the_target() -> None:
    out = apply_transforms(
        {"name": "alice", "last": "example"},
        [{"merge": {"fields": ["name", "last"], "into": "name"}}],
    )
    assert out == {"name": "alice example"}


def test_transforms_apply_in_order() -> None:
    out = apply_transforms(
        {"old": "alice example"},
        [
            {"rename": {"from": "old", "to": "name"}},
            {"split": {"field": "name", "into": ["first", "last"]}},
        ],
    )
    assert out == {"first": "alice", "last": "example"}


def test_apply_transforms_does_not_mutate_the_input() -> None:
    original = {"old": 1}
    apply_transforms(original, [{"rename": {"from": "old", "to": "new"}}])
    assert original == {"old": 1}


# --- artifact_files ------------------------------------------------------


def test_artifact_files_lists_yaml_kinds_sorted(tmp_path: Path) -> None:
    (tmp_path / "claims").mkdir()
    for name in ("b.yaml", "a.yaml"):
        (tmp_path / "claims" / name).write_text("id: x\n", encoding="utf-8")
    (tmp_path / "claims" / "ignore.md").write_text("nope", encoding="utf-8")
    assert [p.name for p in artifact_files(tmp_path, "claims")] == ["a.yaml", "b.yaml"]


def test_artifact_files_lists_markdown_for_pages(tmp_path: Path) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "p1.md").write_text("---\nid: p1\n---\nbody", encoding="utf-8")
    (tmp_path / "pages" / "ignore.yaml").write_text("id: x\n", encoding="utf-8")
    assert [p.name for p in artifact_files(tmp_path, "pages")] == ["p1.md"]


def test_artifact_files_on_a_missing_subdir(tmp_path: Path) -> None:
    assert artifact_files(tmp_path, "claims") == []


def test_artifact_kinds_covers_every_durable_dir() -> None:
    assert set(ARTIFACT_KINDS) == {
        "claims", "entities", "relations", "evidence", "sessions", "pages",
    }


# --- transform_text ------------------------------------------------------


def test_transform_text_rewrites_yaml_artifacts() -> None:
    out = transform_text(
        "old: 1\n", "claims", [{"rename": {"from": "old", "to": "new"}}]
    )
    assert "new: 1" in out
    assert "old" not in out


def test_transform_text_leaves_non_mapping_yaml_untouched() -> None:
    text = "- just\n- a\n- list\n"
    assert transform_text(text, "claims", [{"drop": {"field": "x"}}]) == text


def test_transform_text_rewrites_page_frontmatter_only() -> None:
    text = "---\nold: 1\n---\nthe body stays [claim: c1]\n"
    out = transform_text(text, "pages", [{"rename": {"from": "old", "to": "new"}}])
    assert "new: 1" in out
    assert "the body stays [claim: c1]" in out


def test_transform_text_leaves_a_page_without_frontmatter_untouched() -> None:
    text = "no frontmatter here\n"
    assert transform_text(text, "pages", [{"drop": {"field": "x"}}]) == text


def test_transform_text_handles_empty_page_frontmatter() -> None:
    text = "---\n\n---\nbody\n"
    out = transform_text(text, "pages", [{"default": {"field": "f", "value": 1}}])
    assert "f: 1" in out
    assert "body" in out


def test_transform_text_leaves_non_mapping_frontmatter_untouched() -> None:
    text = "---\n- a\n- b\n---\nbody\n"
    assert transform_text(text, "pages", [{"drop": {"field": "x"}}]) == text
