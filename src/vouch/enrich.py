"""Session enrichment: one LLM pass that gives a session page semantic handles.

A port of the strongest idea in Ditto's "dream" pipeline — per-memory subject
extraction (`{"summary", "subjects": [{name, description, type}]}` as strict
JSON from a small model) — adapted to vouch's shape: it runs ONCE over the
whole session record instead of per exchange, and its output decorates the
PENDING mechanical rollup page (title, tags, body sections, metadata) rather
than writing anything durable itself. The review gate is untouched; a session
with enrichment is still one proposal a human approves or rejects.

Differences from the source design, on purpose:

* session-holistic, not per-pair — the extractor sees the intent, the tool
  activity, the changed files, and the git stat together, which is exactly
  the cross-turn view per-exchange extraction cannot have;
* subjects become searchable page tags (FTS-indexed) instead of a separate
  graph table — retrieval benefits immediately with no new storage;
* the taxonomy adds ``decision`` — for coding sessions the durable fact is
  usually a decision, not a preference;
* failure never blocks or degrades capture: any error files the plain
  mechanical page, same as before this module existed.

Like ``compile.llm_cmd`` and ``capture.split.llm_cmd``, the model is
deployment config (``capture.enrich.llm_cmd``, falling back to
``compile.llm_cmd``); vouch ships no model dependency and the base install
behaves exactly as if this module were absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

from . import llm_draft
from .config_coerce import coerce_bool
from .llm_draft import LLMDraftError
from .storage import KBStore

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_SUBJECTS = 5
DEFAULT_MAX_INPUT_CHARS = 24000

# The taxonomy the extraction prompt offers. Unknown types are coerced to
# "topic" rather than dropped — a mislabeled subject is still a subject.
SUBJECT_TYPES = frozenset({"person", "project", "preference", "topic", "decision"})


@dataclass(frozen=True)
class EnrichConfig:
    enabled: bool = True
    llm_cmd: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_subjects: int = DEFAULT_MAX_SUBJECTS
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS


@dataclass(frozen=True)
class Subject:
    name: str
    description: str
    type: str


@dataclass(frozen=True)
class Update:
    """A value change the extractor observed: ``attribute`` went old -> new.

    Both values are verbatim strings from the session; downstream matching is
    strict (a value must identify exactly one durable claim) so a hallucinated
    update simply matches nothing and is dropped.
    """

    attribute: str
    old: str
    new: str


@dataclass(frozen=True)
class Enrichment:
    summary: str
    subjects: list[Subject] = field(default_factory=list)
    updates: list[Update] = field(default_factory=list)


def _coerce(value: Any, default: Any, cast: Any) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def load_enrich_config(store: KBStore) -> EnrichConfig:
    """Read ``capture.enrich`` from config.yaml; fall back to defaults."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return EnrichConfig()
    if not isinstance(loaded, dict):
        return EnrichConfig()
    cap = loaded.get("capture")
    raw = cap.get("enrich") if isinstance(cap, dict) else None
    if not isinstance(raw, dict):
        return EnrichConfig()
    llm_cmd = raw.get("llm_cmd")
    return EnrichConfig(
        enabled=coerce_bool(raw.get("enabled", True), True),
        llm_cmd=str(llm_cmd) if llm_cmd else None,
        timeout_seconds=_coerce(
            raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            DEFAULT_TIMEOUT_SECONDS, float),
        max_subjects=_coerce(
            raw.get("max_subjects", DEFAULT_MAX_SUBJECTS),
            DEFAULT_MAX_SUBJECTS, int),
        max_input_chars=_coerce(
            raw.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS),
            DEFAULT_MAX_INPUT_CHARS, int),
    )


def build_enrich_prompt(
    observations: list[dict[str, Any]],
    changed_files: list[str],
    git_stat: str,
    *,
    intent: str | None,
    max_subjects: int,
    max_input_chars: int,
) -> str:
    """Render the whole-session record into one strict-JSON extraction prompt.

    Written so a small local model reliably emits parseable JSON — same
    contract the dream pipeline's EXTRACT stage uses, with the record
    char-capped like every other llm_cmd input in this codebase.
    """
    record: list[str] = []
    if intent:
        record += [f"USER'S OPENING REQUEST: {intent}", ""]
    if changed_files:
        record += ["FILES CHANGED:", *(f"- {f}" for f in sorted(changed_files)[:30]), ""]
    if git_stat:
        record += ["GIT STAT:", git_stat, ""]
    if observations:
        record.append("TOOL ACTIVITY:")
        record += [f"- {o.get('summary', '')}" for o in observations]
        record.append("")
    text = "\n".join(record)
    if len(text) > max_input_chars:
        text = text[:max_input_chars]
    return (
        "You are a memory analyst reviewing one coding-agent session. "
        "Read the session record below and extract what is durably worth "
        "remembering long-term.\n"
        "\n"
        "Durable subjects are people, projects, preferences, decisions, and "
        "recurring topics that will matter in future sessions. Ignore one-off "
        "mechanics (individual commands, transient errors) unless they changed "
        "a durable outcome.\n"
        "\n"
        "Respond with STRICT JSON only — no prose, no code fences, exactly "
        "this shape:\n"
        '{"summary": "<one sentence: what this session accomplished and why '
        'it matters>", "subjects": [{"name": "<short subject name>", '
        '"description": "<one sentence describing the subject>", '
        '"type": "<person|project|preference|topic|decision>"}], '
        '"updates": [{"attribute": "<what changed>", '
        '"old": "<verbatim superseded value>", '
        '"new": "<verbatim current value>"}]}\n'
        "\n"
        "Rules:\n"
        f"- 0 to {max_subjects} subjects; an EMPTY subjects array is a valid "
        "answer — not every session merits subjects.\n"
        "- Keep names short (1-4 words) and descriptions to one sentence.\n"
        "- updates: ONLY changes the record itself states (a value replaced "
        "by a newer value). Copy both values verbatim from the record; an "
        "EMPTY updates array is the normal answer.\n"
        "\n"
        "SESSION RECORD:\n"
        f"{text}"
    )


def _first_json_object(text: str) -> str | None:
    """Substring from the first ``{`` to the last ``}``.

    Strips surrounding prose and markdown fences without regex fragility —
    small local models routinely wrap their JSON, and enrichment must
    tolerate that rather than discard the whole extraction.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_enrichment(raw: str, *, max_subjects: int = DEFAULT_MAX_SUBJECTS) -> Enrichment | None:
    """Defensively parse EXTRACT output.

    Returns None only when no JSON object can be recovered at all; malformed
    subject entries are skipped, never fatal. A parseable object with an
    empty summary and no subjects is treated as "nothing durable" (None) so
    callers file the plain page.
    """
    snippet = _first_json_object(raw)
    if snippet is None:
        return None
    try:
        value = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    raw_summary = value.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) else ""
    subjects: list[Subject] = []
    raw_subjects = value.get("subjects")
    if isinstance(raw_subjects, list):
        for entry in raw_subjects:
            if not isinstance(entry, dict):
                continue
            name_val = entry.get("name") or entry.get("text")
            name = name_val.strip() if isinstance(name_val, str) else ""
            if not name:
                continue
            desc_val = entry.get("description")
            description = desc_val.strip() if isinstance(desc_val, str) else ""
            type_val = entry.get("type")
            stype = type_val.strip().lower() if isinstance(type_val, str) else ""
            if stype not in SUBJECT_TYPES:
                stype = "topic"
            subjects.append(Subject(name=name, description=description, type=stype))
            if len(subjects) == max_subjects:
                break
    updates: list[Update] = []
    raw_updates = value.get("updates")
    if isinstance(raw_updates, list):
        for entry in raw_updates:
            if not isinstance(entry, dict):
                continue
            attr = entry.get("attribute")
            old = entry.get("old")
            new = entry.get("new")
            if not (
                isinstance(attr, str) and isinstance(old, str) and isinstance(new, str)
            ):
                continue
            attr, old, new = attr.strip(), old.strip(), new.strip()
            if not attr or not old or not new or old.lower() == new.lower():
                continue
            updates.append(Update(attribute=attr, old=old, new=new))
            if len(updates) == max_subjects:
                break
    if not summary and not subjects and not updates:
        return None
    return Enrichment(summary=summary, subjects=subjects, updates=updates)


def enrich_session(
    store: KBStore,
    session_id: str,
    observations: list[dict[str, Any]],
    changed_files: list[str],
    git_stat: str,
    *,
    intent: str | None,
    config: EnrichConfig | None = None,
) -> Enrichment | None:
    """Run the extraction over one session record. Never raises.

    Returns None when enrichment is disabled, unconfigured, or fails for any
    reason — the caller files the plain mechanical page either way.
    """
    from . import compile as compile_mod  # deferred: compile imports proposals

    cfg = config or load_enrich_config(store)
    if not cfg.enabled:
        return None
    cmd = cfg.llm_cmd or compile_mod.load_config(store).llm_cmd
    if not cmd:
        return None
    prompt = build_enrich_prompt(
        observations, changed_files, git_stat,
        intent=intent, max_subjects=cfg.max_subjects,
        max_input_chars=cfg.max_input_chars,
    )
    try:
        raw = llm_draft.run_llm(
            cmd, prompt, timeout_seconds=cfg.timeout_seconds,
            label="capture.enrich.llm_cmd",
        )
    except LLMDraftError as e:
        logger.warning("enrich: llm call failed for %s (%s)", session_id, e)
        return None
    enrichment = parse_enrichment(raw, max_subjects=cfg.max_subjects)
    if enrichment is None:
        logger.warning("enrich: unparseable output for %s, filing plain page", session_id)
    return enrichment


def subject_tags(enrichment: Enrichment | None) -> list[str]:
    """Slugified subject names, deduplicated, for the page's tag list.

    Tags are FTS-indexed, so this is the retrieval payoff: a session that
    touched "audit log race" is findable by subject even when the raw
    observation text never used those words together.
    """
    if enrichment is None:
        return []
    from .proposals import _slugify  # deferred: proposals imports are heavy

    seen: set[str] = set()
    tags: list[str] = []
    for s in enrichment.subjects:
        slug = _slugify(s.name)
        if slug and slug not in seen:
            seen.add(slug)
            tags.append(slug)
    return tags


def subjects_metadata(enrichment: Enrichment | None) -> list[dict[str, str]]:
    """Subjects as plain dicts for page proposal metadata."""
    if enrichment is None:
        return []
    return [
        {"name": s.name, "description": s.description, "type": s.type}
        for s in enrichment.subjects
    ]
