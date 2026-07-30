"""The claim-lifecycle, source and notify CLI surface.

`claims-clear` and `wipe-dead-refs` are the two destructive commands in the
set, and both were import-covered only — including their confirm prompts and
their dry-run short-circuits. `supersede`/`contradict`/`archive`/`confirm`/
`cite`/`redact` are the thin human mirrors of `lifecycle.*`, and the notify
pair fires outbound webhooks, so the test double matters there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from vouch import notify as notify_mod
from vouch.cli import cli
from vouch.models import Claim, Evidence, Page
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path)
    monkeypatch.chdir(s.root)
    return s


def _run(args: list[str], stdin: str | None = None) -> Result:
    return CliRunner().invoke(cli, args, input=stdin)


def _ok(args: list[str], stdin: str | None = None) -> Result:
    result = _run(args, stdin)
    assert result.exit_code == 0, result.output
    return result


def _clean_error(args: list[str]) -> Result:
    result = _run(args)
    assert result.exit_code != 0, result.output
    assert "Traceback" not in result.output, result.output
    assert "Error:" in result.output, result.output
    return result


def _claim(
    store: KBStore, claim_id: str, text: str, *, auto: bool = False
) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(
        Claim(id=claim_id, text=text, evidence=[src.id], auto_approved=auto)
    )


# --- supersede / contradict / archive / confirm / redact ------------------


def test_supersede_links_old_to_new(store: KBStore) -> None:
    _claim(store, "old", "the first version")
    _claim(store, "new", "the corrected version")
    assert "superseded old -> new" in _ok(["supersede", "old", "new"]).output
    assert store.get_claim("old").superseded_by == "new"


def test_supersede_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _claim(store, "new", "the corrected version")
    _clean_error(["supersede", "ghost", "new"])


def test_contradict_records_both_directions(store: KBStore) -> None:
    _claim(store, "a", "the gate is on")
    _claim(store, "b", "the gate is off")
    out = _ok(["contradict", "a", "b"]).output
    assert "contradiction recorded: a <-> b" in out
    assert "b" in store.get_claim("a").contradicts


def test_contradict_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _claim(store, "a", "the gate is on")
    _clean_error(["contradict", "a", "ghost"])


def test_archive_marks_the_claim(store: KBStore) -> None:
    _claim(store, "c1", "a claim to retire")
    assert "archived c1" in _ok(["archive", "c1"]).output


def test_archive_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["archive", "ghost"])


def test_confirm_bumps_last_confirmed(store: KBStore) -> None:
    _claim(store, "c1", "a claim to re-confirm")
    assert "confirmed c1" in _ok(["confirm", "c1"]).output
    assert store.get_claim("c1").last_confirmed_at is not None


def test_confirm_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["confirm", "ghost"])


def test_redact_masks_the_claim(store: KBStore) -> None:
    _claim(store, "c1", "the api token is tok-live-abcdef123456")
    assert "redacted c1" in _ok(["redact", "c1"]).output


def test_redact_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["redact", "ghost"])


# --- cite -----------------------------------------------------------------


def test_cite_resolves_a_bare_source_id(store: KBStore) -> None:
    src = store.put_source(b"body", title="the memo")
    store.put_claim(Claim(id="c1", text="cited claim", evidence=[src.id]))
    doc = json.loads(_ok(["cite", "c1"]).output)
    assert doc[0]["kind"] == "source"
    assert doc[0]["source_id"] == src.id


def test_cite_resolves_an_evidence_id(store: KBStore) -> None:
    src = store.put_source(b"body")
    store.put_evidence(Evidence(id="ev1", source_id=src.id, locator="p2"))
    store.put_claim(Claim(id="c1", text="cited claim", evidence=["ev1"]))
    doc = json.loads(_ok(["cite", "c1"]).output)
    assert doc[0]["id"] == "ev1"
    assert doc[0]["locator"] == "p2"


def test_cite_unknown_claim_is_a_clean_error(store: KBStore) -> None:
    _clean_error(["cite", "ghost"])


# --- claims-clear ---------------------------------------------------------


def test_claims_clear_with_nothing_matching(store: KBStore) -> None:
    _claim(store, "c1", "a human-approved claim", auto=False)
    assert "no claims match the criteria" in _ok(["claims-clear"]).output


def test_claims_clear_dry_run_makes_no_changes(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    out = _ok(["claims-clear", "--dry-run"]).output
    assert "found 1 claims to clear" in out
    assert "(dry-run mode: no changes made)" in out
    assert store.get_claim("c1").status.value != "archived"


def test_claims_clear_declined_at_the_prompt_cancels(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    out = _ok(["claims-clear"], stdin="n\n").output
    assert "cancelled" in out


def test_claims_clear_confirmed_archives_the_claims(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    out = _ok(["claims-clear", "--confirm"]).output
    assert "cleared 1 claims" in out


def test_claims_clear_accepted_at_the_prompt_archives(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    out = _ok(["claims-clear"], stdin="y\n").output
    assert "cleared 1 claims" in out


def test_claims_clear_truncates_the_preview_at_ten(store: KBStore) -> None:
    for i in range(12):
        _claim(store, f"c{i}", f"auto claim {i}", auto=True)
    out = _ok(["claims-clear", "--dry-run"]).output
    assert "found 12 claims to clear" in out
    assert "... and 2 more" in out


def test_claims_clear_rejects_a_bad_before_date(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    result = _clean_error(["claims-clear", "--before", "not-a-date"])
    assert "invalid date format" in result.output


def test_claims_clear_honours_a_before_cutoff(store: KBStore) -> None:
    _claim(store, "c1", "an auto-saved claim", auto=True)
    # the claim was created now, so a cutoff in the past matches nothing
    out = _ok(["claims-clear", "--before", "2000-01-01", "--dry-run"]).output
    assert "no claims match the criteria" in out


# --- wipe-dead-refs -------------------------------------------------------


def _page_with_dead_ref(store: KBStore) -> None:
    """A page citing a claim whose file is gone.

    `put_page` validates claim refs, so the ref has to go dead *after* the
    page lands -- which is exactly how these arise in practice (the claim was
    redacted or bulk-cleared out from under the page).
    """
    _claim(store, "c1", "a claim that will vanish")
    store.put_page(Page(id="p1", title="a page", claims=["c1"]))
    store._claim_path("c1").unlink()


def test_wipe_dead_refs_with_nothing_to_do(store: KBStore) -> None:
    assert "no dead claim references found" in _ok(["wipe-dead-refs"]).output


def test_wipe_dead_refs_dry_run_keeps_the_ref(store: KBStore) -> None:
    _page_with_dead_ref(store)
    out = _ok(["wipe-dead-refs", "--dry-run"]).output
    assert "found 1 dead claim reference(s)" in out
    assert "page p1: c1" in out
    assert "(dry-run mode: no changes made)" in out
    assert store.get_page("p1").claims == ["c1"]


def test_wipe_dead_refs_declined_at_the_prompt_cancels(store: KBStore) -> None:
    _page_with_dead_ref(store)
    out = _ok(["wipe-dead-refs"], stdin="n\n").output
    assert "cancelled" in out
    assert store.get_page("p1").claims == ["c1"]


def test_wipe_dead_refs_confirmed_strips_the_ref(store: KBStore) -> None:
    _page_with_dead_ref(store)
    out = _ok(["wipe-dead-refs", "--confirm"]).output
    assert "stripped 1 dead reference(s)" in out
    assert store.get_page("p1").claims == []


# --- source ---------------------------------------------------------------


def test_source_add_registers_and_prints_the_id(
    store: KBStore, tmp_path: Path
) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("some evidence", encoding="utf-8")
    src_id = _ok(["source", "add", str(doc)]).output.strip()
    assert store.get_source(src_id).title == "note.txt"


def test_source_add_honours_title_url_and_type(
    store: KBStore, tmp_path: Path
) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("some evidence", encoding="utf-8")
    src_id = _ok([
        "source", "add", str(doc),
        "--title", "a titled note",
        "--url", "https://example.invalid/note",
        "--type", "url",
    ]).output.strip()
    src = store.get_source(src_id)
    assert src.title == "a titled note"
    assert src.type.value == "url"
    # `--url` is accepted but discarded for `source add`: put_source folds url
    # into `locator` only when locator is unset, and the command always passes
    # the resolved path. documenting, not endorsing.
    assert src.locator == str(doc.resolve())


def test_source_list_empty_and_populated(store: KBStore) -> None:
    assert "no sources found" in _ok(["source", "list"]).output
    src = store.put_source(b"body", title="the memo")
    assert src.id in _ok(["source", "list"]).output


def test_source_list_json(store: KBStore) -> None:
    store.put_source(b"body", title="the memo")
    doc = json.loads(_ok(["source", "list", "--json"]).output)
    assert doc[0]["title"] == "the memo"


def test_source_verify_reports_each_source(store: KBStore) -> None:
    store.put_source(b"body", title="the memo")
    out = _ok(["source", "verify"]).output
    assert "stored=" in out
    assert "external=" in out


def test_source_verify_passes_when_the_file_still_matches(
    store: KBStore, tmp_path: Path
) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("some evidence", encoding="utf-8")
    _ok(["source", "add", str(doc)])
    result = _ok(["source", "verify", "--fail-on-issue"])
    assert "external=match" in result.output


def test_source_verify_fail_on_issue_exits_nonzero_on_drift(
    store: KBStore, tmp_path: Path
) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("some evidence", encoding="utf-8")
    _ok(["source", "add", str(doc)])
    # rewriting the file behind the recorded sha256 is the drift case
    doc.write_text("tampered evidence", encoding="utf-8")
    result = _run(["source", "verify", "--fail-on-issue"])
    assert result.exit_code == 1
    assert "!" in result.output


# --- notify ---------------------------------------------------------------


def test_notify_sweep_with_nothing_to_fire(store: KBStore) -> None:
    assert "nothing to fire" in _ok(["notify", "sweep"]).output


def test_notify_sweep_reports_fired_events(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notify_mod, "sweep", lambda _store: ["pending.threshold"])
    out = _ok(["notify", "sweep"]).output
    assert "fired 1 event(s): pending.threshold" in out


def test_notify_test_reports_delivery(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notify_mod, "send_test", lambda url, secret=None: True)
    out = _ok(["notify", "test", "--url", "https://example.invalid/hook"]).output
    assert "delivered" in out


def test_notify_test_exits_nonzero_on_failure(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notify_mod, "send_test", lambda url, secret=None: False)
    result = _run(["notify", "test", "--url", "https://example.invalid/hook"])
    assert result.exit_code == 1
    assert "delivery failed" in result.output


def test_notify_test_resolves_a_secret_from_env(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str | None] = {}

    def _send(url: str, secret: str | None = None) -> bool:
        seen["secret"] = secret
        return True

    monkeypatch.setenv("VOUCH_TEST_HOOK_SECRET", "not-a-real-secret")
    monkeypatch.setattr(notify_mod, "send_test", _send)
    _ok([
        "notify", "test",
        "--url", "https://example.invalid/hook",
        "--secret", "env:VOUCH_TEST_HOOK_SECRET",
    ])
    assert seen["secret"] == "not-a-real-secret"


def test_notify_test_unresolvable_secret_is_a_clean_error(store: KBStore) -> None:
    _clean_error([
        "notify", "test",
        "--url", "https://example.invalid/hook",
        "--secret", "env:VOUCH_NOT_SET_ANYWHERE",
    ])
