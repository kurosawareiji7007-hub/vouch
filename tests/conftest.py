"""Suite-wide fixtures.

The embedder registry in `vouch.embeddings.base` is a module-global dict.
Several test modules register a MockEmbedder as the default adapter
(test_context, test_propose_similarity, test_clear_claims, test_triage) and
never unregister it, so the registration leaks forward into every later test
in the session.

That leak is invisible while numpy is absent -- MockEmbedder can't encode, so
the embedding path stays dormant. Install numpy (the `[embeddings]` extra) and
the leak starts flipping later tests onto the embedding backend: the fts5
backend-label assertions in test_cli, the deindex assertions in test_delete,
and the salience-sidebar cases in test_hot_memory all fail, none of them for a
reason connected to what they test.

`tests/embeddings/conftest.py` already isolates the registry for its own
directory. This lifts the same guarantee to the whole suite so the base tests
pass with or without the extra installed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_embedder_registry() -> Iterator[None]:
    """Snapshot and restore the global embedder registry around every test."""
    from vouch.embeddings import base

    saved = dict(base._REGISTRY)
    try:
        yield
    finally:
        base._REGISTRY.clear()
        base._REGISTRY.update(saved)
