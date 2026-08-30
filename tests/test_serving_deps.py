"""The serving path must run on NumPy and the standard library alone.

This is an executable version of a claim that is otherwise easy to make and
easy to be wrong about. Twice now the deployed container has failed for reasons
that never showed up locally -- once because pandas pushed the bundle past the
size limit, once because a scikit-learn pickle needed a module that the
platform had pruned. Both were invisible to a normal test run, because the
development machine has every library installed.

So instead of trusting the import graph, this test forbids the heavy libraries
outright and then does real work: load the artifacts, run a recommendation,
run a search. If any of them creeps back into the serving path, this fails
here rather than in production.

The block must raise ``ModuleNotFoundError`` specifically, not ``ImportError``.
Several libraries catch ``ModuleNotFoundError`` to probe for optional
dependencies, and a guard that raised the parent class would be caught by that
handling and silently pass -- which is a bug in the guard, not evidence that
the code is clean.
"""

from __future__ import annotations

import builtins
import sys

import pytest

#: Everything the build needs and serving must not.
FORBIDDEN = ("pandas", "sklearn", "scipy", "joblib", "pyarrow")


@pytest.fixture()
def no_heavy_imports(monkeypatch):
    """Make the forbidden libraries un-importable for the duration of a test."""
    for name in list(sys.modules):
        if name.split(".")[0] in FORBIDDEN:
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in FORBIDDEN:
            raise ModuleNotFoundError(
                f"No module named {name!r} -- the serving path must not import it"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


def test_serving_runs_without_heavy_dependencies(no_heavy_imports):
    from recsys.artifacts import load_artifacts
    from recsys.engine import RecommendationEngine

    artifacts = load_artifacts()
    engine = RecommendationEngine(artifacts)

    seed = [str(v) for v in artifacts.catalog["video_id"].values[:3]]
    result = engine.recommend(history=seed, n=12)
    assert len(result.items) == 12
    # Explanations are part of the deliverable, so exercise them here too --
    # they are what pulls the multi-task heads through the NumPy path.
    assert all(item.explanation for item in result.items)


def test_search_runs_without_heavy_dependencies(no_heavy_imports):
    """Search is the one path that needs the query encoder at request time."""
    from recsys.artifacts import load_artifacts
    from recsys.engine import RecommendationEngine

    engine = RecommendationEngine(load_artifacts())
    result = engine.search("machine learning tutorial", n=10)
    assert len(result.items) > 0


def test_the_guard_itself_blocks_imports(no_heavy_imports):
    """A guard that does not actually block would make the tests above vacuous."""
    for name in FORBIDDEN:
        with pytest.raises(ModuleNotFoundError):
            __import__(name)


def test_vercel_mount_prefix_is_stripped():
    """The deployed routing depends on this, and nothing else would catch it.

    ``vercel.json`` rewrites every request to ``/api/index/<path>`` because a
    rewrite to a bare ``/api/index`` loses the path entirely and 404s the whole
    site. That makes the strip below load-bearing: if it and the rewrite ever
    disagree, every route breaks at once and only in production.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
    try:
        import index as vercel_entry
    finally:
        sys.path.pop(0)

    strip = vercel_entry.strip_mount_prefix
    assert strip("/api/index") == "/"
    assert strip("/api/index/") == "/"
    assert strip("/api/index/api/health") == "/api/health"
    assert strip("/api/index/static/app.js") == "/static/app.js"
    # Idempotent for un-rewritten paths, so Docker and `uvicorn` are unaffected.
    assert strip("/api/health") == "/api/health"
    assert strip("/") == "/"
    # A route that merely starts with the same letters must not be mangled.
    assert strip("/api/indexer") == "/api/indexer"


def test_out_of_vocabulary_search_reports_no_match():
    """F10: a query matching no vocabulary term must not look like a result set.

    A TF-IDF query with no in-vocabulary token encodes to exactly zero, so every
    similarity ties at 0.0 and the top-k becomes tie-break order. The output has
    the shape of a ranked list and carries no information. The failure worth
    guarding is not the empty retrieval -- that is inherent to lexical search --
    but presenting it as a successful search.
    """
    import numpy as np

    from recsys.artifacts import load_artifacts
    from recsys.engine import RecommendationEngine

    artifacts = load_artifacts()
    engine = RecommendationEngine(artifacts)
    encoder = artifacts.text_index.encoder

    # Establish the premise rather than assuming it: this query really does
    # encode to zero, and a known-good one does not.
    assert not np.any(encoder.transform(["machine learning"])[0])
    assert np.any(encoder.transform(["brown butter"])[0])

    missed = engine.search("machine learning", n=6)
    assert missed.diagnostics["query_matched"] is False
    # It still returns something useful -- an empty page would be a worse
    # answer than "here is what is trending", as long as it says which it is.
    assert len(missed.items) > 0
    assert not any("search" in item.explanation.lower() for item in missed.items)

    matched = engine.search("brown butter", n=6)
    assert matched.diagnostics["query_matched"] is True
    assert any("search" in item.explanation.lower() for item in matched.items)


def test_partial_vocabulary_match_still_counts_as_a_match():
    """Only a fully out-of-vocabulary query is a miss; one real term is enough."""
    from recsys.artifacts import load_artifacts
    from recsys.engine import RecommendationEngine

    engine = RecommendationEngine(load_artifacts())
    result = engine.search("quantum blockchain nft", n=6)
    assert result.diagnostics["query_matched"] is True
