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
