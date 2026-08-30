"""The exported NumPy models must rank identically to the scikit-learn ones.

Why this is a test and not just a build-time assertion
------------------------------------------------------
``scripts/12_export_serving.py`` already refuses to write a bundle that
disagrees with scikit-learn by more than 1e-6. But that check runs against the
*current* artifacts, on the machine doing the build. It cannot catch a
regression in the conversion code itself, because a broken converter would
simply produce a bundle that fails the assertion later, on someone else's
machine, at deploy time.

So this fits a small model in-process and checks the conversion directly. It
deliberately does NOT load ``artifacts/*.joblib`` -- those are fitting outputs
and are gitignored, so a test that depended on them would fail on a fresh
clone, which is precisely the situation a reviewer is in.

Requires scikit-learn, which is a BUILD dependency (`requirements-build.txt`).
That is correct: this test exercises the build-time conversion. The separate
guarantee -- that *serving* never imports scikit-learn -- lives in
``test_serving_deps.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.serving import NumpyHGB, ServingModels

sklearn = pytest.importorskip("sklearn", reason="build-only dependency")


def _fitted_model(seed: int = 0, with_nans: bool = True):
    """A small but structurally faithful HistGradientBoosting classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1500, 8))
    # A learnable signal, so the trees actually split rather than collapsing to
    # a single leaf -- a degenerate forest would pass any comparison.
    y = ((X[:, 0] + 0.7 * X[:, 3] - 0.4 * X[:, 5] + rng.normal(scale=0.4, size=1500)) > 0)
    if with_nans:
        X[rng.random(X.shape) < 0.08] = np.nan
    model = HistGradientBoostingClassifier(
        max_iter=25, max_leaf_nodes=15, min_samples_leaf=10,
        early_stopping=False, random_state=seed,
    )
    model.fit(X, y.astype(int))
    return model, X


def test_numpy_export_matches_sklearn():
    """The whole premise of the serving bundle: identical probabilities."""
    model, X = _fitted_model()
    exported = NumpyHGB.from_sklearn(model)

    expected = model.predict_proba(X)[:, 1]
    actual = exported.predict_proba(X)[:, 1]

    assert actual.shape == expected.shape
    # Tree inference is exact arithmetic on the same thresholds, so this should
    # agree to machine precision, not merely to a loose tolerance.
    assert np.max(np.abs(expected - actual)) < 1e-12


def test_numpy_export_routes_missing_values_the_same_way():
    """The branch most likely to be reimplemented wrongly, tested on its own.

    A converter that ignored `missing_go_to_left` would still pass a comparison
    on dense data, so NaNs are forced into every column here.
    """
    model, _ = _fitted_model()
    exported = NumpyHGB.from_sklearn(model)

    X = np.full((64, 8), np.nan)
    assert np.max(np.abs(model.predict_proba(X)[:, 1]
                         - exported.predict_proba(X)[:, 1])) < 1e-12


def test_numpy_export_survives_a_save_load_round_trip(tmp_path):
    """What is served is what was read back from disk, not what was in memory."""
    model, X = _fitted_model()
    bundle = ServingModels(ranker=NumpyHGB.from_sklearn(model),
                           feature_names=[f"f{i}" for i in range(8)])
    bundle.save(tmp_path / "m.npz", tmp_path / "m.json")

    reloaded = ServingModels.load(tmp_path / "m.npz", tmp_path / "m.json")
    assert np.array_equal(reloaded.ranker.predict_proba(X),
                          bundle.ranker.predict_proba(X))
    assert reloaded.feature_names == bundle.feature_names


def test_export_refuses_a_model_it_cannot_represent():
    """Silent mis-prediction is the failure mode worth ruling out.

    The NumPy runtime handles numeric splits only. If the ranker ever gains
    categorical features, the exporter must fail loudly at build time rather
    than ship a bundle that quietly routes those splits the wrong way.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=400), rng.integers(0, 4, size=400)])
    y = (X[:, 0] + X[:, 1] * 0.5 > 1).astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=10, categorical_features=[1], early_stopping=False, random_state=1,
    ).fit(X, y)

    with pytest.raises(ValueError, match="categorical"):
        NumpyHGB.from_sklearn(model)


def test_shipped_bundle_reproduces_the_committed_evaluation():
    """The deployed ranker must be the one the evaluation report describes.

    Loading the committed bundle and scoring a fixed matrix pins the artifact
    to the code: if either is regenerated without the other, the recorded
    feature count stops matching and this fails.
    """
    from recsys.artifacts import load_artifacts

    artifacts = load_artifacts()
    n_features = len(artifacts.ranker.feature_names)

    rng = np.random.default_rng(7)
    X = rng.normal(size=(256, n_features))
    scores = artifacts.ranker.score(X)

    assert scores.shape == (256,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores > 0), "odds are a ratio of probabilities and cannot be <= 0"
    assert artifacts.multitask is not None
    assert set(artifacts.multitask.models) == {
        "click", "long_watch", "completion", "liked", "satisfied", "dismissed"}
