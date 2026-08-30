"""Convert the trained scikit-learn artifacts into one dependency-free bundle.

The build pipeline writes ``*.joblib`` files, because that is the natural output
of scikit-learn and the training scripts genuinely need the estimators back
(permutation importance, refitting, evaluation). Serving needs none of that --
it needs ``predict_proba`` and ``transform`` and nothing else.

So the last build step reads the joblib files and writes:

    artifacts/serving_models.npz    the tree arrays, idf vector, SVD components
    artifacts/serving_models.json   the vocabulary, stop words, task weights

Loading those needs NumPy and the standard library. That is the whole point:
it removes scikit-learn, SciPy and joblib from the deployed image, which is
~200MB of the bundle and the entire source of the unpickling fragility
documented in :mod:`recsys.serving.trees`.

The export is verified, not assumed. ``export_serving_models`` re-scores the
real feature matrix through both paths and refuses to write a bundle that
disagrees with scikit-learn by more than ``tolerance``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .text_encoder import NumpyTfidfSvd
from .trees import NumpyHGB

#: How far the NumPy path may drift from scikit-learn before we refuse to ship.
#: Chosen to be far below the gap between adjacent candidates in a ranked list,
#: so no achievable disagreement can reorder a feed.
TOLERANCE = 1e-6


@dataclass
class ServingModels:
    """Everything the engine needs from the trained models, minus scikit-learn."""

    ranker: NumpyHGB | None = None
    multitask: dict = field(default_factory=dict)
    multitask_weights: dict = field(default_factory=dict)
    multitask_tasks: list = field(default_factory=list)
    text: NumpyTfidfSvd | None = None
    feature_names: list = field(default_factory=list)

    # -- persistence ------------------------------------------------------
    def save(self, npz_path: Path, json_path: Path) -> None:
        arrays: dict = {}
        meta: dict = {
            "feature_names": list(self.feature_names),
            "multitask_weights": dict(self.multitask_weights),
            "multitask_tasks": list(self.multitask_tasks),
            "has_ranker": self.ranker is not None,
            "has_text": self.text is not None,
        }
        if self.ranker is not None:
            arrays.update(self.ranker.arrays("ranker"))
        for task, model in self.multitask.items():
            arrays.update(model.arrays(f"task__{task}"))
        if self.text is not None:
            arrays.update(self.text.arrays())
            meta["text"] = self.text.meta()
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(meta, separators=(",", ":")),
                             encoding="utf-8")

    @classmethod
    def load(cls, npz_path: Path, json_path: Path) -> "ServingModels":
        blob = np.load(npz_path, allow_pickle=False)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        ranker = NumpyHGB.from_arrays(blob, "ranker") if meta["has_ranker"] else None
        multitask = {t: NumpyHGB.from_arrays(blob, f"task__{t}")
                     for t in meta["multitask_tasks"]}
        text = NumpyTfidfSvd.from_arrays(blob, meta["text"]) if meta["has_text"] else None
        return cls(ranker=ranker, multitask=multitask,
                   multitask_weights=meta["multitask_weights"],
                   multitask_tasks=meta["multitask_tasks"], text=text,
                   feature_names=meta["feature_names"])


def load_serving_models(npz_path: Path, json_path: Path) -> ServingModels:
    return ServingModels.load(npz_path, json_path)


def export_serving_models(ranker=None, multitask=None, text_pipeline=None,
                          verify_X: np.ndarray | None = None,
                          verify_texts=None, tolerance: float = TOLERANCE,
                          verbose: bool = True) -> ServingModels:
    """Convert fitted scikit-learn objects, checking equivalence as we go.

    ``verify_X`` and ``verify_texts`` are not optional in practice -- the build
    script always passes them. They are keyword arguments with defaults only so
    the conversion can be unit-tested on a toy model without a feature matrix
    to hand.
    """
    bundle = ServingModels()

    if ranker is not None and getattr(ranker, "model", None) is not None:
        bundle.ranker = NumpyHGB.from_sklearn(ranker.model)
        bundle.feature_names = list(getattr(ranker, "feature_names", []))
        if verify_X is not None and len(verify_X):
            _assert_close("ranker",
                          ranker.model.predict_proba(verify_X)[:, 1],
                          bundle.ranker.predict_proba(verify_X)[:, 1],
                          tolerance, verbose)

    if multitask is not None and getattr(multitask, "models", None):
        bundle.multitask_weights = dict(multitask.weights)
        for task, model in multitask.models.items():
            bundle.multitask[task] = NumpyHGB.from_sklearn(model)
            if verify_X is not None and len(verify_X):
                _assert_close(f"task/{task}",
                              model.predict_proba(verify_X)[:, 1],
                              bundle.multitask[task].predict_proba(verify_X)[:, 1],
                              tolerance, verbose)
        bundle.multitask_tasks = list(bundle.multitask)
        if not bundle.feature_names:
            bundle.feature_names = list(getattr(multitask, "feature_names", []))

    if text_pipeline is not None:
        bundle.text = NumpyTfidfSvd.from_pipeline(text_pipeline)
        if verify_texts:
            texts = list(verify_texts)
            # float32 SVD components mean the two paths cannot agree to
            # float64 precision; compare in the space we actually serve from.
            expected = np.asarray(text_pipeline.transform(texts), dtype=np.float32)
            actual = bundle.text.transform(texts)
            expected = _l2(expected)
            actual = _l2(actual)
            _assert_close("text_encoder", expected.ravel(), actual.ravel(),
                          max(tolerance, 1e-5), verbose)

    return bundle


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def _assert_close(what: str, expected: np.ndarray, actual: np.ndarray,
                  tolerance: float, verbose: bool) -> None:
    delta = float(np.max(np.abs(np.asarray(expected) - np.asarray(actual))))
    if delta > tolerance:
        raise AssertionError(
            f"NumPy export of '{what}' disagrees with scikit-learn by {delta:.3e} "
            f"(tolerance {tolerance:.1e}). Refusing to write a bundle that would "
            "serve different recommendations than the model that was evaluated."
        )
    if verbose:
        print(f"  [export] {what:<18} max |sklearn - numpy| = {delta:.3e}  OK")
