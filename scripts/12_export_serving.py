"""Stage 6: convert the trained scikit-learn models into the serving bundle.

    python scripts/12_export_serving.py

Reads the ``*.joblib`` artifacts and writes ``artifacts/serving_models.npz``
plus ``artifacts/serving_models.json`` -- the same trees and the same encoder,
stored as plain arrays.

Why the build has this stage at all
-----------------------------------
The deployed container should contain nothing it does not use. After this step
the serving path imports NumPy and the standard library and nothing else:

    before   pandas + scikit-learn + SciPy + joblib   ~339MB   (over the limit)
    after    NumPy only                                ~70MB   (comfortable)

Size was the visible problem. The invisible one was worse: a fitted
``HistGradientBoostingClassifier`` pickle references a Cython class whose
``__module__`` is the bare name ``_loss``, so unpickling depends on an import
that only resolves by accident. It failed in production with
``ModuleNotFoundError: No module named '_loss'`` and no amount of version
pinning fixed it, because the version was never the cause. Exporting the trees
removes the reference. See ``src/recsys/serving/trees.py``.

The conversion is checked, not trusted. Every model is re-scored through both
paths on the real feature matrix and the real catalog text, and the script
fails rather than writing a bundle that would rank differently from the model
the evaluation report describes.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import joblib
import numpy as np

from recsys.config import Paths, load_config
from recsys.serving import export_serving_models


def _verification_features(n_features: int, seed: int = 0) -> np.ndarray:
    """A feature matrix that exercises the branches real traffic reaches.

    Random normals cover the numeric splits; the injected NaNs are the point,
    because missing-value routing is the one part of tree inference that is
    easy to reimplement subtly wrong and impossible to notice from a spot check.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(4000, n_features))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    load_config()
    started = time.time()

    print("\n>>> exporting serving models (NumPy only)\n")

    ranker = joblib.load(Paths.ranker) if Paths.ranker.exists() else None
    multitask = (joblib.load(Paths.multitask_ranker)
                 if Paths.multitask_ranker.exists() else None)
    text_pipeline = (joblib.load(Paths.text_encoder)
                     if Paths.text_encoder.exists() else None)
    if ranker is None:
        raise SystemExit("no ranker found -- run scripts/04_train_ranker.py first")

    n_features = len(ranker.feature_names)
    catalog = json.loads(Paths.serving_json.read_text(encoding="utf-8"))
    # Verify the encoder on the real corpus, not on sample sentences: the
    # tokenizer is hand-ported, and the documents it must agree on are these.
    texts = [f"{title} {description}" for title, description
             in zip(catalog["title"], catalog["description"])]

    bundle = export_serving_models(
        ranker=ranker, multitask=multitask, text_pipeline=text_pipeline,
        verify_X=_verification_features(n_features),
        verify_texts=texts, tolerance=args.tolerance,
    )
    bundle.save(Paths.serving_models_npz, Paths.serving_models_json)

    size = (Paths.serving_models_npz.stat().st_size
            + Paths.serving_models_json.stat().st_size) / 1e6
    trees = 0 if bundle.ranker is None else len(bundle.ranker.tree_offsets)
    heads = len(bundle.multitask)
    print(f"\n  wrote {Paths.serving_models_npz.name} + "
          f"{Paths.serving_models_json.name}  ({size:.1f} MB)")
    line = f"  ranker: {trees} trees   multi-task heads: {heads}"
    if bundle.text is not None:
        line += f"   vocab: {len(bundle.text.vocabulary):,}"
    print(line)
    print(f"  done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
