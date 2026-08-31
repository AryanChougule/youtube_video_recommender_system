"""A NumPy re-implementation of HistGradientBoosting inference.

Why this exists
---------------
Training a gradient-boosted model is hard; *evaluating* one is not. A fitted
``HistGradientBoostingClassifier`` is, at prediction time, nothing more than a
list of binary trees plus a baseline, and the whole of inference is

    raw = baseline + sum_t tree_t(x)
    p   = sigmoid(raw)

Everything else in scikit-learn -- the histogram binning, the gradient/hessian
machinery, the loss objects -- exists only to *fit* the trees. None of it is
needed to run them.

That distinction turned out to matter a lot, for two reasons.

**1. The pickle was not portable.** ``HistGradientBoostingClassifier`` holds a
``sklearn._loss.loss.HalfBinomialLoss``, which holds a Cython object whose
``__module__`` is the bare string ``_loss``. Unpickling therefore requires a
top-level ``import _loss`` to succeed, which happens to work in a normal
scikit-learn install and does not work once a platform prunes "unused" files
from the dependency tree. The first deployment failed with::

    ModuleNotFoundError: No module named '_loss'

Pinning versions did not help, because the version was never the problem: the
model carries a reference to a *training-time* object that inference never
calls. Exporting the trees drops that reference entirely.

**2. The dependency was the bundle.** scikit-learn plus SciPy is roughly 200MB
of Linux wheels. Removing them from the serving path took the deployed bundle
from 339MB to well under the serverless limit -- and it was exceeding that
limit which triggered the file pruning in the first place. So the two problems
had one cause and one fix.

Equivalence
-----------
This is not an approximation. ``scripts/12_export_serving.py`` asserts that the
exported model reproduces scikit-learn's ``predict_proba`` on the real feature
matrix to within 1e-6, and ``tests/test_serving_export.py`` keeps it honest.

Only numeric splits are supported. The models are trained without categorical
features, and the exporter raises rather than silently mis-predicting if that
ever changes.
"""

from __future__ import annotations

import numpy as np

# Node fields we need. The others (count, gain, depth, bin_threshold,
# bitset_idx) are fitting diagnostics and play no part in prediction.
_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left",
           "left", "right", "is_leaf")


class NumpyHGB:
    """Inference-only HistGradientBoosting: a forest as flat arrays.

    Every tree in the ensemble is concatenated into one set of node arrays,
    with ``tree_offsets`` marking where each starts. That keeps the artifact to
    seven arrays regardless of ensemble size, and lets traversal run over every
    row of a candidate set at once instead of row by row.
    """

    __slots__ = ("value", "feature_idx", "num_threshold", "missing_go_to_left",
                 "left", "right", "is_leaf", "tree_offsets", "baseline")

    def __init__(self, value, feature_idx, num_threshold, missing_go_to_left,
                 left, right, is_leaf, tree_offsets, baseline):
        self.value = np.asarray(value, dtype=np.float64)
        self.feature_idx = np.asarray(feature_idx, dtype=np.int32)
        self.num_threshold = np.asarray(num_threshold, dtype=np.float64)
        self.missing_go_to_left = np.asarray(missing_go_to_left, dtype=bool)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.is_leaf = np.asarray(is_leaf, dtype=bool)
        self.tree_offsets = np.asarray(tree_offsets, dtype=np.int64)
        self.baseline = float(baseline)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_sklearn(cls, est) -> "NumpyHGB":
        """Extract the trees from a fitted binary HistGradientBoostingClassifier."""
        if getattr(est, "n_trees_per_iteration_", 1) != 1:
            raise ValueError(
                "only binary classifiers are supported; got "
                f"n_trees_per_iteration_={est.n_trees_per_iteration_}"
            )
        parts: dict[str, list] = {f: [] for f in _FIELDS}
        offsets = [0]
        total = 0
        for iteration in est._predictors:
            nodes = iteration[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError(
                    "categorical splits are not supported by the NumPy runtime; "
                    "the ranker is trained on numeric features only"
                )
            for field in _FIELDS:
                parts[field].append(nodes[field])
            total += len(nodes)
            offsets.append(total)

        # Child indices are tree-local, so shift them to make one flat array.
        left, right = [], []
        for start, l, r in zip(offsets, parts["left"], parts["right"]):
            left.append(np.asarray(l, dtype=np.int64) + start)
            right.append(np.asarray(r, dtype=np.int64) + start)

        return cls(
            value=np.concatenate(parts["value"]),
            feature_idx=np.concatenate(parts["feature_idx"]),
            num_threshold=np.concatenate(parts["num_threshold"]),
            missing_go_to_left=np.concatenate(parts["missing_go_to_left"]),
            left=np.concatenate(left), right=np.concatenate(right),
            is_leaf=np.concatenate(parts["is_leaf"]),
            tree_offsets=np.asarray(offsets[:-1], dtype=np.int64),
            baseline=float(np.ravel(est._baseline_prediction)[0]),
        )

    # -- inference --------------------------------------------------------
    def decision_function(self, X) -> np.ndarray:
        """Raw additive score, before the logistic link."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        n = len(X)
        out = np.full(n, self.baseline, dtype=np.float64)
        if n == 0:
            return out

        # Every (row, tree) pair walks its tree simultaneously. Looping per
        # tree instead costs one Python iteration per tree, and with ~100 trees
        # over a few hundred candidates the NumPy calls are so small that the
        # interpreter overhead dominates -- that version measured 4.5x slower
        # than scikit-learn's compiled predictor. Batching turns ~100 Python
        # iterations into ~15 (one per level of depth), and the arrays get big
        # enough for NumPy to be worth calling.
        n_trees = len(self.tree_offsets)
        node = np.repeat(self.tree_offsets[None, :], n, axis=0).ravel()
        rows = np.repeat(np.arange(n), n_trees)

        active = ~self.is_leaf[node]
        while active.any():
            idx = np.flatnonzero(active)
            here = node[idx]
            x = X[rows[idx], self.feature_idx[here]]
            # NaN is a real branch decision, not an error: the tree learned
            # which way missing values should go while it was being fitted.
            go_left = np.where(np.isnan(x), self.missing_go_to_left[here],
                               x <= self.num_threshold[here])
            node[idx] = np.where(go_left, self.left[here], self.right[here])
            active[idx] = ~self.is_leaf[node[idx]]

        # One leaf value per (row, tree); sum the trees back per row.
        out += self.value[node].reshape(n, n_trees).sum(axis=1)
        return out

    def predict_proba(self, X) -> np.ndarray:
        """Two-column probabilities, matching the scikit-learn signature.

        Same shape and column order as the estimator this replaced, so the
        calling code in Ranker and MultiTaskRanker did not have to change.
        """
        raw = self.decision_function(X)
        # The logistic function, written out both-ways-stable rather than
        # importing SciPy's expit for four lines of algebra.
        p = np.empty_like(raw)
        pos = raw >= 0
        neg = ~pos
        p[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
        exp_raw = np.exp(raw[neg])
        p[neg] = exp_raw / (1.0 + exp_raw)
        return np.column_stack((1.0 - p, p))

    # -- persistence ------------------------------------------------------
    def arrays(self, prefix: str) -> dict:
        out = {f"{prefix}__{f}": getattr(self, f) for f in _FIELDS}
        out[f"{prefix}__tree_offsets"] = self.tree_offsets
        out[f"{prefix}__baseline"] = np.array([self.baseline])
        return out

    @classmethod
    def from_arrays(cls, blob, prefix: str) -> "NumpyHGB":
        kwargs = {f: blob[f"{prefix}__{f}"] for f in _FIELDS}
        return cls(tree_offsets=blob[f"{prefix}__tree_offsets"],
                   baseline=float(blob[f"{prefix}__baseline"][0]), **kwargs)
