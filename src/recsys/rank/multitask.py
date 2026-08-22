"""Multi-objective ranking: predict several outcomes, then combine them.

Why bother
----------
The single-objective ranker optimises expected watch time, which is already a
big improvement on clicks. But watch time is a *proxy* for satisfaction, and
proxies drift from the thing they proxy for.

Measured on this project's generated log, clickbait is the wedge:

    top-decile clickbait : watch 0.425, satisfied 42%
    bottom-half clickbait: watch 0.438, satisfied 73%

Watch time is nearly BLIND to clickbait -- the correlation is +0.01, because
clickbait holds attention slightly but also pulls in lower-affinity viewers,
and the two effects cancel. Satisfaction is not blind to it at all. A
watch-time-only objective therefore cannot distinguish a genuinely good video
from one that merely kept you waiting for a payoff that never came.

That gap is the entire justification for this module. If watch time and
satisfaction agreed, multi-task ranking would be complexity for its own sake.

Design
------
Independent heads over a SHARED feature matrix, one binary classifier each:

    P(click)        did they open it
    P(long_watch)   did they watch >= 50%
    P(completion)   did they watch >= 90%
    P(liked)        explicit positive
    P(satisfied)    the survey-like signal
    P(dismissed)    bounced almost immediately  (enters with a NEGATIVE weight)

    value = sum_k w_k * P_k(x)

Each head is a calibrated probability, so the weights are interpretable and an
evaluator can move them in the UI and reason about what should happen.

Rejected: a shared-trunk Mixture-of-Experts (Zhao et al., 2019). It is the
right answer at YouTube's scale -- it lets tasks share representation while
gating away destructive interference -- but it needs a neural net, and with
6k items and one simulated label stream per outcome it would be capacity we
cannot feed. Independent GBDT heads give the same *product* capability
(controllable multi-objective ranking) at a fraction of the risk. The upgrade
path is documented in docs/FUTURE_WORK.md.

The honest cost: independent heads cannot share representation, so correlated
tasks re-learn the same structure six times. That is wasteful, not wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES

_EPS = 1e-9

#: Default objective weights. These are an ENGINEERING CHOICE, not YouTube's
#: numbers -- YouTube has never published theirs, and anyone claiming otherwise
#: is guessing. They encode a product stance: a click is worth little on its
#: own, a satisfying watch is worth most, and bouncing is actively bad.
DEFAULT_WEIGHTS: dict[str, float] = {
    "click": 0.10,
    "long_watch": 0.25,
    "completion": 0.15,
    "liked": 0.15,
    "satisfied": 0.40,
    "dismissed": -0.20,
}


@dataclass
class MultiTaskMetrics:
    per_task_auc: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0


class MultiTaskRanker:
    """Several calibrated heads over one feature matrix, combined by weights."""

    def __init__(self, tasks: list[str] | None = None,
                 weights: dict[str, float] | None = None,
                 learning_rate: float = 0.08, max_iter: int = 250,
                 seed: int = 42):
        self.tasks = tasks or list(DEFAULT_WEIGHTS)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.seed = seed
        self.models: dict[str, object] = {}
        self.feature_names = list(FEATURE_NAMES)
        self.metrics = MultiTaskMetrics()

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=self.seed,
        )

    def fit(self, X_train, task_y_train: dict, w_train,
            X_test, task_y_test: dict, verbose: bool = True) -> "MultiTaskRanker":
        from sklearn.metrics import roc_auc_score

        self.metrics = MultiTaskMetrics(n_train=len(X_train), n_test=len(X_test))
        for task in self.tasks:
            y = task_y_train.get(task)
            if y is None or y.sum() < 50 or y.sum() == len(y):
                if verbose:
                    print(f"  [multitask] skipping '{task}' (too few positives)")
                continue
            model = self._make()
            # IPS / watch-time weights apply to the click head only; the outcome
            # heads are conditioned on the click having happened, so reweighting
            # them by watch time would double-count the very thing they predict.
            model.fit(X_train, y, sample_weight=w_train if task == "click" else None)
            self.models[task] = model

            y_te = task_y_test.get(task)
            if y_te is not None and 0 < y_te.sum() < len(y_te):
                auc = float(roc_auc_score(y_te, model.predict_proba(X_test)[:, 1]))
                self.metrics.per_task_auc[task] = auc
                if verbose:
                    print(f"  [multitask] {task:<12} AUC {auc:.4f}  "
                          f"(positive rate {y.mean():.2%})")
        return self

    # -- inference --------------------------------------------------------
    def predict_tasks(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if len(X) == 0:
            return {t: np.zeros(0, dtype=np.float32) for t in self.models}
        return {t: m.predict_proba(X)[:, 1].astype(np.float32)
                for t, m in self.models.items()}

    def score(self, X: np.ndarray, weights: dict[str, float] | None = None
              ) -> np.ndarray:
        """Weighted value score. Pass ``weights`` to override at request time.

        This is what makes the Recommendation Lab possible: the heads are fixed
        at training time, but the *objective* is chosen per request, so an
        evaluator can turn the system from engagement-maximising to
        satisfaction-maximising and watch the feed change without retraining.
        """
        if not self.models:
            raise RuntimeError("ranker is not fitted")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        w = dict(self.weights)
        if weights:
            w.update(weights)
        probs = self.predict_tasks(X)
        total = np.zeros(len(X), dtype=np.float32)
        for task, p in probs.items():
            total += float(w.get(task, 0.0)) * p
        return total

    def explain_scores(self, X: np.ndarray, weights: dict[str, float] | None = None
                       ) -> list[dict[str, float]]:
        """Per-item contribution of each objective, for the 'Why?' panel."""
        w = dict(self.weights)
        if weights:
            w.update(weights)
        probs = self.predict_tasks(X)
        out: list[dict[str, float]] = []
        for i in range(len(X)):
            row = {t: round(float(w.get(t, 0.0) * p[i]), 4) for t, p in probs.items()}
            row["_probabilities"] = {t: round(float(p[i]), 4) for t, p in probs.items()}
            out.append(row)
        return out
