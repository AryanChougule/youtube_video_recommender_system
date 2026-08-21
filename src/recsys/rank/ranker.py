"""Stage 2 model: a weighted classifier whose ODDS estimate watch time.

Why pointwise and not pairwise/listwise
---------------------------------------
Pointwise (predict P(engage) per item, then sort) is the least sophisticated
of the three learning-to-rank families, and it is what we use. The reasons are
practical, and worth being explicit about rather than hand-waving:

* The watch-time weighting trick below only works pointwise -- it gives the
  score a *calibrated interpretation* (expected watch seconds) rather than an
  arbitrary ordering score. Calibration matters downstream: Stage 3 blends the
  ranker score with freshness and diversity terms, and you cannot sensibly
  blend an uncalibrated pairwise margin with anything.
* Pairwise (LambdaMART/RankNet) optimises ordering directly and would likely
  win on NDCG. It is the honest "next thing to try" -- see docs/FUTURE_WORK.md.

The objective (Covington et al., 2016)
--------------------------------------
Train a binary classifier on clicked/not-clicked, but weight positives by
watch time and negatives by 1. Then

    odds = P/(1-P) = (sum_i T_i) / (N - k)  ~=  E[T]

so ranking by odds ranks by expected watch time. We get a watch-time model out
of a click model. This is exactly the fix YouTube shipped after click-optimised
ranking produced clickbait.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES

_EPS = 1e-9


@dataclass
class RankerMetrics:
    auc: float
    weighted_auc: float
    log_loss: float
    n_train: int
    n_test: int


class Ranker:
    """Gradient-boosted (or logistic) engagement model, scored by odds."""

    def __init__(self, model: str = "hgb", learning_rate: float = 0.08,
                 max_iter: int = 350, seed: int = 42):
        self.model_name = model
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.seed = seed
        self.model = None
        self.metrics: RankerMetrics | None = None
        self.feature_names = list(FEATURE_NAMES)

    def _make_model(self):
        if self.model_name == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, random_state=self.seed),
            )
        if self.model_name == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=self.max_iter, learning_rate=self.learning_rate,
                num_leaves=63, random_state=self.seed, verbose=-1,
            )
        from sklearn.ensemble import HistGradientBoostingClassifier
        # HistGradientBoosting is the default: it ships with scikit-learn (no
        # extra dependency), handles unscaled heterogeneous features natively,
        # and supports the sample weights the whole objective depends on.
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=self.seed,
        )

    def fit(self, X_train, y_train, w_train, X_test, y_test, w_test,
            verbose: bool = True) -> "Ranker":
        from sklearn.metrics import log_loss, roc_auc_score

        self.model = self._make_model()
        self.model.fit(X_train, y_train, sample_weight=w_train)

        probability = self.model.predict_proba(X_test)[:, 1]
        # Unweighted AUC answers "does it rank clicks above skips". Weighted AUC
        # answers "does it rank LONG watches above skips" -- the objective we
        # actually care about. Reporting only the first would hide a model that
        # finds clickbait perfectly well.
        self.metrics = RankerMetrics(
            auc=float(roc_auc_score(y_test, probability)),
            weighted_auc=float(roc_auc_score(y_test, probability, sample_weight=w_test)),
            log_loss=float(log_loss(y_test, probability, labels=[0, 1])),
            n_train=int(len(y_train)), n_test=int(len(y_test)),
        )
        if verbose:
            print(f"  [rank] AUC={self.metrics.auc:.4f}  "
                  f"watch-weighted AUC={self.metrics.weighted_auc:.4f}  "
                  f"logloss={self.metrics.log_loss:.4f}")
        return self

    # -- inference --------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Expected-watch-time proxy: the ODDS, not the probability.

        Because positives were weighted by watch time during training, the odds
        P/(1-P) estimate E[watch time]. Probability alone would rank by "will
        they click", which is the objective we deliberately rejected.
        """
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        p = np.clip(self.model.predict_proba(X)[:, 1], _EPS, 1.0 - _EPS)
        return (p / (1.0 - p)).astype(np.float32)

    def score_probability(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    # -- interpretation ---------------------------------------------------
    def permutation_importance(self, X, y, w, n_repeats: int = 3,
                               seed: int = 42) -> dict[str, float]:
        """Drop in weighted AUC when each feature is shuffled.

        Permutation importance rather than split-gain: gain is biased towards
        high-cardinality features, and permutation answers the question we
        actually have -- "how much does the model's ranking rely on this?"
        """
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(seed)
        base = roc_auc_score(y, self.score_probability(X), sample_weight=w)
        out: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            drops = []
            for _ in range(n_repeats):
                shuffled = X.copy()
                shuffled[:, i] = rng.permutation(shuffled[:, i])
                drops.append(base - roc_auc_score(
                    y, self.score_probability(shuffled), sample_weight=w))
            out[name] = float(np.mean(drops))
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
