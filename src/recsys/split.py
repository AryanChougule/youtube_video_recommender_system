"""The one temporal cutoff that every model in the pipeline must respect.

Why this module exists
----------------------
The first version of this project trained ALS and co-visitation on the FULL
interaction log, then trained the ranker with their outputs as features and
tested it on held-out feeds. The ranker scored AUC 0.957 and permutation
importance said ``als_score`` accounted for essentially all of it.

That was leakage, not skill. ALS had fitted item factors using the very clicks
the ranker was being asked to predict, so ``als_score`` was quietly carrying the
answer. The user side was causal (we fold in from history), but the ITEM side
had seen the future.

The lesson generalises past this codebase: **in a multi-stage system, an
upstream model trained on the full log poisons every downstream evaluation.**
The cutoff has to be global, not per-model.

So: one cutoff, defined here, imported everywhere.

    interactions before cutoff  ->  train co-visitation, ALS, and the ranker
    interactions after  cutoff  ->  test the ranker, measure NDCG/recall/etc.

Deploying on 80% of the data is a deliberate choice. It means the numbers in
docs/EVALUATION.md describe exactly the artifact that is running in the demo,
rather than a different, better-informed one. ``--full`` refits on everything
for a real production deploy, at the cost of no longer having a clean holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    cutoff_ns: int
    test_size: float

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff_ns)

    def train_mask(self, timestamps: pd.Series | np.ndarray) -> np.ndarray:
        return self._as_ns(timestamps) <= self.cutoff_ns

    def test_mask(self, timestamps: pd.Series | np.ndarray) -> np.ndarray:
        return self._as_ns(timestamps) > self.cutoff_ns

    @staticmethod
    def _as_ns(timestamps) -> np.ndarray:
        series = pd.Series(timestamps)
        if not np.issubdtype(series.dtype, np.integer):
            series = series.astype("int64")
        return series.to_numpy()

    def to_dict(self) -> dict:
        return {"cutoff": str(self.cutoff), "cutoff_ns": int(self.cutoff_ns),
                "test_size": self.test_size}


def temporal_split(interactions: pd.DataFrame, test_size: float = 0.2,
                   time_col: str = "ts") -> TemporalSplit:
    """Split by wall-clock time so the test period is strictly in the future.

    Not a random split, and not a per-user split. A random split lets a user's
    later feeds train the model that is then scored on their earlier ones,
    which is exactly backwards from how the system runs in production.
    """
    ts = interactions[time_col].astype("int64").to_numpy()
    cutoff = int(np.quantile(ts, 1.0 - test_size))
    return TemporalSplit(cutoff_ns=cutoff, test_size=test_size)


def split_interactions(interactions: pd.DataFrame, split: TemporalSplit,
                       time_col: str = "ts") -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = split.train_mask(interactions[time_col])
    return interactions[mask].copy(), interactions[~mask].copy()
