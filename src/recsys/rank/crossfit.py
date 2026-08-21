"""Cross-fitted CF features for ranker training.

The bug this exists to fix
--------------------------
Stage 2 uses Stage 1 model outputs (``als_score``, ``covisit_score``) as
features. Train ALS on the training period, then train the ranker on that same
training period, and every training row is IN-SAMPLE for ALS: the item factors
were fitted using the very clicks the ranker is being asked to predict.

Measured on this project:

    als_score AUC on ranker-train rows : 0.856
    als_score AUC on held-out rows     : 0.584

The ranker sees a feature that looks near-oracular, leans on it almost
exclusively, and then collapses at serving time when the feature reverts to its
true strength. The resulting model scored WORSE than its own single best
feature (0.578 vs 0.641 for content similarity alone).

This is not specific to ALS or to recommenders. **Any stacked model whose
inputs include another model's predictions must receive OUT-OF-FOLD
predictions**, or the meta-learner calibrates against optimism it will never
see in production. It is the same reason stacking ensembles use out-of-fold
predictions rather than in-sample ones.

The fix
-------
K-fold cross-fitting by USER. For each fold, refit ALS and co-visitation with
that fold's users held out entirely, then generate their rows' features from
that model. A user's own clicks never shape the factors used to score them --
which is exactly the regime at serving time for anyone the model has not been
retrained on since.

Folds are by user rather than by row because the leak travels through the
user's own interactions; splitting rows would leave it wide open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..rank.dataset import TrainingSet, build_training_set
from ..rank.features import FeatureBuilder
from ..recall.cf import ImplicitALS, build_covisitation, build_interaction_matrices


def assign_user_folds(user_ids: np.ndarray, n_folds: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(user_ids))
    return {u: i % n_folds for i, u in enumerate(shuffled)}


def build_crossfitted_training_set(
    interactions: pd.DataFrame,
    fit_interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    item_stats: pd.DataFrame,
    item_vectors: np.ndarray,
    all_user_ids: list[str],
    cfg: Config,
    objective: str,
    position_decay: float,
    n_folds: int = 4,
    verbose: bool = True,
) -> TrainingSet:
    """Replay every user's timeline using CF models that never saw that user.

    ``fit_interactions`` must already be restricted to the training period;
    this function only removes users, never re-introduces future data.
    """
    video_ids = catalog["video_id"].tolist()
    folds = assign_user_folds(np.asarray(all_user_ids), n_folds, cfg.project.seed)
    fold_of = interactions["user_id"].map(folds)

    parts: list[TrainingSet] = []
    for fold in range(n_folds):
        held_out_users = {u for u, f in folds.items() if f == fold}
        fit_slice = fit_interactions[~fit_interactions["user_id"].isin(held_out_users)]

        if verbose:
            print(f"  [crossfit] fold {fold + 1}/{n_folds}: fitting CF on "
                  f"{fit_slice['user_id'].nunique():,} users "
                  f"(holding out {len(held_out_users):,})")

        user_item, session_item = build_interaction_matrices(
            fit_slice, all_user_ids, video_ids, signal="watch_fraction", verbose=False
        )
        covis = build_covisitation(
            session_item, damping=cfg.recall.covisit_damping,
            min_cooccurrence=2, top_k=100, verbose=False,
        )
        als = ImplicitALS(
            factors=cfg.als.factors, regularization=cfg.als.regularization,
            alpha=cfg.als.alpha, iterations=cfg.als.iterations, seed=cfg.project.seed,
        ).fit(user_item, verbose=False)

        fold_builder = FeatureBuilder(
            catalog=catalog, item_stats=item_stats, item_vectors=item_vectors,
            covisitation=covis, als=als,
        )
        fold_rows = interactions[fold_of == fold]
        if fold_rows.empty:
            continue
        parts.append(build_training_set(
            interactions=fold_rows, feature_builder=fold_builder, video_ids=video_ids,
            negatives_per_positive=cfg.ranker.negatives_per_positive,
            random_negatives_per_positive=cfg.ranker.random_negatives_per_positive,
            objective=objective, position_decay=position_decay,
            seed=cfg.project.seed + fold, verbose=False,
        ))
        if verbose:
            print(f"  [crossfit] fold {fold + 1}: {parts[-1].meta['n_rows']:,} rows")

    if not parts:
        raise RuntimeError("cross-fitting produced no training rows")

    # Feed ids are per-fold; offset them so groups stay unique after concat.
    groups, offset = [], 0
    for part in parts:
        groups.append(part.group + offset)
        offset += int(part.group.max()) + 1

    merged = TrainingSet(
        X=np.concatenate([p.X for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        sample_weight=np.concatenate([p.sample_weight for p in parts]),
        group=np.concatenate(groups),
        timestamp=np.concatenate([p.timestamp for p in parts]),
        meta={
            "n_rows": int(sum(p.meta["n_rows"] for p in parts)),
            "n_positives": int(sum(p.meta["n_positives"] for p in parts)),
            "n_feeds": int(offset),
            "objective": objective,
            "crossfit_folds": n_folds,
            "position_decay_assumed": position_decay,
            "n_features": parts[0].meta["n_features"],
        },
    )
    merged.meta["positive_rate"] = round(
        merged.meta["n_positives"] / max(merged.meta["n_rows"], 1), 4
    )
    if verbose:
        print(f"  [crossfit] merged: {merged.meta['n_rows']:,} rows, "
              f"{merged.meta['n_positives']:,} positives")
    return merged
