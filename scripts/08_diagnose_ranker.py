"""Diagnostic: is the ranker weak, or is the task near its noise ceiling?

An AUC number alone is uninterpretable. 0.57 could mean a broken model, or it
could mean the label is mostly irreducible noise. The only way to tell is to
compare against an ORACLE that scores candidates using the simulator's own
generative parameters -- the true user persona, the true hidden video quality,
the true duration preference.

The oracle is the best any model could possibly do without knowing the random
draws, because it literally IS the click model, minus the two things no ranker
can observe:

  * WHERE in the feed the item was shown. Our cascade click model makes
    position hugely predictive (34% CTR at rank 0 vs 0.14% at rank 7), and the
    slot order is randomised, so this is pure noise from the ranker's side.
  * The Bernoulli coin flips themselves.

If the oracle also scores near 0.6, then 0.57 is close to optimal and the model
is fine. If the oracle scores 0.9, we have a real bug.

    python scripts/08_diagnose_ranker.py [--users N]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.data.simulator import (
    CLICK_BIAS, W_AFFINITY, W_DURATION_FIT, W_POPULARITY, W_QUALITY,
)
from recsys.rank.dataset import build_training_set
from recsys.split import temporal_split


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def within_feed_top1(scores: np.ndarray, labels: np.ndarray,
                     groups: np.ndarray) -> float:
    """Fraction of feeds where the clicked item is scored highest.

    The right metric for this data: negatives are sampled from the SAME feed,
    so the question is genuinely "pick the click out of this page", and a
    global AUC would blur together feeds of very different difficulty.
    """
    order = np.argsort(groups, kind="stable")
    scores, labels, groups = scores[order], labels[order], groups[order]
    bounds = np.flatnonzero(np.diff(groups)) + 1
    hits = total = 0
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(groups)]):
        if labels[start:stop].sum() != 1:
            continue                       # only feeds with exactly one click
        total += 1
        hits += int(np.argmax(scores[start:stop]) == np.argmax(labels[start:stop]))
    return hits / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=700)
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg, with_ranker=True)
    interactions = pd.read_parquet(Paths.interactions)
    users = pd.read_parquet(Paths.users)
    gt_user_topics = np.load(Paths.gt_user_topics)
    gt_item_topics = np.load(Paths.gt_item_topics)

    rng = np.random.default_rng(cfg.project.seed)
    sample_users = rng.choice(users["user_id"].to_numpy(),
                              size=min(args.users, len(users)), replace=False)
    subset = interactions[interactions["user_id"].isin(set(sample_users))]
    print(f"diagnosing on {len(sample_users):,} users / {len(subset):,} impressions")

    dataset = build_training_set(
        interactions=subset, feature_builder=art.features,
        video_ids=art.catalog["video_id"].tolist(),
        negatives_per_positive=cfg.ranker.negatives_per_positive,
        objective=cfg.ranker.objective,
        position_decay=cfg.simulator.position_bias_decay,
        seed=cfg.project.seed, verbose=False,
    )
    split = temporal_split(interactions, test_size=cfg.ranker.test_size)
    test = dataset.timestamp > split.cutoff_ns
    print(f"test rows: {test.sum():,} in {len(np.unique(dataset.group[test])):,} feeds\n")

    y = dataset.y[test].astype(int)
    g = dataset.group[test]
    X = dataset.X[test]

    # ---- rebuild the oracle score from the simulator's own parameters ----
    # We need (user, item) identity per row, which the training set does not
    # carry, so replay the same selection deterministically via the features:
    # log_views and duration are unique enough to recover the item index.
    # Simpler and exact: recompute from the raw log instead.
    order_cols = ["user_id", "ts", "rank_shown"]
    sub = subset.sort_values(order_cols, kind="stable").reset_index(drop=True)
    item_index = {v: i for i, v in enumerate(art.catalog["video_id"])}
    user_index = {u: i for i, u in enumerate(users["user_id"])}
    sub["item"] = sub["video_id"].map(item_index)
    sub["uidx"] = sub["user_id"].map(user_index)
    sub = sub[sub["item"].notna() & sub["uidx"].notna()]
    sub["item"] = sub["item"].astype(int)
    sub["uidx"] = sub["uidx"].astype(int)
    sub = sub[sub["ts"].astype("int64") > split.cutoff_ns]

    views = art.catalog["view_count"].to_numpy(dtype=float)
    log_views_z = zscore(np.log1p(views))
    quality = art.catalog["latent_quality"].to_numpy(dtype=float)
    quality_z = zscore(np.log(quality))
    duration_min = art.catalog["duration_seconds"].to_numpy(dtype=float) / 60.0
    mainstream = users["mainstream"].to_numpy()
    pref_minutes = users["preferred_minutes"].to_numpy()

    it = sub["item"].to_numpy()
    ut = sub["uidx"].to_numpy()
    affinity = np.einsum("ij,ij->i", gt_user_topics[ut], gt_item_topics[it])
    oracle = (
        CLICK_BIAS
        + W_AFFINITY * affinity * 12.0
        + W_POPULARITY * mainstream[ut] * 2.0 * log_views_z[it]
        + W_QUALITY * quality_z[it]
        + W_DURATION_FIT * (-np.abs(np.log(duration_min[it] / pref_minutes[ut])))
    )
    oracle_y = sub["clicked"].to_numpy(dtype=int)
    oracle_g = pd.factorize(sub["user_id"].astype(str) + "|" + sub["ts"].astype(str))[0]

    # position-only baseline, on the same rows, to size the noise floor
    position = -sub["rank_shown"].to_numpy(dtype=float)

    print(f"{'scorer':<34} {'AUC':>8} {'top-1 in feed':>15}")
    print("-" * 60)
    rand = rng.normal(size=len(y))
    print(f"{'random':<34} {roc_auc_score(y, rand):>8.4f} "
          f"{within_feed_top1(rand, y, g):>15.1%}")
    pop_col = art.features.log_views  # popularity-only, from the feature matrix
    pop_score = X[:, art.ranker.feature_names.index("log_views")]
    print(f"{'popularity only (log_views)':<34} {roc_auc_score(y, pop_score):>8.4f} "
          f"{within_feed_top1(pop_score, y, g):>15.1%}")
    als_col = X[:, art.ranker.feature_names.index("als_score")]
    print(f"{'ALS score only':<34} {roc_auc_score(y, als_col):>8.4f} "
          f"{within_feed_top1(als_col, y, g):>15.1%}")
    content_col = X[:, art.ranker.feature_names.index("content_sim_profile")]
    print(f"{'content sim only':<34} {roc_auc_score(y, content_col):>8.4f} "
          f"{within_feed_top1(content_col, y, g):>15.1%}")
    ranked = art.ranker.score(X)
    print(f"{'LEARNED RANKER (21 features)':<34} {roc_auc_score(y, ranked):>8.4f} "
          f"{within_feed_top1(ranked, y, g):>15.1%}")
    print("-" * 60)
    print(f"{'ORACLE (true generative params)':<34} {roc_auc_score(oracle_y, oracle):>8.4f} "
          f"{within_feed_top1(oracle, oracle_y, oracle_g):>15.1%}")
    print(f"{'position only (unobservable)':<34} {roc_auc_score(oracle_y, position):>8.4f} "
          f"{within_feed_top1(position, oracle_y, oracle_g):>15.1%}")
    print("\nThe oracle is the ceiling for any feature-based ranker on this log.")
    print("The gap between the oracle and 'position only' is how much of the")
    print("label is decided by WHERE an item was shown rather than what it is.")


if __name__ == "__main__":
    main()
