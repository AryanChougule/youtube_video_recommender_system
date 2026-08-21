"""Stage 4 of the build: the Stage 2 learning-to-rank model.

    python scripts/04_train_ranker.py [--objective watch_time|click] [--no-ips]
                                      [--no-crossfit] [--folds N]

Two safeguards make the reported numbers trustworthy, and both were added
after measurement showed the model was broken without them:

1. **Temporal split.** Train on the past, test on the future. A random split
   lets a user's later feeds train the model that is then scored on their
   earlier ones -- backwards from how the system runs.

2. **Cross-fitted CF features.** ALS and co-visitation feed the ranker, so
   in-sample rows made ``als_score`` look near-oracular (AUC 0.856 in-sample vs
   0.584 held out). The ranker over-trusted it and ended up WORSE than its own
   best single feature. See src/recsys/rank/crossfit.py.

Writes artifacts/ranker.joblib and artifacts/ranker_report.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.metrics import within_group_top1
from recsys.rank.crossfit import build_crossfitted_training_set
from recsys.rank.dataset import build_training_set
from recsys.rank.ranker import Ranker
from recsys.split import split_interactions, temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", default=None, choices=["watch_time", "click"])
    parser.add_argument("--no-ips", action="store_true")
    parser.add_argument("--no-crossfit", action="store_true",
                        help="reproduce the leaky baseline (for the ablation)")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--model", default=None, choices=["hgb", "logistic", "lightgbm"])
    parser.add_argument("--out", default=None, help="output filename (for ablations)")
    args = parser.parse_args()

    Paths.ensure()
    cfg = load_config()
    objective = args.objective or cfg.ranker.objective
    use_ips = cfg.ranker.use_ips and not args.no_ips
    model_name = args.model or cfg.ranker.model
    out_path = Paths.artifacts / args.out if args.out else Paths.ranker
    position_decay = cfg.simulator.position_bias_decay if use_ips else 1.0

    started = time.time()
    print("=" * 72)
    print("STAGE 4/5  TRAIN RANKER")
    print("=" * 72)
    print(f"  objective={objective}  ips={use_ips}  crossfit="
          f"{'off' if args.no_crossfit else args.folds}  model={model_name}")

    art = load_artifacts(cfg, with_ranker=False)
    interactions = pd.read_parquet(Paths.interactions)
    users = pd.read_parquet(Paths.users)

    split = temporal_split(interactions, test_size=cfg.ranker.test_size)
    cf_cutoff = art.index_meta.get("temporal_split", {}).get("cutoff_ns")
    if cf_cutoff is not None and cf_cutoff != split.cutoff_ns:
        raise RuntimeError(
            f"cutoff {split.cutoff_ns} disagrees with the one CF was trained "
            f"under ({cf_cutoff}). Rerun scripts/03_train_cf.py."
        )
    if art.index_meta.get("trained_on_full_log"):
        print("    !! CF trained with --full: these metrics are NOT leak-free.")
    fit_interactions, _ = split_interactions(interactions, split)

    print("\n[1] Replaying user timelines (causal feature construction)")
    if args.no_crossfit:
        print("    !! cross-fitting DISABLED: als_score will be in-sample and "
              "the model will over-trust it.")
        dataset = build_training_set(
            interactions=interactions, feature_builder=art.features,
            video_ids=art.catalog["video_id"].tolist(),
            negatives_per_positive=cfg.ranker.negatives_per_positive,
            objective=objective, position_decay=position_decay,
            seed=cfg.project.seed,
        )
    else:
        dataset = build_crossfitted_training_set(
            interactions=interactions, fit_interactions=fit_interactions,
            catalog=art.catalog, item_stats=art.item_stats,
            item_vectors=art.text_index.vectors,
            all_user_ids=users["user_id"].tolist(), cfg=cfg,
            objective=objective, position_decay=position_decay,
            n_folds=args.folds,
        )

    print("\n[2] Temporal split")
    train = dataset.timestamp <= split.cutoff_ns
    test = ~train
    if test.sum() == 0 or train.sum() == 0:
        raise RuntimeError("temporal split produced an empty side")
    print(f"    train {train.sum():,} rows  |  test {test.sum():,} rows  "
          f"(cutoff {split.cutoff:%Y-%m-%d %H:%M})")

    print("\n[3] Fitting")
    ranker = Ranker(model=model_name, learning_rate=cfg.ranker.learning_rate,
                    max_iter=cfg.ranker.max_iter, seed=cfg.project.seed)
    ranker.fit(
        dataset.X[train], dataset.y[train], dataset.sample_weight[train],
        dataset.X[test], dataset.y[test], dataset.sample_weight[test],
    )

    # Within-feed top-1 is the metric that matches the product question:
    # given this page, did we put the video they actually watched on top?
    # Negatives are sampled from the SAME feed, so random = 1/(1+negatives).
    scores_test = ranker.score(dataset.X[test])
    top1 = within_group_top1(scores_test, dataset.y[test].astype(int), dataset.group[test])
    random_top1 = 1.0 / (1 + cfg.ranker.negatives_per_positive
                         + cfg.ranker.random_negatives_per_positive)
    print(f"  [rank] within-feed top-1 = {top1:.1%}  (random = {random_top1:.1%})")

    # Sanity gate: a stacked model must beat its own best single input, or
    # something upstream is leaking. This check is what caught the ALS leak.
    print("\n[4] Single-feature baselines on the same test rows")
    from sklearn.metrics import roc_auc_score
    best_single, best_name = 0.0, ""
    for name in ("content_sim_profile", "content_sim_max_hist", "als_score", "covisit_score"):
        col = dataset.X[test][:, ranker.feature_names.index(name)]
        auc = roc_auc_score(dataset.y[test], col)
        single_top1 = within_group_top1(col, dataset.y[test].astype(int), dataset.group[test])
        print(f"    {name:<24} AUC {auc:.4f}   top-1 {single_top1:.1%}")
        if auc > best_single:
            best_single, best_name = auc, name
    if ranker.metrics.auc < best_single:
        print(f"\n    WARNING: the model (AUC {ranker.metrics.auc:.4f}) is worse than "
              f"'{best_name}' alone ({best_single:.4f}).\n"
              f"    That usually means an upstream feature is leaking. Investigate "
              f"before trusting any downstream metric.")

    print("\n[5] Permutation importance (weighted AUC drop)")
    rng = np.random.default_rng(cfg.project.seed)
    test_idx = np.where(test)[0]
    probe = rng.choice(test_idx, size=min(40000, len(test_idx)), replace=False)
    importance = ranker.permutation_importance(
        dataset.X[probe], dataset.y[probe], dataset.sample_weight[probe],
        n_repeats=3, seed=cfg.project.seed,
    )
    for name, drop in list(importance.items())[:10]:
        print(f"    {name:<26} {drop:+.4f} {'#' * max(1, int(drop * 400))}")

    joblib.dump(ranker, out_path, compress=3)
    report = {
        "objective": objective, "use_ips": use_ips, "model": model_name,
        "crossfit_folds": None if args.no_crossfit else args.folds,
        "metrics": {**ranker.metrics.__dict__,
                    "within_feed_top1": round(top1, 4),
                    "within_feed_top1_random": round(random_top1, 4)},
        "dataset": dataset.meta,
        "permutation_importance": importance,
        "temporal_split": split.to_dict(),
    }
    report_path = out_path.with_name(out_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n    {out_path.relative_to(Paths.root)}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
