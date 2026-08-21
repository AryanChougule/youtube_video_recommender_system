"""Stage 3 of the build: collaborative filtering models.

    python scripts/03_train_cf.py [--factors N] [--iterations N] [--damping F]

Writes to artifacts/:
    item_cooc.npz          top-k co-visitation neighbours + scores + popularity
    als_user_factors.npy   (n_users, f) learned user vectors
    als_item_factors.npy   (n_items, f) learned item vectors
    index_meta.json        id <-> row-index contract for every matrix above
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np
import pandas as pd

from recsys.config import Paths, load_config
from recsys.split import split_interactions, temporal_split
from recsys.recall.cf import (
    ImplicitALS,
    build_covisitation,
    build_interaction_matrices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--damping", type=float, default=None,
                        help="co-visitation popularity damping (0.5 = cosine)")
    parser.add_argument("--full", action="store_true",
                        help="train on ALL interactions (production refit; "
                             "destroys the clean holdout -- see src/recsys/split.py)")
    args = parser.parse_args()

    Paths.ensure()
    cfg = load_config()
    factors = args.factors or cfg.als.factors
    damping = args.damping if args.damping is not None else cfg.recall.covisit_damping
    iterations = args.iterations or cfg.als.iterations

    started = time.time()
    print("=" * 72)
    print("STAGE 3/5  TRAIN COLLABORATIVE FILTERING")
    print("=" * 72)

    catalog = pd.read_parquet(Paths.catalog)
    users = pd.read_parquet(Paths.users)
    interactions = pd.read_parquet(Paths.interactions)

    video_ids = catalog["video_id"].tolist()
    user_ids = users["user_id"].tolist()

    # CRITICAL: CF must not see the test period. It feeds the ranker, and an
    # upstream model trained on the full log poisons every downstream metric.
    # See src/recsys/split.py for the leakage this prevents.
    split = temporal_split(interactions, test_size=cfg.ranker.test_size)
    if args.full:
        print("\n  !! --full: training on ALL interactions. Holdout metrics "
              "from this build are NOT valid.")
        fit_interactions = interactions
    else:
        fit_interactions, _held_out = split_interactions(interactions, split)
        print(f"\n  temporal cutoff {split.cutoff:%Y-%m-%d %H:%M} -> fitting on "
              f"{len(fit_interactions):,}/{len(interactions):,} impressions")

    print("\n[1] Interaction matrices")
    user_item, session_item = build_interaction_matrices(
        fit_interactions, user_ids, video_ids, signal="watch_fraction"
    )

    print("\n[2] Item-item co-visitation")
    covis = build_covisitation(
        session_item, damping=damping, min_cooccurrence=2, top_k=100
    )
    np.savez_compressed(
        Paths.cooccurrence,
        neighbours=covis.neighbours, scores=covis.scores, popularity=covis.popularity,
    )

    print(f"\n[3] Implicit ALS  (f={factors}, alpha={cfg.als.alpha}, "
          f"lambda={cfg.als.regularization}, {iterations} iters)")
    als = ImplicitALS(
        factors=factors, regularization=cfg.als.regularization,
        alpha=cfg.als.alpha, iterations=iterations, seed=cfg.project.seed,
    ).fit(user_item)
    np.save(Paths.als_user_factors, als.user_factors.astype(np.float32))
    np.save(Paths.als_item_factors, als.item_factors.astype(np.float32))

    # The row-index contract. Every artifact (vectors, factors, neighbours) is
    # indexed by catalog row order; if that ever drifts, recommendations
    # silently point at the wrong videos. Recording the order lets the serving
    # layer assert it at load time instead of failing invisibly.
    meta = {
        "n_users": len(user_ids),
        "n_items": len(video_ids),
        "als_factors": factors,
        "als_iterations": iterations,
        "als_alpha": cfg.als.alpha,
        "als_regularization": cfg.als.regularization,
        "covisitation_damping": damping,
        "items_with_covisit_neighbours": int((covis.scores[:, 0] > 0).sum()),
        "signal": "watch_fraction",
        "trained_on_full_log": bool(args.full),
        "temporal_split": split.to_dict(),
        "first_video_id": video_ids[0],
        "last_video_id": video_ids[-1],
        "first_user_id": user_ids[0],
        "last_user_id": user_ids[-1],
        "seed": cfg.project.seed,
    }
    Paths.index_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for path in (Paths.cooccurrence, Paths.als_user_factors,
                 Paths.als_item_factors, Paths.index_meta):
        print(f"    {path.relative_to(Paths.root)}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
