"""Stage 1 of the build: catalog + simulated watch log.

    python scripts/01_build_data.py [--users N] [--videos N] [--source SRC]

Writes to data/processed/:
    catalog.parquet        the video corpus
    interactions.parquet   impression-level log (clicked = 0/1)
    users.parquet          user metadata
    gt_item_topics.npy     latent item mixtures  (ground truth / inferred)
    gt_user_topics.npy     latent user personas  (ground truth)
    data_meta.json         provenance for everything above
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import json
import time

import numpy as np

from recsys.config import Paths, load_config
from recsys.data.catalog import build_catalog
from recsys.data.simulator import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=None, help="override simulator.n_users")
    parser.add_argument("--videos", type=int, default=None, help="override catalog size")
    parser.add_argument("--source", default=None,
                        choices=["auto", "youtube_api", "kaggle", "synthetic"])
    args = parser.parse_args()

    Paths.ensure()
    cfg = load_config()

    if args.source:
        cfg.catalog.source = args.source
    if args.videos:
        cfg.catalog.target_size = args.videos
        cfg.catalog.synthetic.n_videos = args.videos
    if args.users:
        cfg.simulator.n_users = args.users

    started = time.time()
    print("=" * 72)
    print("STAGE 1/5  BUILD DATA")
    print("=" * 72)

    print("\n[1] Catalog")
    catalog, item_topics, meta = build_catalog(cfg)

    print("\n[2] Simulating user behaviour")
    result = simulate(catalog, item_topics, cfg.simulator, seed=cfg.project.seed,
                      reference_date=cfg.project.reference_date)

    print("\n[3] Writing artifacts")
    # `latent_quality` is a hidden generative variable. It stays in the stored
    # catalog because evaluation needs it, but the serving layer strips it --
    # see recsys.engine. Anything the model can see, a real system must see too.
    catalog.to_parquet(Paths.catalog, index=False)
    result.interactions.to_parquet(Paths.interactions, index=False)
    result.users.to_parquet(Paths.users, index=False)
    np.save(Paths.gt_item_topics, item_topics.astype(np.float32))
    np.save(Paths.gt_user_topics, result.user_topics.astype(np.float32))
    result.session_intent.to_parquet(Paths.gt_session_intent, index=False)

    clicks = int(result.interactions["clicked"].sum())
    n_users = len(result.users)
    meta.update({
        "n_users": n_users,
        "n_impressions": int(len(result.interactions)),
        "n_clicks": clicks,
        "ctr": round(clicks / max(len(result.interactions), 1), 4),
        "clicks_per_user": round(clicks / max(n_users, 1), 2),
        "matrix_density": round(clicks / max(n_users * len(catalog), 1), 6),
        "n_sessions": int(len(result.session_intent)),
        "sessions_with_intent": float(result.session_intent["has_intent"].mean()),
        "satisfaction_rate": round(float(
            result.interactions.loc[result.interactions["clicked"] == 1, "satisfied"].mean()), 4),
        "seed": cfg.project.seed,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    Paths.data_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for path in (Paths.catalog, Paths.interactions, Paths.users,
                 Paths.gt_item_topics, Paths.gt_user_topics,
                 Paths.gt_session_intent, Paths.data_meta):
        print(f"    {path.relative_to(Paths.root)}  ({path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n  density {meta['matrix_density']:.4%}  |  "
          f"{meta['clicks_per_user']} clicks/user  |  CTR {meta['ctr']:.1%}")
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
