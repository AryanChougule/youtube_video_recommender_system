"""Stage 2 of the build: item text vectors + derived item statistics.

    python scripts/02_build_features.py [--backend tfidf_svd|sentence_transformers]

Writes to artifacts/:
    item_vectors.npy      (n_items, d) L2-normalised item vectors
    text_encoder.joblib   the fitted encoder, so queries land in the same space
    text_backend.json     which backend was used, and its quality stats
    item_stats.parquet    cheap always-available item features for ranking
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from recsys.config import Paths, load_config
from recsys.data.catalog import catalog_text
from recsys.data.schema import catalog_reference_now, derive_catalog_features
from recsys.features.text import build_text_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None,
                        choices=["auto", "tfidf_svd", "sentence_transformers"])
    args = parser.parse_args()

    Paths.ensure()
    cfg = load_config()
    backend = args.backend or cfg.features.text_backend

    started = time.time()
    print("=" * 72)
    print("STAGE 2/5  BUILD FEATURES")
    print("=" * 72)

    catalog = pd.read_parquet(Paths.catalog)
    print(f"\n[1] Text vectors  ({len(catalog):,} videos)")
    index = build_text_index(
        catalog_text(catalog),
        backend=backend,
        dims=cfg.features.svd_dims,
        max_features=cfg.features.tfidf_max_features,
        ngram_max=cfg.features.tfidf_ngram_max,
        st_model=cfg.features.st_model,
        seed=cfg.project.seed,
    )

    np.save(Paths.item_vectors, index.vectors)
    joblib.dump(index.encoder, Paths.text_encoder, compress=3)
    Paths.text_backend.write_text(json.dumps(index.meta, indent=2), encoding="utf-8")

    print("\n[2] Item statistics")
    now = catalog_reference_now(catalog, cfg.project.reference_date)
    stats = derive_catalog_features(catalog, now=now)
    keep = [
        "video_id", "age_days", "log_views", "like_rate", "comment_rate",
        "engagement_rate", "views_per_day", "log_views_per_day",
        "duration_minutes", "is_short", "title_length", "n_tags",
    ]
    stats[keep].to_parquet(Paths.item_stats, index=False)

    # Pandas-free serving bundle. Serving needs column access and row lookup,
    # not joins or parquet IO -- and shipping pandas costs 55MB, which is the
    # difference between fitting a 250MB serverless limit and not.
    from recsys.catalog_view import CatalogView
    CatalogView.from_frames(catalog, stats).save(Paths.serving_npz, Paths.serving_json)
    print(f"    reference 'now' = {now:%Y-%m-%d} "
          f"(anchored to newest video so freshness stays meaningful)")

    # Sanity: a degenerate index (all vectors identical) would silently make
    # every recommendation the same. Catch it here, not in the UI.
    spread = float(np.std(index.vectors @ index.vectors[0]))
    if spread < 1e-3:
        raise RuntimeError(f"item vectors are degenerate (similarity std={spread:.2e})")

    for path in (Paths.item_vectors, Paths.text_encoder, Paths.text_backend,
                 Paths.item_stats, Paths.serving_npz, Paths.serving_json):
        print(f"    {path.relative_to(Paths.root)}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
