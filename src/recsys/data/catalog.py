"""Catalog orchestration: pick a source, normalise it, guarantee latents.

The rest of the system depends on exactly three things regardless of where the
data came from:

    catalog       DataFrame conforming to CATALOG_SCHEMA
    item_topics   (n_items, N_TOPICS) row-normalised latent mixtures
    meta          provenance, so every artifact is traceable to its source

For the synthetic source the latent topics are ground truth by construction.
For real sources they do not exist, so we *infer* them with NMF over TF-IDF of
the video text.  That is an honest substitution: the simulator needs some
latent structure to generate coherent behaviour from, and an inferred topic
model is the standard way to get one.  It does mean that on real data the
"recovered latent structure" evaluation measures agreement with NMF rather
than with truth -- flagged in docs/EVALUATION.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, Paths
from . import kaggle_loader, synthetic, youtube_api
from .schema import split_tags
from .topics import N_TOPICS


def infer_item_topics(
    catalog: pd.DataFrame, n_topics: int = N_TOPICS, seed: int = 42, verbose: bool = True
) -> np.ndarray:
    """Infer latent topic mixtures for a real catalog via TF-IDF + NMF."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = (
        catalog["title"].fillna("") + " . "
        + catalog["tags"].fillna("").str.replace("|", " ", regex=False) + " . "
        + catalog["description"].fillna("").str.slice(0, 400)
    )
    tfidf = TfidfVectorizer(
        max_features=40000, ngram_range=(1, 2), min_df=3,
        stop_words="english", sublinear_tf=True,
    )
    matrix = tfidf.fit_transform(text)
    if verbose:
        print(f"  [topics] TF-IDF {matrix.shape}, fitting NMF({n_topics})...")
    model = NMF(n_components=n_topics, init="nndsvda", random_state=seed,
                max_iter=400, tol=1e-4)
    weights = model.fit_transform(matrix)
    weights = np.maximum(weights, 1e-9)
    return weights / weights.sum(axis=1, keepdims=True)


def _ensure_latent_quality(catalog: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a ``latent_quality`` column for the simulator.

    Synthetic data has true hidden appeal. For real data we proxy it with the
    like rate, normalised to a comparable log-normal-ish scale. It is a proxy,
    not truth -- but the simulator only needs a plausible appeal signal.
    """
    if "latent_quality" in catalog.columns:
        return catalog
    views = catalog["view_count"].clip(lower=1)
    like_rate = (catalog["like_count"] / views).clip(0, 0.25)
    median = float(like_rate[like_rate > 0].median() or 0.02)
    catalog = catalog.copy()
    catalog["latent_quality"] = (like_rate / median).clip(0.15, 6.0).fillna(1.0)
    return catalog


def build_catalog(
    cfg: Config, verbose: bool = True
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build the catalog from the configured source, with fallbacks."""
    source = cfg.catalog.source
    order = (
        ["youtube_api", "kaggle", "synthetic"] if source == "auto" else [source]
    )

    catalog: Optional[pd.DataFrame] = None
    item_topics: Optional[np.ndarray] = None
    used = ""
    notes: list[str] = []

    for candidate in order:
        try:
            if candidate == "youtube_api":
                key = youtube_api.api_key_from_env(cfg.catalog.youtube_api.api_key_env)
                if not key:
                    notes.append(
                        f"youtube_api skipped: ${cfg.catalog.youtube_api.api_key_env} not set"
                    )
                    continue
                if verbose:
                    print("  [catalog] fetching from YouTube Data API v3...")
                catalog = youtube_api.fetch_catalog(
                    api_key=key,
                    target_size=cfg.catalog.target_size,
                    queries_per_category=cfg.catalog.youtube_api.queries_per_category,
                    max_results_per_query=cfg.catalog.youtube_api.max_results_per_query,
                    cache_path=Paths.raw / "youtube_api_cache.jsonl",
                    verbose=verbose,
                )

            elif candidate == "kaggle":
                catalog = kaggle_loader.load_catalog(
                    csv_glob=cfg.catalog.kaggle.csv_glob,
                    target_size=cfg.catalog.target_size,
                    verbose=verbose,
                )

            else:
                if verbose:
                    print("  [catalog] generating synthetic catalog...")
                catalog, item_topics, _channels = synthetic.generate_catalog(
                    n_videos=cfg.catalog.synthetic.n_videos,
                    n_channels=cfg.catalog.synthetic.n_channels,
                    seed=cfg.project.seed,
                    reference_date=cfg.project.reference_date,
                )

            used = candidate
            break

        except Exception as exc:                     # noqa: BLE001 - fall through
            notes.append(f"{candidate} failed: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  [catalog] {candidate} unavailable ({exc}); trying next source")
            catalog = None

    if catalog is None or catalog.empty:
        raise RuntimeError(
            "could not build a catalog from any source.\n  " + "\n  ".join(notes)
        )

    if item_topics is None:
        item_topics = infer_item_topics(catalog, seed=cfg.project.seed, verbose=verbose)

    catalog = _ensure_latent_quality(catalog).reset_index(drop=True)

    meta = {
        "source": used,
        "n_videos": int(len(catalog)),
        "n_channels": int(catalog["channel_id"].nunique()),
        "n_categories": int(catalog["category"].nunique()),
        "latent_topics": int(item_topics.shape[1]),
        "latent_topics_origin": "ground_truth" if used == "synthetic" else "inferred_nmf",
        "notes": notes,
    }
    if verbose:
        print(f"  [catalog] source={used}  videos={meta['n_videos']:,}  "
              f"channels={meta['n_channels']:,}  topics={meta['latent_topics_origin']}")
    return catalog, item_topics, meta


def catalog_text(catalog: pd.DataFrame) -> pd.Series:
    """The text field used for content-based retrieval.

    Field weighting was chosen by MEASUREMENT, not intuition. The obvious move
    is to repeat the title (creators optimise titles hard, so they carry the
    most relevance signal). Measured against ground-truth topic vectors --
    "what fraction of a video's top-10 text neighbours share its true topic
    mixture" -- that hypothesis loses:

        title x2 + tags      0.8875
        title x1 + tags      0.9040   <- chosen
        title x1 + tags x3   0.9028
        tags only            0.8863
        title only           0.6518
        random pair          0.0943

    Titles follow strong FORMAT templates ("... Ranked From Worst to Best"),
    so up-weighting them makes the index retrieve videos with the same title
    shape rather than the same subject. Tags carry topic with far less
    stylistic noise. Reproduce with scripts/07_ablate_text_fields.py.
    """
    tags = catalog["tags"].fillna("").map(lambda s: " ".join(split_tags(s)))
    return (
        catalog["title"].fillna("") + " . "
        + tags + " . "
        + catalog["channel_title"].fillna("") + " . "
        + catalog["category"].fillna("") + " . "
        + catalog["description"].fillna("").str.slice(0, 500)
    ).str.strip()
