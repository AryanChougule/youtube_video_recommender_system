"""Ablation: which text fields should the content index be built from?

Reproduces the measurement quoted in ``recsys.data.catalog.catalog_text``.

The metric is *topical precision@k*: for a sample of videos, take their top-k
neighbours in text space and measure the mean cosine similarity of the
GROUND-TRUTH latent topic mixtures. It answers "is this representation
retrieving videos about the same subject, or merely videos that look alike?"
-- a question you cannot ask of real data, which is exactly the payoff of
having a simulator with known latents.

    python scripts/07_ablate_text_fields.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from recsys.config import Paths, load_config
from recsys.data.schema import split_tags
from recsys.features.text import build_text_index


def topical_precision(vectors: np.ndarray, gt_norm: np.ndarray,
                      k: int = 10, sample: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sample = min(sample, len(vectors))
    idx = rng.choice(len(vectors), size=sample, replace=False)
    sims = vectors[idx] @ vectors.T
    sims[np.arange(sample), idx] = -np.inf
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return float(np.mean([gt_norm[idx[i]] @ gt_norm[top[i]].T for i in range(sample)]))


def main() -> None:
    cfg = load_config()
    catalog = pd.read_parquet(Paths.catalog)
    gt = np.load(Paths.gt_item_topics)
    gt_norm = gt / np.linalg.norm(gt, axis=1, keepdims=True)

    tags = catalog["tags"].fillna("").map(lambda s: " ".join(split_tags(s)))
    title = catalog["title"].fillna("")
    chan = catalog["channel_title"].fillna("")
    cate = catalog["category"].fillna("")
    desc = catalog["description"].fillna("").str.slice(0, 500)
    tail = " . " + chan + " . " + cate + " . " + desc

    variants = {
        "title x2 + tags":  title + " . " + title + " . " + tags + tail,
        "title x1 + tags":  title + " . " + tags + tail,
        "title x1 + tags x3": title + " . " + tags + " . " + tags + " . " + tags + tail,
        "tags only":        tags,
        "title only":       title,
    }

    print(f"{'variant':<24} {'topical precision@10':>22}")
    print("-" * 48)
    results: dict[str, float] = {}
    for name, text in variants.items():
        index = build_text_index(text, dims=cfg.features.svd_dims,
                                 seed=cfg.project.seed, verbose=False)
        results[name] = topical_precision(index.vectors, gt_norm)
        print(f"{name:<24} {results[name]:>22.4f}")

    rng = np.random.default_rng(1)
    a, b = rng.integers(0, len(gt), 5000), rng.integers(0, len(gt), 5000)
    baseline = float(np.mean(np.sum(gt_norm[a] * gt_norm[b], axis=1)))
    print("-" * 48)
    print(f"{'random pair baseline':<24} {baseline:>22.4f}")
    print(f"\nbest: {max(results, key=results.get)}")


if __name__ == "__main__":
    main()
