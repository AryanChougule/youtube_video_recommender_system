"""Ranking and beyond-accuracy metrics.

Accuracy metrics (precision/recall/NDCG/MAP/MRR) answer "did we put the right
things at the top". They are necessary and insufficient: a recommender that
shows every user the same 20 viral videos can score respectably on all of them
while being a terrible product.

So the beyond-accuracy family is treated as first-class here, not an appendix:

    coverage    what fraction of the catalog ever gets recommended
    diversity   how different the items within one list are from each other
    novelty     how obscure the recommendations are (self-information)
    serendipity relevant AND not what a popularity baseline would have shown

Reporting accuracy alone is how you ship a filter bubble and call it a win.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for i in top if i in relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for i in top if i in relevant) / len(relevant)


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """AP: precision measured at each hit, averaged. Rewards early hits."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def reciprocal_rank(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain with the log2(rank+1) discount.

    The discount encodes the empirical fact that attention falls off sharply
    with position -- which our own logs confirm (34% CTR at rank 0 vs 0.14% at
    rank 7). A metric that treated position 1 and position 10 alike would be
    measuring something users do not experience.
    """
    return sum(g / np.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(recommended: Sequence[int], relevance: dict[int, float], k: int) -> float:
    """Normalised DCG. Supports graded relevance (e.g. watch fraction)."""
    top = list(recommended)[:k]
    actual = dcg(relevance.get(i, 0.0) for i in top)
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def hit_rate(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return float(any(i in relevant for i in list(recommended)[:k]))


# ---------------------------------------------------------------------------
# Beyond accuracy
# ---------------------------------------------------------------------------


def catalog_coverage(all_recommended: Iterable[Sequence[int]], n_items: int) -> float:
    """Share of the catalog that appears in at least one recommendation list.

    Low coverage means most of the corpus is unreachable -- bad for creators,
    and a sign the model has collapsed onto a popular core.
    """
    seen: set[int] = set()
    for lst in all_recommended:
        seen.update(int(i) for i in lst)
    return len(seen) / max(n_items, 1)


def gini_coefficient(counts: np.ndarray) -> float:
    """Inequality of exposure across items. 0 = perfectly even, 1 = winner-take-all.

    Coverage says how many items were shown at all; Gini says whether the
    exposure was concentrated on a handful of them. A system can have 90%
    coverage and still send 99% of impressions to 1% of the catalog.
    """
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * counts).sum()) / (n * counts.sum()) - (n + 1) / n)


def intra_list_diversity(recommended: Sequence[int], vectors: np.ndarray) -> float:
    """Mean pairwise (1 - cosine) inside one list. Higher = more varied page."""
    idx = list(recommended)
    if len(idx) < 2:
        return 0.0
    sub = vectors[np.asarray(idx, dtype=int)]
    sims = sub @ sub.T
    n = len(idx)
    off_diagonal = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off_diagonal)


def novelty(recommended: Sequence[int], popularity: np.ndarray) -> float:
    """Mean self-information -log2(p(item)) of the recommended items.

    Recommending only blockbusters gives low novelty. This is the metric that
    catches a system quietly optimising itself into a popularity list.
    """
    idx = np.asarray(list(recommended), dtype=int)
    if len(idx) == 0:
        return 0.0
    total = max(float(popularity.sum()), 1.0)
    p = np.maximum(popularity[idx], 1.0) / total
    return float(np.mean(-np.log2(p)))


def serendipity(recommended: Sequence[int], relevant: set[int],
                baseline: Sequence[int], k: int) -> float:
    """Relevant hits that a popularity baseline would NOT have surfaced.

    This is the one metric that directly measures the thing a recommender is
    for. Being right about a video the user would have found anyway on the
    trending page adds no value.
    """
    top = list(recommended)[:k]
    obvious = set(int(i) for i in list(baseline)[:k])
    if not top:
        return 0.0
    return sum(1 for i in top if i in relevant and i not in obvious) / len(top)


# ---------------------------------------------------------------------------
# Grouped / listwise
# ---------------------------------------------------------------------------


def within_group_top1(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                      seed: int = 7) -> float:
    """Share of groups where the single positive is scored highest.

    Ties are broken RANDOMLY, deliberately. Training rows are laid out
    positive-first, so a naive ``argmax`` silently awards every tie to the
    positive -- which made a near-useless sparse feature appear to score 80%.
    Any metric over possibly-tied scores needs this.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    scores, labels, groups = scores[order], labels[order], groups[order]
    bounds = np.flatnonzero(np.diff(groups)) + 1
    hits = total = 0
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(groups)]):
        block_labels = labels[start:stop]
        if block_labels.sum() != 1:
            continue
        total += 1
        block_scores = scores[start:stop]
        best = np.flatnonzero(block_scores == block_scores.max())
        hits += int(block_labels[rng.choice(best)] == 1)
    return hits / max(total, 1)
