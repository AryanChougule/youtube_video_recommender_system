"""Stage 1 fusion: merge heterogeneous recall sources into one candidate set.

The problem
-----------
Content cosine lives in [0, 1]. Damped co-visitation counts live in [0, ~0.5]
with a long tail. ALS dot products are unbounded and correlate with item
popularity. Summing them is adding metres to kilograms.

Two ways out, and why we picked the second
------------------------------------------
*Score normalisation* (min-max / z-score per source) is simple but fragile: a
single outlier compresses everything else, and each source's distribution
shifts on every rebuild, so the weights need recalibrating every time.

*Reciprocal Rank Fusion* throws the scores away and keeps only ranks:

    RRF(d) = sum_s  w_s / (K + rank_s(d))

It is scale-free, needs no calibration, and survives distribution drift. The
``K`` term flattens the head of the curve, so rank 1 vs rank 2 is a nudge
rather than a cliff and no single over-confident source can steamroll the
fusion. It is the same technique hybrid search uses to combine BM25 with vector
retrieval.

What RRF costs us
-----------------
It discards score MAGNITUDE: "0.95 similar" and "0.51 similar" both count as
rank 1 if they lead their source. We get that information back in Stage 2 --
the ranker receives every raw per-source score as a feature. Fusion decides
what gets *considered*; the ranker decides what *wins*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .content import RecallResult


@dataclass
class CandidateSet:
    """Fused candidates with full provenance.

    ``source_ranks`` and ``source_scores`` are kept per item rather than
    collapsed, because they serve two jobs beyond fusion: they become ranker
    features, and they are what the "Why this video?" panel renders.
    """

    indices: np.ndarray                                   # (n_candidates,)
    fused_scores: np.ndarray                              # RRF score
    source_ranks: dict[str, dict[int, int]] = field(default_factory=dict)
    source_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    source_sizes: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    def sources_for(self, item: int) -> dict[str, int]:
        """Which sources proposed this item, and at what rank."""
        return {
            name: ranks[item]
            for name, ranks in self.source_ranks.items()
            if item in ranks
        }

    def score_from(self, source: str, item: int, default: float = 0.0) -> float:
        return self.source_scores.get(source, {}).get(item, default)


def reciprocal_rank_fusion(
    results: dict[str, RecallResult],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    max_candidates: int = 400,
) -> CandidateSet:
    """Fuse named recall results into a single ranked candidate set."""
    weights = weights or {}
    fused: dict[int, float] = {}
    source_ranks: dict[str, dict[int, int]] = {}
    source_scores: dict[str, dict[int, float]] = {}
    source_sizes: dict[str, int] = {}

    for name, result in results.items():
        if result is None or len(result.indices) == 0:
            source_sizes[name] = 0
            continue
        weight = float(weights.get(name, 1.0))
        ranks: dict[int, int] = {}
        scores: dict[int, float] = {}
        for rank, (item, score) in enumerate(zip(result.indices, result.scores)):
            item = int(item)
            ranks[item] = rank
            scores[item] = float(score)
            fused[item] = fused.get(item, 0.0) + weight / (rrf_k + rank + 1)
        source_ranks[name] = ranks
        source_scores[name] = scores
        source_sizes[name] = len(ranks)

    if not fused:
        return CandidateSet(np.array([], dtype=int), np.array([]),
                            source_ranks, source_scores, source_sizes)

    order = sorted(fused.items(), key=lambda kv: -kv[1])[:max_candidates]
    return CandidateSet(
        indices=np.array([i for i, _ in order], dtype=int),
        fused_scores=np.array([s for _, s in order], dtype=np.float32),
        source_ranks=source_ranks,
        source_scores=source_scores,
        source_sizes=source_sizes,
    )
