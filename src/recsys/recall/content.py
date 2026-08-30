"""Recall source: content-based retrieval over item text vectors.

Answers three questions, all with the same dot product:

    similar_items(i)     "more like this video"      -> watch page rail
    for_history(hist)    "more like what you watch"  -> personalised home
    search(text)         "videos matching this text" -> search + cold start

Strengths and weaknesses, stated plainly, because the hybrid is designed
around them:

  + Works on day-zero items with no interactions at all (no cold-start hole).
  + Every recommendation is explainable in one sentence.
  + Works for a brand-new user the moment they watch one thing.
  - It can only ever return more of the same. It cannot discover that people
    who watch Rust tutorials also watch mechanical keyboard reviews, because
    those share no vocabulary. That is the FILTER BUBBLE, and it is structural,
    not a tuning problem -- no amount of parameter fiddling fixes it.

Collaborative filtering (recsys.recall.cf) exists precisely to cover that gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..features.text import TextIndex

# Half-life in *positions back through history*, not wall-clock: what you
# watched most recently predicts your next click far better than what you
# watched a month ago. 8 means the 8th-most-recent watch counts half as much
# as the most recent one.
DEFAULT_HISTORY_HALFLIFE = 8.0


@dataclass
class RecallResult:
    """Candidate ids with scores and a per-source provenance tag.

    Provenance is not decoration -- it is what makes the "Why this video?"
    panel possible, and what lets us measure which recall source is actually
    earning its latency budget.
    """

    indices: np.ndarray
    scores: np.ndarray
    source: str

    def as_dict(self) -> dict[int, float]:
        return {int(i): float(s) for i, s in zip(self.indices, self.scores)}


def history_profile(
    text_index: TextIndex,
    history: Sequence[int],
    weights: Sequence[float] | None = None,
    halflife: float = DEFAULT_HISTORY_HALFLIFE,
) -> np.ndarray:
    """Collapse a watch history into a single 'taste' vector.

    profile = normalise( sum_i  w_i * decay_i * v_i )

    where ``w_i`` is engagement (watch fraction) and ``decay_i`` falls off
    exponentially with how far back the watch was.

    The averaging is the weakness worth understanding: a user who watches
    cooking AND Formula 1 gets a mean vector pointing *between* the two, which
    may match nothing at all. Production systems handle this by clustering the
    history into several profiles and retrieving for each. We approximate that
    cheaply in ``for_history`` by also retrieving around the single most recent
    item, so at least one coherent interest always survives the average.
    """
    if len(history) == 0:
        return np.zeros(text_index.dims, dtype=np.float32)

    idx = np.asarray(list(history), dtype=int)
    engagement = (np.ones(len(idx), dtype=np.float32) if weights is None
                  else np.asarray(list(weights), dtype=np.float32))
    # history is oldest -> newest, so reverse the position index for decay
    position_back = np.arange(len(idx) - 1, -1, -1, dtype=np.float32)
    decay = np.power(0.5, position_back / max(halflife, 1e-6))

    combined = (engagement * decay)[:, None] * text_index.vectors[idx]
    profile = combined.sum(axis=0)
    norm = np.linalg.norm(profile)
    return (profile / norm).astype(np.float32) if norm > 1e-9 else profile.astype(np.float32)


class ContentRecall:
    """Content-based candidate generation."""

    def __init__(self, text_index: TextIndex):
        self.index = text_index

    # -- item -> items ----------------------------------------------------
    def similar_items(self, item: int, k: int = 50,
                      exclude: Sequence[int] = ()) -> RecallResult:
        idx, scores = self.index.similar_to_vector(
            self.index.vectors[item], k=k, exclude=list(exclude) + [item]
        )
        return RecallResult(idx, scores, "content_similar")

    # -- history -> items -------------------------------------------------
    def for_history(self, history: Sequence[int], k: int = 50,
                    weights: Sequence[float] | None = None,
                    exclude: Sequence[int] = ()) -> RecallResult:
        """Retrieve for a user profile, with a recency anchor.

        Two retrievals are merged: one around the averaged profile (broad
        taste) and one around the most recent watch (current session intent).
        Without the anchor, a multi-interest user gets a meaningless centroid.
        """
        if len(history) == 0:
            return RecallResult(np.array([], dtype=int), np.array([]), "content_history")

        blocked = list(exclude) + list(history)
        profile = history_profile(self.index, history, weights)
        idx_p, sc_p = self.index.similar_to_vector(profile, k=k, exclude=blocked)

        recent = int(history[-1])
        idx_r, sc_r = self.index.similar_to_vector(
            self.index.vectors[recent], k=max(k // 3, 8), exclude=blocked
        )

        merged: dict[int, float] = {int(i): float(s) for i, s in zip(idx_p, sc_p)}
        for i, s in zip(idx_r, sc_r):
            # 0.85 keeps the session anchor slightly below the profile so it
            # nudges rather than dominates.
            merged[int(i)] = max(merged.get(int(i), 0.0), float(s) * 0.85)

        order = sorted(merged.items(), key=lambda kv: -kv[1])[:k]
        return RecallResult(
            np.array([i for i, _ in order], dtype=int),
            np.array([s for _, s in order], dtype=np.float32),
            "content_history",
        )

    # -- text -> items ----------------------------------------------------
    def search(self, query: str, k: int = 50,
               exclude: Sequence[int] = ()) -> RecallResult:
        """Retrieve by semantic similarity to the query.

        Returns an EMPTY result when the query matches no vocabulary term. That
        case is not hypothetical and it is not a rounding error: a TF-IDF query
        with no in-vocabulary token produces a vector of exact zeros, every
        similarity ties at 0.0, and the top-k is then decided by whatever order
        argpartition happened to leave the ties in. The output looks like a
        ranked list and carries no information at all.

        Returning nothing is the honest answer, and it lets the caller say so --
        see RecommendationEngine._recall, which falls back to trending and
        labels the response rather than silently presenting arbitrary videos as
        search results. Documented as F10 in docs/TEST_CASES.md.
        """
        vector = self.index.encode([query])[0]
        if not np.any(vector):
            return RecallResult(np.array([], dtype=int), np.array([]),
                                "content_search")
        idx, scores = self.index.similar_to_vector(vector, k=k, exclude=exclude)
        return RecallResult(idx, scores, "content_search")

    # -- explanation ------------------------------------------------------
    def nearest_reason(self, candidate: int, history: Sequence[int]) -> tuple[int, float]:
        """Which watched item best explains this candidate?

        Powers "Because you watched X". Returning the single closest history
        item is honest: it is literally the vector that pulled the candidate in.
        """
        if len(history) == 0:
            return -1, 0.0
        hist = np.asarray(list(history), dtype=int)
        sims = self.index.vectors[hist] @ self.index.vectors[candidate]
        best = int(np.argmax(sims))
        return int(hist[best]), float(sims[best])
