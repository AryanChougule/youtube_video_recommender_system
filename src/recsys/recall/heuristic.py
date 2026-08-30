"""Non-learned recall sources: trending/popular and channel affinity.

These are the "knowledge-based" leg of the hybrid. They learn nothing, which
is exactly why they are indispensable:

* they are the ONLY sources that work for a user with zero history,
* they are the only ones that can surface a video uploaded an hour ago,
* and they encode product strategy directly (freshness matters on YouTube in a
  way it does not on Netflix, because the catalog turns over hourly).

A learned model cannot express "prefer things published recently" unless you
feed it recency, and even then it will only reproduce whatever recency bias was
in the training log. Some things belong in policy, not in the loss.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .content import RecallResult


class TrendingRecall:
    """Popularity- and freshness-based recall.

    ``trending`` uses a time-decayed velocity score rather than raw views:

        score = views_per_day * 0.5 ** (age_days / halflife)

    Raw view count would return the same all-time-greatest videos forever --
    it measures accumulated history, not current interest. Velocity plus decay
    is the same shape as Hacker News' gravity ranking, and it is what makes
    "Trending" mean something different from "Most viewed".
    """

    def __init__(self, item_stats, halflife_days: float = 21.0):
        self.age_days = item_stats["age_days"].to_numpy(dtype=np.float64)
        self.views_per_day = item_stats["views_per_day"].to_numpy(dtype=np.float64)
        self.log_views = item_stats["log_views"].to_numpy(dtype=np.float64)
        self.engagement = item_stats["engagement_rate"].to_numpy(dtype=np.float64)

        decay = np.power(0.5, self.age_days / max(halflife_days, 1e-6))
        self._trending = np.log1p(self.views_per_day) * decay
        # A quality tilt stops "trending" from becoming "most clickbaited".
        self._trending *= (1.0 + 2.0 * self.engagement)
        self._popular = self.log_views * (1.0 + 1.5 * self.engagement)

    def _top(self, scores: np.ndarray, k: int, exclude: Sequence[int],
             source: str) -> RecallResult:
        scores = scores.copy()
        if len(exclude):
            scores[np.asarray(list(exclude), dtype=int)] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return RecallResult(np.array([], dtype=int), np.array([]), source)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return RecallResult(top.astype(int), scores[top].astype(np.float32), source)

    def trending(self, k: int = 60, exclude: Sequence[int] = ()) -> RecallResult:
        return self._top(self._trending, k, exclude, "trending")

    def popular(self, k: int = 60, exclude: Sequence[int] = ()) -> RecallResult:
        return self._top(self._popular, k, exclude, "popular")


class ChannelRecall:
    """More from channels the user already watches.

    Weak as a standalone recommender -- it is definitionally incapable of
    discovery -- but it is a strong *feature* and a strong minor source,
    because channel loyalty is one of the most reliable behavioural signals on
    YouTube. Subscribing is an explicit, durable statement of intent in a way a
    single click never is.

    Note the deliberate tension with the diversity policy in Module 6: this
    source pushes same-channel items UP, and ``max_per_channel`` pushes them
    back DOWN. That is not a contradiction -- one is about what to consider,
    the other about what a good page looks like.
    """

    def __init__(self, catalog, item_stats):
        self.channel_of = catalog["channel_id"].to_numpy()
        self.quality = (
            item_stats["log_views"].to_numpy(dtype=np.float64)
            * (1.0 + 1.5 * item_stats["engagement_rate"].to_numpy(dtype=np.float64))
        )
        # channel_id -> item row indices. CatalogView precomputes this; a
        # DataFrame is still accepted so build-time callers keep working.
        if hasattr(catalog, "by_channel"):
            self.by_channel = catalog.by_channel
        else:
            self.by_channel = {
                ch: grp.to_numpy()
                for ch, grp in catalog.reset_index().groupby("channel_id")["index"]
            }

    def for_history(self, history: Sequence[int], k: int = 60,
                    weights: Sequence[float] | None = None,
                    exclude: Sequence[int] = ()) -> RecallResult:
        if len(history) == 0:
            return RecallResult(np.array([], dtype=int), np.array([]), "channel")

        hist = np.asarray(list(history), dtype=int)
        engagement = (np.ones(len(hist)) if weights is None
                      else np.asarray(list(weights), dtype=float))

        # Affinity accumulates: three watches from one channel beats one.
        affinity: dict[str, float] = {}
        for h, w in zip(hist, engagement):
            ch = self.channel_of[h]
            affinity[ch] = affinity.get(ch, 0.0) + float(w)

        blocked = set(int(x) for x in list(exclude) + list(hist))
        pooled: dict[int, float] = {}
        for ch, aff in affinity.items():
            for item in self.by_channel.get(ch, ()):
                item = int(item)
                if item in blocked:
                    continue
                # log1p keeps a mega-popular back-catalogue video from
                # dominating purely on view count.
                pooled[item] = aff * (1.0 + 0.15 * float(np.log1p(self.quality[item])))

        if not pooled:
            return RecallResult(np.array([], dtype=int), np.array([]), "channel")
        order = sorted(pooled.items(), key=lambda kv: -kv[1])[:k]
        return RecallResult(
            np.array([i for i, _ in order], dtype=int),
            np.array([s for _, s in order], dtype=np.float32),
            "channel",
        )
