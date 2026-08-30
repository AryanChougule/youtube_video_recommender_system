"""Stage 2 feature engineering -- used by BOTH training and serving.

This module exists as a single shared class for one reason: **train/serve
skew**. The most common way a recommender looks excellent offline and fails in
production is that the training pipeline and the serving pipeline compute
"the same" feature slightly differently. Sharing the code makes that class of
bug structurally impossible rather than merely unlikely.

Second design rule: every feature here is computable from a
``(history, candidate)`` pair alone. Nothing depends on having executed Stage 1
retrieval. That keeps training cheap (no need to replay candidate generation
for 300k rows) and, more importantly, means the ranker never sees a feature at
training time that serving cannot reproduce.

Consequence worth stating: we deliberately do NOT use candidate rank or the
fused RRF score as features, even though they are informative, because they
depend on which other candidates happened to be retrieved. Raw per-source
scores carry the same information without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

FEATURE_NAMES: list[str] = [
    # --- match between the user and this candidate (the important ones) ---
    "content_sim_profile",     # cosine to the averaged taste vector
    "content_sim_recent",      # cosine to the most recent watch (session intent)
    "content_sim_max_hist",    # max cosine to ANY single watched item
    "covisit_score",           # co-visitation mass from the whole history
    "als_score",               # latent-factor affinity
    "channel_affinity",        # engagement-weighted watches of this channel
    "channel_seen",            # has the user ever watched this channel
    "category_affinity",       # share of history in this candidate's category
    "duration_fit",            # -|log(candidate duration / user's typical)|
    # --- candidate-only features -----------------------------------------
    "log_views",
    "log_views_per_day",
    "engagement_rate",
    "like_rate",
    "log_age_days",
    "log_duration_min",
    "is_short",
    "n_tags",
    "title_length",
    # --- RELATIVE user features -------------------------------------------
    # Absolute user-only features (history length, distinct categories, mean
    # watch fraction) were removed after measurement. They are CONSTANT within
    # a feed, so they cannot help rank one candidate above another, yet they
    # drift ~0.9 sigma between the train and test periods as histories grow --
    # so the tree splits learned on them simply do not transfer. Expressing the
    # user side as a RELATIVE quantity keeps the signal (does this person watch
    # more mainstream things than this candidate?) while varying within a feed.
    "views_vs_user",           # log_views(item) - user's mean log_views
]


@dataclass
class UserContext:
    """Everything about a user needed to score any candidate.

    Computed once per request (or per training feed) and reused across all
    candidates -- the expensive parts (profile vector, ALS fold-in) must not be
    recomputed per item.
    """

    history: np.ndarray
    weights: np.ndarray
    profile_vector: np.ndarray
    als_vector: np.ndarray
    recent_item: int
    channel_affinity: dict[str, float] = field(default_factory=dict)
    category_affinity: dict[str, float] = field(default_factory=dict)
    # Dense (n_items,) rather than a dict. The dict version cost 5ms per call
    # because pooling meant a Python triple-loop over history x neighbours;
    # np.bincount does the same accumulation in one vectorised pass.
    covisit_pool: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    median_duration: float = 10.0
    mean_watch_fraction: float = 0.0
    mean_log_views: float = 0.0
    n_categories: int = 0

    @property
    def is_cold(self) -> bool:
        return len(self.history) == 0


class FeatureBuilder:
    """Builds the Stage 2 feature matrix. One instance, shared by both paths."""

    def __init__(
        self,
        catalog,
        item_stats,
        item_vectors: np.ndarray,
        covisitation=None,
        als=None,
    ):
        self.item_vectors = item_vectors
        self.covisitation = covisitation
        self.als = als
        self.n_items = len(catalog)

        self.channel_of = catalog["channel_id"].to_numpy()
        self.category_of = catalog["category"].to_numpy()

        self.log_views = item_stats["log_views"].to_numpy(dtype=np.float32)
        self.log_views_per_day = item_stats["log_views_per_day"].to_numpy(dtype=np.float32)
        self.engagement_rate = item_stats["engagement_rate"].to_numpy(dtype=np.float32)
        self.like_rate = item_stats["like_rate"].to_numpy(dtype=np.float32)
        self.log_age_days = np.log1p(item_stats["age_days"].to_numpy(dtype=np.float32))
        self.duration_min = item_stats["duration_minutes"].to_numpy(dtype=np.float32)
        self.log_duration_min = np.log1p(self.duration_min)
        self.is_short = item_stats["is_short"].to_numpy(dtype=np.float32)
        self.n_tags = item_stats["n_tags"].to_numpy(dtype=np.float32)
        self.title_length = item_stats["title_length"].to_numpy(dtype=np.float32)

    # -- context ----------------------------------------------------------
    def build_context(
        self, history: Sequence[int], weights: Sequence[float] | None = None,
        covisit_depth: int = 60,
    ) -> UserContext:
        hist = np.asarray(list(history), dtype=int)
        if weights is None:
            w = np.ones(len(hist), dtype=np.float32)
        else:
            w = np.asarray(list(weights), dtype=np.float32)

        dims = self.item_vectors.shape[1]
        if len(hist) == 0:
            als_dims = self.als.factors if self.als is not None else 1
            return UserContext(
                history=hist, weights=w,
                profile_vector=np.zeros(dims, dtype=np.float32),
                als_vector=np.zeros(als_dims), recent_item=-1,
                covisit_pool=np.zeros(self.n_items, dtype=np.float32),
            )

        # Recency decay: the 8th-most-recent watch counts half as much as the
        # most recent one. Recency dominates next-click prediction.
        position_back = np.arange(len(hist) - 1, -1, -1, dtype=np.float32)
        decay = np.power(0.5, position_back / 8.0)
        combined = w * decay

        profile = (combined[:, None] * self.item_vectors[hist]).sum(axis=0)
        norm = np.linalg.norm(profile)
        profile = (profile / norm).astype(np.float32) if norm > 1e-9 else profile.astype(np.float32)

        als_vector = (self.als.fold_in(hist, w) if self.als is not None
                      else np.zeros(1))

        channel_aff: dict[str, float] = {}
        category_aff: dict[str, float] = {}
        for h, weight in zip(hist, combined):
            channel_aff[self.channel_of[h]] = channel_aff.get(self.channel_of[h], 0.0) + float(weight)
            category_aff[self.category_of[h]] = category_aff.get(self.category_of[h], 0.0) + float(weight)
        total_cat = sum(category_aff.values()) or 1.0
        category_aff = {k: v / total_cat for k, v in category_aff.items()}

        # Pool co-visitation mass across the history into a dense vector.
        # Padded neighbour slots carry score 0, so they contribute nothing and
        # need no masking. One bincount replaces a history x neighbours loop.
        covisit_pool = np.zeros(self.n_items, dtype=np.float32)
        if self.covisitation is not None:
            depth = min(covisit_depth, self.covisitation.neighbours.shape[1])
            neigh = self.covisitation.neighbours[hist][:, :depth]
            sc = self.covisitation.scores[hist][:, :depth]
            contribution = (sc * combined[:, None]).ravel()
            covisit_pool = np.bincount(
                neigh.ravel(), weights=contribution, minlength=self.n_items
            ).astype(np.float32)

        return UserContext(
            history=hist, weights=w, profile_vector=profile, als_vector=als_vector,
            recent_item=int(hist[-1]),
            channel_affinity=channel_aff, category_affinity=category_aff,
            covisit_pool=covisit_pool,
            median_duration=float(np.median(self.duration_min[hist])),
            mean_watch_fraction=float(w.mean()),
            mean_log_views=float(self.log_views[hist].mean()),
            n_categories=len(category_aff),
        )

    # -- features ---------------------------------------------------------
    def build(self, ctx: UserContext, candidates: Sequence[int]) -> np.ndarray:
        """Feature matrix for one user context against many candidates."""
        idx = np.asarray(list(candidates), dtype=int)
        n = len(idx)
        if n == 0:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        if ctx.is_cold:
            sim_profile = np.zeros(n, dtype=np.float32)
            sim_recent = np.zeros(n, dtype=np.float32)
            sim_max = np.zeros(n, dtype=np.float32)
            als_score = np.zeros(n, dtype=np.float32)
        else:
            sim_profile = (self.item_vectors[idx] @ ctx.profile_vector).astype(np.float32)
            sim_recent = (self.item_vectors[idx] @ self.item_vectors[ctx.recent_item]).astype(np.float32)
            # Max-over-history rather than mean. A user who watches cooking AND
            # Formula 1 has a profile centroid pointing between the two, which
            # matches neither; the max recovers "is this close to ANY single
            # thing they watched" and is invariant to history length.
            sim_max = (self.item_vectors[idx] @ self.item_vectors[ctx.history].T).max(axis=1).astype(np.float32)
            als_score = (
                (self.als.item_factors[idx] @ ctx.als_vector).astype(np.float32)
                if self.als is not None else np.zeros(n, dtype=np.float32)
            )

        covisit = (ctx.covisit_pool[idx] if len(ctx.covisit_pool)
                   else np.zeros(n, dtype=np.float32))
        chan = np.array([ctx.channel_affinity.get(self.channel_of[i], 0.0) for i in idx], dtype=np.float32)
        cat = np.array([ctx.category_affinity.get(self.category_of[i], 0.0) for i in idx], dtype=np.float32)

        # Symmetric in log space: a 2x-too-long and a 2x-too-short video are
        # equally mismatched, which a raw difference would not capture.
        pref = max(ctx.median_duration, 0.5)
        duration_fit = (-np.abs(np.log(np.maximum(self.duration_min[idx], 0.1) / pref))).astype(np.float32)

        columns = [
            sim_profile, sim_recent, sim_max, covisit, als_score, chan,
            (chan > 0).astype(np.float32), cat, duration_fit,
            self.log_views[idx], self.log_views_per_day[idx],
            self.engagement_rate[idx], self.like_rate[idx],
            self.log_age_days[idx], self.log_duration_min[idx],
            self.is_short[idx], self.n_tags[idx], self.title_length[idx],
            (self.log_views[idx] - np.float32(ctx.mean_log_views)),
        ]
        matrix = np.stack(columns, axis=1).astype(np.float32)
        assert matrix.shape[1] == len(FEATURE_NAMES), (
            f"feature count drift: built {matrix.shape[1]}, "
            f"declared {len(FEATURE_NAMES)}"
        )
        return matrix
