"""Offline evaluation harness: protocol, baselines, ablation.

Protocol
--------
One global temporal cutoff (``recsys.split``). For each evaluation user:

    history   = their clicks BEFORE the cutoff   -> what the model may see
    relevant  = their clicks AFTER  the cutoff   -> what it must predict

Graded relevance is the WATCH FRACTION of each held-out click, not a binary
flag, so NDCG rewards surfacing a video someone actually watched through over
one they opened and abandoned. Evaluating on clicks while training on watch
time would quietly undo the objective the whole system is built around.

Every model upstream of this file was fitted on pre-cutoff data only; see
``recsys.split`` for the leak that motivated it.

What is being measured, honestly
--------------------------------
These numbers describe performance on SIMULATED interactions. They demonstrate
that the algorithms recover the latent structure that generated the data. They
are NOT a prediction of real-world YouTube performance, and no offline metric
ever is -- the only way to know is an online A/B test. See docs/EVALUATION.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .artifacts import Artifacts
from .metrics import (
    average_precision_at_k,
    catalog_coverage,
    gini_coefficient,
    hit_rate,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    serendipity,
)
from .split import TemporalSplit


@dataclass
class EvalUser:
    user_id: str
    history: list[int]
    weights: list[float]
    relevant: set[int]
    relevance: dict[int, float]          # graded: item -> watch fraction


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def build_eval_users(
    interactions: pd.DataFrame,
    split: TemporalSplit,
    video_index: dict[str, int],
    min_history: int = 3,
    min_holdout: int = 1,
    max_users: int = 1500,
    seed: int = 42,
) -> list[EvalUser]:
    """Users with enough signal on both sides of the cutoff to be scorable."""
    clicks = interactions[interactions["clicked"] == 1].copy()
    clicks["item"] = clicks["video_id"].map(video_index)
    clicks = clicks[clicks["item"].notna()]
    clicks["item"] = clicks["item"].astype(int)
    clicks = clicks.sort_values("ts")

    is_train = split.train_mask(clicks["ts"])
    train_clicks = clicks[is_train]
    test_clicks = clicks[~is_train]

    train_by_user = {u: g for u, g in train_clicks.groupby("user_id", sort=False)}
    out: list[EvalUser] = []
    for user, group in test_clicks.groupby("user_id", sort=False):
        history_rows = train_by_user.get(user)
        if history_rows is None or len(history_rows) < min_history:
            continue
        # Held-out items the user already watched BEFORE the cutoff are not a
        # prediction task -- they are a rewatch, and the engine deliberately
        # excludes history. Counting them would penalise correct behaviour.
        seen = set(history_rows["item"].tolist())
        graded = {int(r.item): float(r.watch_fraction)
                  for r in group.itertuples() if int(r.item) not in seen}
        if len(graded) < min_holdout:
            continue
        out.append(EvalUser(
            user_id=str(user),
            history=history_rows["item"].tolist(),
            weights=history_rows["watch_fraction"].tolist(),
            relevant=set(graded),
            relevance=graded,
        ))

    rng = np.random.default_rng(seed)
    if len(out) > max_users:
        picks = rng.choice(len(out), size=max_users, replace=False)
        out = [out[i] for i in sorted(picks)]
    return out


def evaluate_strategy(
    name: str,
    recommend: Callable[[EvalUser, int], Sequence[int]],
    users: Sequence[EvalUser],
    artifacts: Artifacts,
    k_values: Sequence[int] = (5, 10, 20),
    popularity_baseline: Sequence[int] | None = None,
    notes: str = "",
) -> EvalResult:
    k_max = max(k_values)
    accumulators: dict[str, list[float]] = {}
    all_lists: list[Sequence[int]] = []
    exposure = np.zeros(artifacts.n_items, dtype=np.float64)
    latencies: list[float] = []
    vectors = artifacts.text_index.vectors
    pop = np.maximum(artifacts.covisitation.popularity, 0.0) + 1.0
    baseline = list(popularity_baseline or [])

    def add(key: str, value: float) -> None:
        accumulators.setdefault(key, []).append(value)

    for user in users:
        t0 = time.perf_counter()
        recommended = list(recommend(user, k_max))
        latencies.append((time.perf_counter() - t0) * 1000)

        all_lists.append(recommended)
        for item in recommended:
            exposure[int(item)] += 1

        for k in k_values:
            add(f"precision@{k}", precision_at_k(recommended, user.relevant, k))
            add(f"recall@{k}", recall_at_k(recommended, user.relevant, k))
            add(f"ndcg@{k}", ndcg_at_k(recommended, user.relevance, k))
            add(f"map@{k}", average_precision_at_k(recommended, user.relevant, k))
            add(f"hit_rate@{k}", hit_rate(recommended, user.relevant, k))
            if baseline:
                add(f"serendipity@{k}",
                    serendipity(recommended, user.relevant, baseline, k))
        add("mrr", reciprocal_rank(recommended, user.relevant, k_max))
        add("intra_list_diversity", intra_list_diversity(recommended[:10], vectors))
        add("novelty_bits", novelty(recommended[:10], pop))

    metrics = {key: round(float(np.mean(values)), 4)
               for key, values in sorted(accumulators.items())}
    metrics["coverage"] = round(catalog_coverage(all_lists, artifacts.n_items), 4)
    metrics["gini_exposure"] = round(gini_coefficient(exposure), 4)
    metrics["n_users"] = len(users)

    return EvalResult(
        name=name, metrics=metrics,
        latency_ms={
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "mean": round(float(np.mean(latencies)), 2),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------


def make_baselines(artifacts: Artifacts, seed: int = 42) -> dict[str, Callable]:
    """Baselines the hybrid must beat to justify its complexity.

    ``popularity`` is the one that matters. It is trivial, it is what a naive
    system does, and on skewed catalogs it is embarrassingly hard to beat --
    any recommender that cannot clear it is not earning its infrastructure.
    """
    rng = np.random.default_rng(seed)
    n_items = artifacts.n_items
    stats = artifacts.item_stats
    pop_order = np.argsort(-stats["log_views"].to_numpy())
    trend_order = np.argsort(-(np.log1p(stats["views_per_day"].to_numpy())
                               * np.power(0.5, stats["age_days"].to_numpy() / 21.0)))

    def random_rec(user: EvalUser, k: int) -> list[int]:
        seen = set(user.history)
        picks = rng.choice(n_items, size=min(k * 3, n_items), replace=False)
        return [int(i) for i in picks if int(i) not in seen][:k]

    def top_of(order: np.ndarray):
        def strategy(user: EvalUser, k: int) -> list[int]:
            seen = set(user.history)
            return [int(i) for i in order if int(i) not in seen][:k]
        return strategy

    def content_only(user: EvalUser, k: int) -> list[int]:
        result = artifacts.content.for_history(
            user.history, k=k, weights=user.weights, exclude=user.history)
        return [int(i) for i in result.indices][:k]

    def covisit_only(user: EvalUser, k: int) -> list[int]:
        result = artifacts.covisitation.for_history(
            user.history, k=k, weights=user.weights, exclude=user.history)
        return [int(i) for i in result.indices][:k]

    def als_only(user: EvalUser, k: int) -> list[int]:
        vector = artifacts.als.fold_in(user.history, user.weights)
        result = artifacts.als.recommend(vector, k=k, exclude=user.history)
        return [int(i) for i in result.indices][:k]

    return {
        "random": random_rec,
        "popularity": top_of(pop_order),
        "trending": top_of(trend_order),
        "content_only": content_only,
        "cf_covisit_only": covisit_only,
        "cf_als_only": als_only,
    }
