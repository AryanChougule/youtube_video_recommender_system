"""Evaluation protocols that survive the logging-policy problem.

Why the obvious protocol is broken
----------------------------------
The standard offline recipe -- hide each user's future clicks, ask the model to
retrieve them from the full catalog, report NDCG -- silently measures the
LOGGING POLICY rather than the recommender.

A user can only click what they were shown. Our logging policy is popularity-
heavy, so held-out clicks concentrate on popular videos, and a trivial
popularity ranker scores well by reproducing the old policy. A genuinely better
recommender is *penalised* for surfacing something the user never got the
chance to click.

We proved this rather than asserting it. Scoring an ORACLE -- the exact
generative model that produced the data, with true user personas and true
hidden video quality -- on full-catalog retrieval:

    popularity   NDCG@10 = 0.0178
    ORACLE       NDCG@10 = 0.0169     <- the data-generating process LOSES

If the true model cannot win, the metric is not measuring model quality. Any
conclusion drawn from full-catalog NDCG on logged data is unsafe.

Two protocols that are safe
---------------------------
**A. Re-ranking logged impressions.** For each feed in the test period, re-order
the exact items that were actually shown. Every item has a genuine, observed
label, so no counterfactual guess is required. This answers the real product
question -- "given the same page, would we have put the right video on top?" --
and it is the offline protocol that best predicts online lift.

**B. Sampled candidate sets.** Rank one held-out positive against N uniformly
sampled catalog negatives. This isolates pure ranking ability from the "find a
needle the old policy hid" retrieval problem. Sampled metrics have their own
known biases (Krichene & Rendle, KDD 2020) -- they systematically flatter weak
models -- so we report them alongside A, never instead of it.

Neither replaces an online A/B test. Nothing offline does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .artifacts import Artifacts
from .metrics import dcg
from .split import TemporalSplit

Scorer = Callable[[Sequence[int], Sequence[float], np.ndarray], np.ndarray]


@dataclass
class TestFeed:
    history: list[int]
    weights: list[float]
    items: np.ndarray            # exactly what was shown, in logged order
    labels: np.ndarray           # 1 for the clicked item
    gains: np.ndarray            # watch fraction (graded relevance)
    logged_ranks: np.ndarray     # the position the old policy assigned


def replay_test_feeds(
    interactions: pd.DataFrame,
    split: TemporalSplit,
    video_index: dict[str, int],
    min_history: int = 3,
    max_feeds: int = 20000,
    seed: int = 42,
) -> list[TestFeed]:
    """Walk every user's timeline and emit post-cutoff feeds with causal history."""
    df = interactions.copy()
    df["item"] = df["video_id"].map(video_index)
    df = df[df["item"].notna()]
    df = df.sort_values(["user_id", "ts", "rank_shown"], kind="stable")

    user_codes = pd.factorize(df["user_id"], sort=False)[0]
    ts_ns = df["ts"].astype("int64").to_numpy()
    item = df["item"].to_numpy(dtype=np.int64)
    clicked = df["clicked"].to_numpy(dtype=np.int8)
    rank_shown = df["rank_shown"].to_numpy(dtype=np.int64)
    watch_fraction = df["watch_fraction"].to_numpy(dtype=np.float64)

    new_feed = np.empty(len(df), dtype=bool)
    new_feed[0] = True
    new_feed[1:] = (user_codes[1:] != user_codes[:-1]) | (ts_ns[1:] != ts_ns[:-1])
    starts = np.flatnonzero(new_feed)
    stops = np.append(starts[1:], len(df))

    feeds: list[TestFeed] = []
    history: list[int] = []
    weights: list[float] = []
    current_user = -1

    for start, stop in zip(starts, stops):
        if user_codes[start] != current_user:
            current_user = user_codes[start]
            history, weights = [], []

        block = slice(start, stop)
        is_test = ts_ns[start] > split.cutoff_ns
        has_click = clicked[block].sum() == 1

        if is_test and has_click and len(history) >= min_history:
            feeds.append(TestFeed(
                history=list(history), weights=list(weights),
                items=item[block].copy(), labels=clicked[block].astype(int).copy(),
                gains=watch_fraction[block].copy(), logged_ranks=rank_shown[block].copy(),
            ))

        for row in range(start, stop):
            if clicked[row]:
                history.append(int(item[row]))
                weights.append(float(watch_fraction[row]))

    rng = np.random.default_rng(seed)
    if len(feeds) > max_feeds:
        picks = rng.choice(len(feeds), size=max_feeds, replace=False)
        feeds = [feeds[i] for i in sorted(picks)]
    return feeds


# ---------------------------------------------------------------------------
# Scorers -- one uniform interface so every model is compared identically
# ---------------------------------------------------------------------------


def make_scorers(art: Artifacts, include_oracle: bool = False,
                 gt_user_topics: np.ndarray | None = None,
                 gt_item_topics: np.ndarray | None = None,
                 users_df: pd.DataFrame | None = None) -> dict[str, Scorer]:
    stats = art.item_stats
    log_views = stats["log_views"].to_numpy(dtype=np.float64)
    vectors = art.text_index.vectors
    features = art.features

    def popularity(history, weights, cands):
        return log_views[cands]

    def content(history, weights, cands):
        if not len(history):
            return np.zeros(len(cands))
        ctx = features.build_context(history, weights)
        return vectors[cands] @ ctx.profile_vector

    def als(history, weights, cands):
        if not len(history):
            return np.zeros(len(cands))
        return art.als.item_factors[cands] @ art.als.fold_in(history, weights)

    def covisit(history, weights, cands):
        if not len(history):
            return np.zeros(len(cands))
        return features.build_context(history, weights).covisit_pool[cands]

    def ranker(history, weights, cands):
        ctx = features.build_context(history, weights)
        return art.ranker.score(features.build(ctx, cands))

    scorers: dict[str, Scorer] = {
        "popularity": popularity,
        "content_only": content,
        "cf_als_only": als,
        "cf_covisit_only": covisit,
    }
    if art.ranker is not None:
        scorers["LEARNED RANKER"] = ranker

    if include_oracle and gt_user_topics is not None:
        from .data.simulator import (
            CLICK_BIAS, W_AFFINITY, W_DURATION_FIT, W_POPULARITY, W_QUALITY,
        )
        z = lambda x: (x - x.mean()) / (x.std() or 1.0)  # noqa: E731
        lv = z(np.log1p(art.catalog["view_count"].to_numpy(float)))
        qz = z(np.log(art.catalog["latent_quality"].to_numpy(float)))
        dur = art.catalog["duration_seconds"].to_numpy(float) / 60.0
        mainstream = users_df["mainstream"].to_numpy()
        pref = users_df["preferred_minutes"].to_numpy()

        def make_oracle(user_row: int) -> Scorer:
            def oracle(history, weights, cands):
                affinity = gt_item_topics[cands] @ gt_user_topics[user_row]
                return (CLICK_BIAS + W_AFFINITY * affinity * 12.0
                        + W_POPULARITY * mainstream[user_row] * 2.0 * lv[cands]
                        + W_QUALITY * qz[cands]
                        + W_DURATION_FIT * (-np.abs(np.log(dur[cands] / pref[user_row]))))
            return oracle

        scorers["_oracle_factory"] = make_oracle    # handled specially by callers
    return scorers


# ---------------------------------------------------------------------------
# Protocol A: re-rank logged impressions
# ---------------------------------------------------------------------------


def evaluate_reranking(scorers: dict[str, Scorer], feeds: Sequence[TestFeed],
                       seed: int = 7) -> dict[str, dict[str, float]]:
    """Re-order the items actually shown. Counterfactually valid by construction."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}

    # Reported as a reference, NOT as a baseline to beat. Slot position is a
    # CAUSE of the click under a cascade click model, not evidence about video
    # quality, so a scorer that knows the position has access to the label's
    # mechanism. Its score is the size of the position effect -- the ceiling on
    # what re-ranking could ever be worth -- and comparing a content model
    # against it would be a category error.
    def logged(feed: TestFeed) -> np.ndarray:
        return -feed.logged_ranks.astype(float)

    named: dict[str, Callable[[TestFeed], np.ndarray]] = {
        "[ref] shown position": logged,
        "random": lambda f: rng.random(len(f.items)),
    }
    for name, scorer in scorers.items():
        if name.startswith("_"):
            continue
        named[name] = (lambda s: lambda f: np.asarray(
            s(f.history, f.weights, f.items), dtype=float))(scorer)

    for name, fn in named.items():
        top1, ndcg, mrr = [], [], []
        for feed in feeds:
            scores = fn(feed)
            order = np.lexsort((rng.random(len(scores)), -scores))
            top1.append(float(feed.labels[order[0]] == 1))
            gains = feed.gains * feed.labels          # only the click carries gain
            actual = dcg(gains[order])
            ideal = dcg(sorted(gains, reverse=True))
            ndcg.append(actual / ideal if ideal > 0 else 0.0)
            position = int(np.flatnonzero(feed.labels[order] == 1)[0]) + 1
            mrr.append(1.0 / position)
        out[name] = {
            "top1": round(float(np.mean(top1)), 4),
            "ndcg": round(float(np.mean(ndcg)), 4),
            "mrr": round(float(np.mean(mrr)), 4),
            "n_feeds": len(feeds),
        }
    return out


# ---------------------------------------------------------------------------
# Protocol B: sampled candidate sets
# ---------------------------------------------------------------------------


def evaluate_sampled(scorers: dict[str, Scorer], users, art: Artifacts,
                     n_negatives: int = 100, k: int = 10,
                     seed: int = 42) -> dict[str, dict[str, float]]:
    """Rank one held-out positive against N sampled catalog negatives."""
    rng = np.random.default_rng(seed)
    n_items = art.n_items
    out: dict[str, dict[str, float]] = {}

    tasks: list[tuple] = []
    for user in users:
        seen = set(user.history) | set(user.relevant)
        for positive in list(user.relevant)[:3]:      # cap heavy users
            negatives = []
            while len(negatives) < n_negatives:
                draw = rng.integers(0, n_items, size=n_negatives)
                negatives += [int(d) for d in draw if int(d) not in seen]
            candidates = np.array([positive] + negatives[:n_negatives], dtype=int)
            tasks.append((user, candidates))

    scorers = dict(scorers)
    scorers["random"] = lambda h, w, c: rng.random(len(c))

    for name, scorer in scorers.items():
        if name.startswith("_"):
            continue
        hits, ndcgs, ranks = [], [], []
        for user, candidates in tasks:
            scores = np.asarray(scorer(user.history, user.weights, candidates), dtype=float)
            order = np.lexsort((rng.random(len(scores)), -scores))
            position = int(np.flatnonzero(order == 0)[0]) + 1   # positive is index 0
            hits.append(float(position <= k))
            ndcgs.append(1.0 / np.log2(position + 1) if position <= k else 0.0)
            ranks.append(position)
        out[name] = {
            f"hit_rate@{k}": round(float(np.mean(hits)), 4),
            f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
            "mean_rank": round(float(np.mean(ranks)), 2),
            "n_tasks": len(tasks),
        }
    return out
