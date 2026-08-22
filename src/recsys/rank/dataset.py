"""Build the Stage 2 training set by replaying each user's timeline forward.

Two properties are non-negotiable here, and both are easy to get wrong:

**Causality.** For a training row logged at time t, the user's history must
contain only clicks strictly before t. Using the full history leaks the future
into the features; offline metrics then look spectacular and mean nothing. We
guarantee it structurally by replaying the log in timestamp order and only
appending to the history *after* a feed has been featurised.

**Mixed negatives.** Two kinds, deliberately:

* *In-feed* negatives are videos the user genuinely saw and genuinely skipped.
  They are hard negatives -- the old policy already judged them plausible -- so
  they teach fine discrimination.
* *Random catalog* negatives cover the regime that actually dominates at
  serving time. The engine scores all 6,000 items, and most of them look
  nothing like a logged impression; a model trained only on in-feed negatives
  is out of distribution there and mis-calibrates badly. (Yi et al., "Mixed
  Negative Sampling for Learning Two-tower Neural Networks", 2019.)

Using only random negatives -- the common shortcut -- would be worse than
either: it trains the model to separate "plausible" from "arbitrary", a far
easier and far less useful task.

Position bias is corrected with Inverse Propensity Scoring. A click at rank 6
is much stronger evidence than a click at rank 0, because the user had to scan
past five other things to get there; IPS says so by weighting each click by
1 / P(examined at that rank).

Implementation note: the log is walked as sorted NumPy arrays with explicit
feed boundaries rather than a nested pandas ``groupby``. Grouping 1.1M rows
into 137k tiny frames costs more in pandas overhead than all the feature
arithmetic combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, FeatureBuilder


#: Multi-task heads. Each is a BINARY label over the same rows and the same
#: feature matrix, so one pass builds them all and each head's output stays a
#: calibrated probability that can be combined with meaningful weights.
#:
#: They are deliberately nested (a completion implies a long watch implies a
#: click), which is what lets the value score express "a click is worth
#: something, a long watch more, a satisfying watch most".
TASK_LABELS: dict[str, str] = {
    "click": "clicked",
    "long_watch": "watch_fraction >= 0.5",
    "completion": "watch_fraction >= 0.9",
    "liked": "liked",
    "satisfied": "satisfied",
    "dismissed": "dismissed",
}


@dataclass
class TrainingSet:
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    group: np.ndarray          # feed id, for grouped/listwise evaluation
    timestamp: np.ndarray      # for temporal splitting
    meta: dict
    #: name -> binary label vector, aligned with X. Empty for older callers.
    task_y: dict = None


def build_training_set(
    interactions: pd.DataFrame,
    feature_builder: FeatureBuilder,
    video_ids: list[str],
    negatives_per_positive: int = 4,
    random_negatives_per_positive: int = 2,
    objective: str = "watch_time",
    position_decay: float = 0.82,
    ips_clip: float = 0.05,
    no_click_feed_rate: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> TrainingSet:
    rng = np.random.default_rng(seed)
    item_index = {v: i for i, v in enumerate(video_ids)}

    df = interactions.copy()
    df["item"] = df["video_id"].map(item_index)
    df = df[df["item"].notna()]
    df = df.sort_values(["user_id", "ts", "rank_shown"], kind="stable")

    # --- flatten to raw arrays; everything below is index arithmetic -------
    user_codes = pd.factorize(df["user_id"], sort=False)[0]
    ts_ns = df["ts"].astype("int64").to_numpy()
    item = df["item"].to_numpy(dtype=np.int64)
    clicked = df["clicked"].to_numpy(dtype=np.int8)
    rank_shown = df["rank_shown"].to_numpy(dtype=np.float64)
    watch_seconds = df["watch_seconds"].to_numpy(dtype=np.float64)
    watch_fraction = df["watch_fraction"].to_numpy(dtype=np.float64)
    liked_arr = (df["liked"].to_numpy(dtype=np.int8) if "liked" in df.columns
                 else np.zeros(len(df), dtype=np.int8))
    satisfied_arr = (df["satisfied"].to_numpy(dtype=np.int8) if "satisfied" in df.columns
                     else np.zeros(len(df), dtype=np.int8))
    dismissed_arr = (df["dismissed"].to_numpy(dtype=np.int8) if "dismissed" in df.columns
                     else np.zeros(len(df), dtype=np.int8))

    # A feed is a maximal run sharing (user, timestamp).
    new_feed = np.empty(len(df), dtype=bool)
    new_feed[0] = True
    new_feed[1:] = (user_codes[1:] != user_codes[:-1]) | (ts_ns[1:] != ts_ns[:-1])
    starts = np.flatnonzero(new_feed)
    stops = np.append(starts[1:], len(df))

    median_watch = float(np.median(watch_seconds[clicked == 1])) or 1.0
    n_catalog = len(video_ids)

    X_parts, y_parts, w_parts, g_parts, t_parts = [], [], [], [], []
    task_parts: dict[str, list[np.ndarray]] = {k: [] for k in TASK_LABELS}
    history: list[int] = []
    weights: list[float] = []
    current_user = -1
    feed_id = 0

    for start, stop in zip(starts, stops):
        if user_codes[start] != current_user:
            current_user = user_codes[start]
            history, weights = [], []

        rows = np.arange(start, stop)
        pos = rows[clicked[start:stop] == 1]
        neg = rows[clicked[start:stop] == 0]

        # Feeds where nothing was clicked are mostly redundant negatives; keep
        # a sample so the model still sees "none of these were good" pages.
        keep = len(pos) > 0 or rng.random() < no_click_feed_rate
        if keep:
            n_neg = min(len(neg), max(1, negatives_per_positive * max(len(pos), 1)))
            neg_sample = (neg[rng.choice(len(neg), size=n_neg, replace=False)]
                          if len(neg) else neg)
            selected = np.concatenate([pos, neg_sample]) if len(pos) else neg_sample

            # MIXED NEGATIVES. In-feed negatives alone teach the model to
            # separate "the click" from "other things the old policy already
            # judged plausible" -- a narrow question. At serving we score the
            # WHOLE catalog, most of which looks nothing like a logged
            # impression, so the model is asked an out-of-distribution question
            # and mis-calibrates. Adding uniformly sampled catalog items as
            # negatives covers that regime. (Yi et al., "Mixed Negative
            # Sampling", 2019.)
            n_random = random_negatives_per_positive * max(len(pos), 1) if keep else 0
            random_items = (rng.integers(0, n_catalog, size=n_random)
                            if n_random else np.zeros(0, dtype=np.int64))

            if keep and (len(selected) or len(random_items)):
                ctx = feature_builder.build_context(history, weights)
                feed_items = item[selected]
                all_items = np.concatenate([feed_items, random_items])
                X_parts.append(feature_builder.build(ctx, all_items))

                labels = np.concatenate([
                    clicked[selected].astype(np.float64),
                    np.zeros(len(random_items), dtype=np.float64),
                ])
                y_parts.append(labels)

                # IPS: 1 / P(examined at this rank), applied to clicks only --
                # the standard Joachims et al. estimator reweights observed
                # positives, not unclicked impressions.
                # Random negatives were never shown, so they have no position
                # and no propensity; they simply carry weight 1.
                propensity = np.maximum(np.power(position_decay, np.concatenate([
                    rank_shown[selected], np.zeros(len(random_items)),
                ])), ips_clip)
                ips = np.where(labels > 0, 1.0 / propensity, 1.0)

                if objective == "watch_time":
                    # THE YouTube 2016 trick: weight positives by watch time so
                    # the learned odds approximate E[watch time]. Ranking is
                    # invariant to a global scale on positive weights, so we
                    # normalise by the median instead of feeding raw seconds.
                    engagement = np.concatenate([
                        watch_seconds[selected] / median_watch,
                        np.zeros(len(random_items)),
                    ])
                    base = np.where(labels > 0, np.maximum(engagement, 0.05), 1.0)
                else:
                    base = np.ones(len(all_items), dtype=np.float64)

                # Multi-task labels for the same rows. Random negatives were
                # never shown, so every outcome label is 0 for them -- which is
                # correct: an item nobody saw produced no watch, like or
                # satisfaction.
                pad = np.zeros(len(random_items), dtype=np.float64)
                clicked_sel = clicked[selected].astype(np.float64)
                wf_sel = watch_fraction[selected]
                task_parts["click"].append(np.concatenate([clicked_sel, pad]))
                task_parts["long_watch"].append(np.concatenate([
                    ((clicked_sel > 0) & (wf_sel >= 0.5)).astype(np.float64), pad]))
                task_parts["completion"].append(np.concatenate([
                    ((clicked_sel > 0) & (wf_sel >= 0.9)).astype(np.float64), pad]))
                task_parts["liked"].append(np.concatenate([
                    liked_arr[selected].astype(np.float64), pad]))
                task_parts["satisfied"].append(np.concatenate([
                    satisfied_arr[selected].astype(np.float64), pad]))
                task_parts["dismissed"].append(np.concatenate([
                    dismissed_arr[selected].astype(np.float64), pad]))

                w_parts.append((base * ips).astype(np.float32))
                g_parts.append(np.full(len(all_items), feed_id, dtype=np.int64))
                t_parts.append(np.concatenate([
                    ts_ns[selected],
                    np.full(len(random_items), ts_ns[start], dtype=ts_ns.dtype),
                ]))
                feed_id += 1

        # ---- only NOW may this feed's clicks enter the history ----
        for r in pos:
            history.append(int(item[r]))
            weights.append(float(watch_fraction[r]))

        if verbose and feed_id and feed_id % 30000 == 0 and keep:
            print(f"  [rank] {feed_id:,} feeds replayed, "
                  f"{sum(len(p) for p in y_parts):,} rows")

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int8)
    w = np.concatenate(w_parts).astype(np.float32)
    g = np.concatenate(g_parts)
    t = np.concatenate(t_parts)

    meta = {
        "n_rows": int(len(y)),
        "n_positives": int(y.sum()),
        "n_feeds": int(feed_id),
        "positive_rate": round(float(y.mean()), 4),
        "objective": objective,
        "negatives_per_positive": negatives_per_positive,
        "random_negatives_per_positive": random_negatives_per_positive,
        "position_decay_assumed": position_decay,
        "n_features": len(FEATURE_NAMES),
    }
    if verbose:
        print(f"  [rank] training set: {meta['n_rows']:,} rows, "
              f"{meta['n_positives']:,} positives ({meta['positive_rate']:.1%}), "
              f"{meta['n_feeds']:,} feeds")
    task_y = {k: np.concatenate(v).astype(np.int8) for k, v in task_parts.items() if v}
    if verbose and task_y:
        rates = "  ".join(f"{k}={task_y[k].mean():.1%}" for k in TASK_LABELS if k in task_y)
        print(f"  [rank] task positive rates: {rates}")
    return TrainingSet(X=X, y=y, sample_weight=w, group=g, timestamp=t, meta=meta,
                       task_y=task_y)
