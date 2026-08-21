"""User-behaviour simulator: turns a catalog into a realistic watch log.

READ THIS FIRST -- it is the most important design decision in the project.

There is no public dataset of real YouTube watch histories (and there never
will be; watch history is among the most re-identifying data a platform holds
-- see the Netflix Prize de-anonymisation).  So the interaction log here is
*simulated*, and this module is the explicit, auditable statement of what we
assume about users.

A simulator is not "fake data".  It is a hypothesis about behaviour written in
code.  Its advantage over real logs is that the latent structure is KNOWN, so
we can ask a question real logs cannot answer: did the model recover genuine
preference structure, or did it just memorise popularity?  Its danger is
circularity -- a low-rank model will trivially fit a low-rank generator.

We defend against circularity in three ways:

  1. The generator is deliberately HARDER than any model we fit to it. It
     contains a sigmoid link, a cascade click process, position bias,
     popularity bias, duration-dependent completion, per-user mainstream-ness
     and taste drift.  None of that is representable by matrix factorisation,
     so if ALS still helps, the signal it found is real.
  2. We only ever observe what the *logging policy* chose to show, so the log
     carries presentation bias exactly like a production log would.
  3. The identical modelling code is validated against MovieLens-100k (real
     human ratings) in ``scripts/06_validate_on_movielens.py``.

The behavioural model, stage by stage
-------------------------------------
    feed (logging policy)  ->  examine (cascade)  ->  click  ->  watch  ->  like

* **Feed / logging policy.**  A weak incumbent recommender: part popularity,
  part persona-relevant, part "related to what you just watched" (autoplay).
  This is what creates presentation bias -- our training data is a biased
  sample of a previous policy's choices, which is the central pathology of all
  real recommender data.
* **Cascade click model** (Craswell et al., 2008).  The user scans top-down,
  clicks an item with probability p_i, and otherwise continues to the next
  position only with probability ``position_bias_decay``.  Position bias is
  therefore an emergent property of scanning, not a hand-applied multiplier.
* **Watch fraction.**  Beta-distributed around an affinity- and quality-driven
  mean, penalised by video length.  This is what lets us optimise WATCH TIME
  rather than clicks -- the single most consequential objective change in
  YouTube's own history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..clock import reference_now
from ..config import SimulatorCfg
from .schema import coerce_interactions

# Feed shape. Kept module-level rather than in config because changing them
# changes the *semantics* of the log (CTR, impressions-per-click), not just a
# volume knob -- the evaluation protocol assumes this shape.
FEED_SIZE = 8
HOPS_BASE = 2
HOPS_EXTRA_MEAN = 5.0

# Click-model coefficients (logit space). Calibrated so that overall CTR lands
# in the 10-20% band typical of a real feed surface.
W_AFFINITY = 3.4      # how much latent topic match drives a click
W_POPULARITY = 0.55   # scaled per-user by "mainstream-ness"
W_QUALITY = 0.45      # hidden appeal, partially visible via engagement rate
W_DURATION_FIT = 0.40 # match between video length and user preference
CLICK_BIAS = -1.95    # intercept; sets the base rate


@dataclass
class SimulationResult:
    interactions: pd.DataFrame
    users: pd.DataFrame
    user_topics: np.ndarray          # (n_users, n_topics) ground truth


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def _topic_top_items(item_topics: np.ndarray, per_topic: int = 400) -> np.ndarray:
    """For each micro-topic, the items most strongly loaded on it.

    Lets the logging policy draw persona-relevant candidates in O(1) instead of
    scoring the whole catalog on every hop.
    """
    n_topics = item_topics.shape[1]
    per_topic = min(per_topic, item_topics.shape[0])
    out = np.zeros((n_topics, per_topic), dtype=np.int32)
    for t in range(n_topics):
        out[t] = np.argpartition(-item_topics[:, t], per_topic - 1)[:per_topic]
    return out


def _item_neighbours(item_topics: np.ndarray, k: int = 60) -> np.ndarray:
    """Top-k nearest items in latent topic space -- the 'autoplay' graph.

    Note this is ground-truth similarity, NOT anything our models can see. It
    is what makes sessions topically coherent, which is in turn what gives
    co-visitation counting something real to recover.
    """
    norm = item_topics / (np.linalg.norm(item_topics, axis=1, keepdims=True) + 1e-12)
    n = norm.shape[0]
    k = min(k, n - 1)
    out = np.zeros((n, k), dtype=np.int32)
    block = 512
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = norm[start:stop] @ norm.T
        for i in range(stop - start):
            sim[i, start + i] = -np.inf          # exclude self
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-sim[rows, idx], axis=1)
        out[start:stop] = idx[rows, order]
    return out


def build_users(
    n_users: int,
    n_topics: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create user personas as sparse mixtures over micro-topics.

    Sparse, not dense: real people have 2-5 things they actually watch. A dense
    persona would make every user look average and destroy the CF signal.
    """
    n_interests = rng.integers(2, 5, size=n_users)
    user_topics = np.zeros((n_users, n_topics))
    for u in range(n_users):
        picks = rng.choice(n_topics, size=int(n_interests[u]), replace=False)
        weights = rng.dirichlet(np.full(len(picks), 1.6))
        user_topics[u, picks] = weights
    # A little background interest in everything: nobody is perfectly narrow,
    # and a hard zero would make held-out items outside the persona impossible.
    user_topics = 0.92 * user_topics + 0.08 / n_topics
    user_topics /= user_topics.sum(axis=1, keepdims=True)

    users = pd.DataFrame({
        "user_id": [f"u{i:05d}" for i in range(n_users)],
        # Mainstream-ness: how much raw popularity drives this person's clicks.
        # The niche end of this distribution is where popularity baselines fail
        # and real personalisation has to earn its keep.
        "mainstream": rng.beta(2.2, 2.2, size=n_users),
        # Preferred video length in minutes (log-normal, median ~11 min).
        "preferred_minutes": np.clip(rng.lognormal(2.4, 0.55, size=n_users), 1.0, 90.0),
        "activity": rng.lognormal(0.0, 0.5, size=n_users),
    })
    return user_topics, users


def simulate(
    catalog: pd.DataFrame,
    item_topics: np.ndarray,
    cfg: SimulatorCfg,
    seed: int = 42,
    verbose: bool = True,
    reference_date: str | None = None,
) -> SimulationResult:
    """Run the behavioural simulation and return an impression-level log."""
    rng = np.random.default_rng(seed + 977)
    n_items = len(catalog)
    n_topics = item_topics.shape[1]

    # ---------------- precomputation ----------------
    views = catalog["view_count"].to_numpy(dtype=np.float64)
    pop_weight = np.power(np.maximum(views, 1.0), cfg.popularity_bias)
    pop_p = pop_weight / pop_weight.sum()
    log_views_z = _zscore(np.log1p(views))

    quality = catalog["latent_quality"].to_numpy(dtype=np.float64) \
        if "latent_quality" in catalog.columns else np.ones(n_items)
    quality_z = _zscore(np.log(quality))

    duration_min = catalog["duration_seconds"].to_numpy(dtype=np.float64) / 60.0
    published = catalog["published_at"].to_numpy(dtype="datetime64[ns]")

    topic_tops = _topic_top_items(item_topics)
    neighbours = _item_neighbours(item_topics)

    user_topics, users = build_users(cfg.n_users, n_topics, rng)

    # Sessions land in a 90-day observation window ending "now".
    now = reference_now(reference_date)
    window_start = now - pd.Timedelta(days=90)
    window_seconds = (now - window_start).total_seconds()

    rows_u: list[int] = []
    rows_i: list[int] = []
    rows_s: list[int] = []
    rows_t: list[float] = []
    rows_rank: list[int] = []
    rows_click: list[int] = []
    rows_wf: list[float] = []
    rows_like: list[int] = []

    mainstream = users["mainstream"].to_numpy()
    pref_minutes = users["preferred_minutes"].to_numpy()
    activity = users["activity"].to_numpy()

    n_sessions_per_user = rng.poisson(
        np.maximum(cfg.sessions_per_user_mean * activity, 0.7)
    ).clip(1, 30)

    session_counter = 0
    beta_k = max(4.0, 1.0 / max(cfg.completion_noise, 1e-3) ** 2)

    for u in range(cfg.n_users):
        u_topics = user_topics[u].copy()
        seen: set[int] = set()

        for _ in range(int(n_sessions_per_user[u])):
            # Taste drifts slowly between sessions -- a stationary user would
            # make the temporal split meaningless.
            if rng.random() < cfg.persona_drift:
                shift = rng.dirichlet(np.full(n_topics, 0.35))
                u_topics = 0.85 * u_topics + 0.15 * shift
                u_topics /= u_topics.sum()

            session_counter += 1
            t0 = window_start + pd.Timedelta(seconds=float(rng.random() * window_seconds))
            clock = t0.timestamp()

            # Videos published after the session started cannot be shown. This
            # keeps the log causally valid for temporal evaluation.
            visible = published <= np.datetime64(t0)

            n_hops = HOPS_BASE + rng.poisson(HOPS_EXTRA_MEAN)
            last_item: int | None = None

            for _hop in range(int(n_hops)):
                # ---- logging policy: assemble a feed -------------------
                if last_item is None:
                    n_pop, n_persona, n_rel = 4, 4, 0
                else:
                    n_pop, n_persona, n_rel = 1, 2, 5

                cand: list[int] = []
                if n_pop:
                    cand += rng.choice(n_items, size=n_pop * 3, p=pop_p).tolist()
                if n_persona:
                    hot = rng.choice(n_topics, size=n_persona * 3, p=u_topics)
                    cand += [int(topic_tops[t, rng.integers(0, topic_tops.shape[1])]) for t in hot]
                if n_rel and last_item is not None:
                    picks = rng.integers(0, neighbours.shape[1], size=n_rel * 3)
                    cand += neighbours[last_item, picks].tolist()
                # Exploration: the incumbent policy occasionally shows something
                # entirely random, which is the ONLY source of unbiased data.
                if rng.random() < cfg.exploration_rate:
                    cand += rng.integers(0, n_items, size=3).tolist()

                feed: list[int] = []
                for c in cand:
                    c = int(c)
                    if c in seen or c in feed or not visible[c]:
                        continue
                    feed.append(c)
                    if len(feed) == FEED_SIZE:
                        break
                if not feed:
                    break

                # Randomise slot order before display. Without this, position
                # is confounded with candidate SOURCE (popularity picks are
                # assembled first, so autoplay picks would always land in the
                # rarely-examined tail) and the log would conflate position
                # bias with source quality. Shuffling makes position bias pure.
                # It also makes our logs *less* biased than a real production
                # log would be -- noted as a limitation in docs/METHODOLOGY.md.
                rng.shuffle(feed)
                idx = np.asarray(feed)

                # ---- click probability --------------------------------
                affinity = item_topics[idx] @ u_topics          # ~0.01 .. 0.35
                aff_scaled = affinity * 12.0
                dur_fit = -np.abs(np.log(duration_min[idx] / pref_minutes[u]))
                logit = (
                    CLICK_BIAS
                    + W_AFFINITY * aff_scaled
                    + W_POPULARITY * mainstream[u] * 2.0 * log_views_z[idx]
                    + W_QUALITY * quality_z[idx]
                    + W_DURATION_FIT * dur_fit
                )
                p_click = 1.0 / (1.0 + np.exp(-logit))

                # ---- cascade scan (Craswell et al. 2008) ---------------
                clicked_pos = -1
                for pos in range(len(idx)):
                    if rng.random() < p_click[pos]:
                        clicked_pos = pos
                        break
                    if rng.random() > cfg.position_bias_decay:
                        break                    # abandoned the scan

                # ---- log every impression in the feed -----------------
                for pos, item in enumerate(idx):
                    item = int(item)
                    is_click = int(pos == clicked_pos)
                    wf = 0.0
                    liked = 0
                    if is_click:
                        aff_n = float(np.clip(affinity[pos] * 6.0, 0.0, 1.0))
                        base = 0.30 + 0.45 * aff_n + 0.12 * float(np.tanh(quality_z[item]))
                        length_penalty = float(
                            np.exp(-max(duration_min[item] - pref_minutes[u], 0.0) / 25.0)
                        )
                        mean_wf = float(np.clip(base * length_penalty, 0.03, 0.97))
                        wf = float(rng.beta(mean_wf * beta_k, (1.0 - mean_wf) * beta_k))
                        liked = int(rng.random() < min(0.45, 0.02 + 0.30 * wf * wf))
                        seen.add(item)
                        last_item = item
                        clock += duration_min[item] * 60.0 * wf + 12.0

                    rows_u.append(u)
                    rows_i.append(item)
                    rows_s.append(session_counter)
                    rows_t.append(clock)
                    rows_rank.append(pos)
                    rows_click.append(is_click)
                    rows_wf.append(wf)
                    rows_like.append(liked)

                if clicked_pos < 0:
                    # Nothing appealed; the user leaves with probability 0.6.
                    if rng.random() < 0.6:
                        break

        if verbose and (u + 1) % 1000 == 0:
            print(f"  simulated {u + 1}/{cfg.n_users} users, {len(rows_u):,} impressions")

    video_ids = catalog["video_id"].to_numpy()
    user_ids = users["user_id"].to_numpy()
    durations = catalog["duration_seconds"].to_numpy(dtype=np.float64)

    item_arr = np.asarray(rows_i)
    user_arr = np.asarray(rows_u)
    wf_arr = np.asarray(rows_wf)

    inter = pd.DataFrame({
        "user_id": user_ids[user_arr],
        "video_id": video_ids[item_arr],
        "session_id": [f"s{s}" for s in rows_s],
        "ts": pd.to_datetime(np.asarray(rows_t), unit="s"),
        "rank_shown": rows_rank,
        "clicked": rows_click,
        "watch_fraction": wf_arr,
        "watch_seconds": wf_arr * durations[item_arr],
        "liked": rows_like,
    })
    inter = coerce_interactions(inter).sort_values(["user_id", "ts"]).reset_index(drop=True)

    # Drop users too sparse to evaluate (cannot hold out from 1 interaction).
    clicks_per_user = inter[inter["clicked"] == 1].groupby("user_id").size()
    keep = set(clicks_per_user[clicks_per_user >= cfg.min_interactions_per_user].index)
    before = inter["user_id"].nunique()
    inter = inter[inter["user_id"].isin(keep)].reset_index(drop=True)

    kept_mask = users["user_id"].isin(keep).to_numpy()
    users = users[kept_mask].reset_index(drop=True)
    user_topics = user_topics[kept_mask]

    if verbose:
        n_click = int(inter["clicked"].sum())
        print(f"  users kept        : {len(users):,} / {before:,} "
              f"(dropped <{cfg.min_interactions_per_user} clicks)")
        print(f"  impressions       : {len(inter):,}")
        print(f"  clicks            : {n_click:,}  (CTR {n_click / max(len(inter),1):.1%})")
        print(f"  clicks per user   : {n_click / max(len(users),1):.1f}")

    return SimulationResult(interactions=inter, users=users, user_topics=user_topics)
