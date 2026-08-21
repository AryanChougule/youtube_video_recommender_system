"""Stage 3: turn a ranked list into a good PAGE.

A pointwise ranker scores every candidate independently, so it cannot know that
the second slot is worth less if it holds the same thing as the first. Left
alone it produces ten near-identical videos, each individually optimal and
collectively useless. Page quality is a property of the SET.

Four policies, applied in order:

1. **Freshness boost** -- a small, explicit exponential-decay bonus. YouTube's
   catalog turns over hourly, so recency is product strategy, not something to
   be inferred from a log that already under-represents new uploads.
2. **MMR** (Carbonell & Goldstein, 1998) -- greedy selection that penalises an
   item by its similarity to what is ALREADY on the page, so redundancy is
   priced contextually rather than absolutely.
3. **Channel cap** -- a hard constraint. Soft penalties leak; if the rule is
   "never more than 2 from one creator", express it as a constraint.
4. **Exploration slots** -- reserved capacity for high-novelty items that still
   clear a relevance floor.

On (4): a recommender trains on data it generated itself. Show only what you
are confident about, users click only that, and it becomes your next training
set -- the long tail goes permanently dark. You cannot learn that a video is
good if you never show it. Reserved slots are a structured epsilon-greedy: less
statistically elegant than Thompson sampling, but it never shows something
irrelevant, which matters when the epsilon is visible to real users.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolicyResult:
    order: np.ndarray
    base_scores: np.ndarray            # relevance after freshness, pre-MMR
    policy_notes: dict[int, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] by RANK, not by value.

    Ranker output is an odds ratio with a long right tail: one candidate at
    odds 40 while the rest sit near 0.5 would, under min-max scaling, compress
    everything else into a sliver and make the MMR diversity term meaningless.
    Rank-normalising keeps the trade-off on a stable scale whatever the score
    distribution does.
    """
    n = len(scores)
    if n == 0:
        return scores
    if n == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(np.argsort(-scores))       # 0 = best
    return (1.0 - order / (n - 1)).astype(np.float32)


def freshness_multiplier(age_days: np.ndarray, halflife: float,
                         weight: float) -> np.ndarray:
    """1 + weight * 0.5 ** (age / halflife). Bounded and interpretable.

    Deliberately multiplicative and small: freshness should break ties between
    comparable videos, not override relevance. With weight=0.08 a brand-new
    video gets at most an 8% lift, which cannot rescue an irrelevant one.
    """
    return 1.0 + weight * np.power(0.5, np.maximum(age_days, 0.0) / max(halflife, 1e-6))


def mmr_select(
    relevance: np.ndarray,
    vectors: np.ndarray,
    k: int,
    lambda_: float = 0.72,
    channel_of: np.ndarray | None = None,
    max_per_channel: int = 2,
    category_of: np.ndarray | None = None,
    category_weight: float = 0.25,
) -> tuple[list[int], dict[int, list[str]]]:
    """Greedy MMR with a hard per-channel cap and category-aware similarity.

    The redundancy term blends text cosine with a category-match indicator:

        sim(a, b) = (1 - w) * cosine(a, b) + w * 1[category_a == category_b]

    Text cosine alone under-penalises stacking: two Food videos on different
    subjects can look textually distinct while still making the page feel
    monotonous to a human. Measured on a 3-Food/2-Gaming history, pure-cosine
    MMR returned 7 of 8 slots as Food. Category is the coarse signal a viewer
    actually perceives as "more of the same", so it belongs in the penalty.

    Returns positions INTO the candidate arrays, not catalog ids.
    """
    n = len(relevance)
    k = min(k, n)
    if n == 0:
        return [], {}

    notes: dict[int, list[str]] = {}
    selected: list[int] = []
    remaining = set(range(n))
    channel_counts: dict[str, int] = {}
    # max similarity of each candidate to anything already selected
    max_sim = np.zeros(n, dtype=np.float32)

    while len(selected) < k and remaining:
        idx = np.fromiter(remaining, dtype=int, count=len(remaining))
        mmr = lambda_ * relevance[idx] - (1.0 - lambda_) * max_sim[idx]

        # Channel cap as a hard constraint. Applied by masking rather than
        # penalising, so "at most 2" means exactly that.
        if channel_of is not None and max_per_channel > 0:
            blocked = np.array(
                [channel_counts.get(channel_of[i], 0) >= max_per_channel for i in idx],
                dtype=bool,
            )
            if blocked.all():
                # Every remaining candidate is capped out; relax rather than
                # return a short page.
                channel_counts.clear()
                blocked[:] = False
            mmr = np.where(blocked, -np.inf, mmr)

        best = int(idx[int(np.argmax(mmr))])
        if len(selected):
            penalty = (1.0 - lambda_) * max_sim[best]
            if penalty > 0.05 * lambda_ * max(relevance[best], 1e-6):
                notes.setdefault(best, []).append("diversity-adjusted")

        selected.append(best)
        remaining.discard(best)
        if channel_of is not None:
            channel_counts[channel_of[best]] = channel_counts.get(channel_of[best], 0) + 1

        # Incremental update: only the newly selected item can raise max_sim.
        if remaining:
            rest = np.fromiter(remaining, dtype=int, count=len(remaining))
            sims = vectors[rest] @ vectors[best]
            if category_of is not None and category_weight > 0:
                same_category = (category_of[rest] == category_of[best]).astype(np.float32)
                sims = (1.0 - category_weight) * sims + category_weight * same_category
            max_sim[rest] = np.maximum(max_sim[rest], sims)

    return selected, notes


def apply_policy(
    candidates: np.ndarray,
    ranker_scores: np.ndarray,
    vectors: np.ndarray,
    age_days: np.ndarray,
    channel_of: np.ndarray,
    category_of: np.ndarray,
    popularity: np.ndarray,
    k: int,
    mmr_lambda: float = 0.72,
    max_per_channel: int = 2,
    freshness_halflife: float = 21.0,
    freshness_weight: float = 0.08,
    exploration_slots: int = 2,
    category_weight: float = 0.25,
    seed: int = 42,
) -> PolicyResult:
    """Full Stage 3: freshness -> MMR + channel cap -> exploration slots."""
    if len(candidates) == 0:
        return PolicyResult(np.array([], dtype=int), np.array([]))

    rng = np.random.default_rng(seed)
    relevance = _rank_normalise(np.asarray(ranker_scores, dtype=np.float32))
    boosted = relevance * freshness_multiplier(age_days, freshness_halflife, freshness_weight)

    n_explore = min(exploration_slots, max(0, k - 1)) if len(candidates) > k else 0
    n_exploit = k - n_explore

    picks, notes = mmr_select(
        boosted, vectors, n_exploit, lambda_=mmr_lambda,
        channel_of=channel_of, max_per_channel=max_per_channel,
        category_of=category_of, category_weight=category_weight,
    )

    explored: list[int] = []
    if n_explore > 0:
        chosen = set(picks)
        # The channel cap is a HARD constraint on the finished page, so
        # exploration has to respect the budget MMR already spent. Selecting
        # exploration items independently let a third video from one channel
        # onto a max-2 page -- caught by tests/test_integration.py.
        used_per_channel: dict[str, int] = {}
        if channel_of is not None and max_per_channel > 0:
            for position in picks:
                key = channel_of[position]
                used_per_channel[key] = used_per_channel.get(key, 0) + 1

        pool = np.array(
            [i for i in range(len(candidates))
             if i not in chosen
             and not (channel_of is not None and max_per_channel > 0
                      and used_per_channel.get(channel_of[i], 0) >= max_per_channel)],
            dtype=int,
        )
        if len(pool):
            # Relevance floor: exploration must not become "show random junk".
            # Only candidates in the top half by relevance are eligible, so we
            # explore among plausible items -- trading a little novelty for the
            # guarantee that an exploration slot is never actively bad.
            floor = np.quantile(relevance[pool], 0.5)
            eligible = pool[relevance[pool] >= floor]
            if len(eligible) == 0:
                eligible = pool
            # Draw one at a time, updating the channel budget after each pick.
            # Sampling the whole batch at once would let two exploration slots
            # land on the same channel and breach the cap together.
            total = max(float(popularity.sum()), 1.0)
            for _ in range(min(n_explore, len(eligible))):
                if len(eligible) == 0:
                    break
                # Novelty = self-information: rarer items are likelier picks.
                novelty = -np.log2(np.maximum(popularity[candidates[eligible]], 1.0) / total)
                weights = novelty / novelty.sum() if novelty.sum() > 0 else None
                position = int(rng.choice(eligible, p=weights))
                explored.append(position)
                notes.setdefault(position, []).append("exploration")

                eligible = eligible[eligible != position]
                if channel_of is not None and max_per_channel > 0 and len(eligible):
                    key = channel_of[position]
                    used_per_channel[key] = used_per_channel.get(key, 0) + 1
                    if used_per_channel[key] >= max_per_channel:
                        eligible = eligible[channel_of[eligible] != key]

    # Interleave: exploration items go into the lower half of the page, where
    # a miss costs less attention than it would at the top.
    order_positions = list(picks)
    for offset, position in enumerate(explored):
        insert_at = min(len(order_positions), max(3, (k // 2) + offset))
        order_positions.insert(insert_at, int(position))
    order_positions = order_positions[:k]

    final = candidates[np.asarray(order_positions, dtype=int)]
    remapped = {int(final[rank]): notes.get(pos, [])
                for rank, pos in enumerate(order_positions) if notes.get(pos)}

    return PolicyResult(
        order=final,
        base_scores=boosted[np.asarray(order_positions, dtype=int)],
        policy_notes=remapped,
        stats={
            "n_candidates": int(len(candidates)),
            "n_exploration_slots": int(len(explored)),
            "mmr_lambda": mmr_lambda,
            "max_per_channel": max_per_channel,
            "distinct_channels": int(len(set(channel_of[order_positions]))),
        },
    )
