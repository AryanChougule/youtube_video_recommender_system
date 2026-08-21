"""Fast unit tests. No artifacts required.

These cover the invariants that ACTUALLY broke while building this system --
tie-breaking in ranking metrics, causal history construction, MMR behaviour --
rather than trivia that was never going to fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys import metrics
from recsys.data.schema import coerce_catalog, derive_catalog_features, split_tags
from recsys.data.synthetic import generate_catalog
from recsys.data.topics import CATEGORIES, N_TOPICS, TOPIC_INDEX
from recsys.policy.rerank import freshness_multiplier, mmr_select
from recsys.recall.blend import reciprocal_rank_fusion
from recsys.recall.content import RecallResult


# --------------------------------------------------------------- taxonomy
def test_every_category_topic_is_known():
    unknown = [(c, t) for c, mix in CATEGORIES.items() for t in mix if t not in TOPIC_INDEX]
    assert unknown == []


def test_categories_share_bridge_topics():
    """Cross-category bridges are the whole point of the latent design."""
    counts: dict[str, int] = {}
    for mix in CATEGORIES.values():
        for topic in mix:
            counts[topic] = counts.get(topic, 0) + 1
    assert sum(1 for n in counts.values() if n > 1) >= 5


# ---------------------------------------------------------------- schema
def test_coerce_catalog_fills_missing_columns_and_drops_untitled():
    raw = pd.DataFrame({"video_id": ["a", "b", ""], "title": ["Hi", "  ", "x"]})
    out = coerce_catalog(raw)
    assert list(out["video_id"]) == ["a"]          # blank title + blank id dropped
    assert "view_count" in out.columns and out["view_count"].dtype == "int64"


def test_split_tags_handles_nan_and_empty():
    assert split_tags(None) == []
    assert split_tags(float("nan")) == []
    assert split_tags("a|b| c ") == ["a", "b", "c"]


def test_derived_features_are_finite():
    catalog, _, _ = generate_catalog(n_videos=200, n_channels=20, seed=1)
    stats = derive_catalog_features(catalog)
    for column in ("age_days", "log_views", "engagement_rate", "views_per_day"):
        assert np.isfinite(stats[column]).all(), column
    assert (stats["age_days"] >= 0).all()


# -------------------------------------------------------- synthetic data
def test_catalog_is_heavy_tailed_and_aligned_with_topics():
    catalog, topics, _ = generate_catalog(n_videos=800, n_channels=40, seed=7)
    assert len(catalog) == 800
    assert topics.shape == (800, N_TOPICS)
    np.testing.assert_allclose(topics.sum(axis=1), 1.0, atol=1e-6)
    views = catalog["view_count"]
    # A power-law catalog: the top 5% must hold a disproportionate share.
    top_share = views.nlargest(40).sum() / views.sum()
    assert top_share > 0.20, f"catalog is not heavy-tailed enough ({top_share:.2%})"


def test_generate_catalog_is_deterministic():
    a, ta, _ = generate_catalog(n_videos=120, n_channels=10, seed=3)
    b, tb, _ = generate_catalog(n_videos=120, n_channels=10, seed=3)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(ta, tb)


# --------------------------------------------------------------- metrics
def test_precision_recall_ndcg_on_known_lists():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 5, 9}
    assert metrics.precision_at_k(recommended, relevant, 5) == pytest.approx(2 / 5)
    assert metrics.recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)
    assert metrics.hit_rate(recommended, relevant, 5) == 1.0
    assert metrics.reciprocal_rank(recommended, relevant, 5) == pytest.approx(1 / 2)


def test_ndcg_is_one_for_perfect_order_and_lower_when_reversed():
    relevance = {10: 1.0, 20: 0.5, 30: 0.2}
    assert metrics.ndcg_at_k([10, 20, 30], relevance, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([30, 20, 10], relevance, 3) < 0.9


def test_average_precision_rewards_early_hits():
    early = metrics.average_precision_at_k([1, 9, 9, 9], {1, 2}, 4)
    late = metrics.average_precision_at_k([9, 9, 9, 1], {1, 2}, 4)
    assert early > late


def test_gini_spans_even_to_concentrated():
    assert metrics.gini_coefficient(np.ones(100)) == pytest.approx(0.0, abs=1e-6)
    winner_take_all = np.zeros(100); winner_take_all[0] = 100
    assert metrics.gini_coefficient(winner_take_all) > 0.95


def test_within_group_top1_breaks_ties_randomly():
    """Regression test for a real bug.

    Training rows are laid out positive-first. With all-tied scores a naive
    argmax awards every group to the positive, which made a near-useless
    feature appear to score 80% top-1. Ties must be broken at random, so a
    constant scorer should land near 1/group_size, not 1.0.
    """
    n_groups, size = 400, 5
    groups = np.repeat(np.arange(n_groups), size)
    labels = np.tile(np.r_[1, np.zeros(size - 1, dtype=int)], n_groups)
    constant = np.ones(n_groups * size)
    score = metrics.within_group_top1(constant, labels, groups, seed=0)
    assert 0.10 < score < 0.30, f"tie-breaking is broken: got {score}"


def test_within_group_top1_is_one_for_a_perfect_scorer():
    groups = np.repeat(np.arange(50), 4)
    labels = np.tile(np.r_[1, 0, 0, 0], 50)
    assert metrics.within_group_top1(labels.astype(float), labels, groups) == 1.0


def test_novelty_prefers_obscure_items():
    popularity = np.array([1000.0, 1000.0, 1.0, 1.0])
    assert metrics.novelty([2, 3], popularity) > metrics.novelty([0, 1], popularity)


# ------------------------------------------------------------------ RRF
def test_rrf_fuses_on_rank_not_score():
    """An item ranked top by two sources must beat one with a huge score in one."""
    a = RecallResult(np.array([1, 2]), np.array([0.9, 0.8]), "a")
    b = RecallResult(np.array([2, 3]), np.array([1e6, 1.0]), "b")
    fused = reciprocal_rank_fusion({"a": a, "b": b}, rrf_k=60)
    # item 2 is rank 1 in 'a' and rank 0 in 'b'; item 1 is rank 0 in 'a' only
    assert fused.indices[0] == 2
    assert set(fused.sources_for(2)) == {"a", "b"}


def test_rrf_handles_empty_sources():
    empty = RecallResult(np.array([], dtype=int), np.array([]), "e")
    good = RecallResult(np.array([5]), np.array([1.0]), "g")
    fused = reciprocal_rank_fusion({"e": empty, "g": good})
    assert list(fused.indices) == [5]
    assert fused.source_sizes["e"] == 0


# ------------------------------------------------------------------ MMR
def _orthogonal_blocks(n_per: int = 6) -> np.ndarray:
    """Two tight clusters that are orthogonal to each other."""
    block_a = np.tile(np.array([1.0, 0.0, 0.0]), (n_per, 1))
    block_b = np.tile(np.array([0.0, 1.0, 0.0]), (n_per, 1))
    return np.vstack([block_a, block_b]).astype(np.float32)


def test_mmr_lower_lambda_picks_more_diversely():
    vectors = _orthogonal_blocks()
    # relevance strongly favours the first cluster
    relevance = np.r_[np.linspace(1.0, 0.9, 6), np.linspace(0.6, 0.5, 6)].astype(np.float32)

    greedy, _ = mmr_select(relevance, vectors, k=4, lambda_=1.0)
    diverse, _ = mmr_select(relevance, vectors, k=4, lambda_=0.3)

    assert all(i < 6 for i in greedy), "lambda=1 should stay in the top cluster"
    assert any(i >= 6 for i in diverse), "low lambda should cross into cluster B"


def test_mmr_respects_the_channel_cap():
    vectors = _orthogonal_blocks()
    relevance = np.linspace(1.0, 0.5, 12).astype(np.float32)
    channels = np.array(["c1"] * 6 + ["c2"] * 6)
    picks, _ = mmr_select(relevance, vectors, k=4, lambda_=1.0,
                          channel_of=channels, max_per_channel=2)
    counts = pd.Series(channels[picks]).value_counts()
    assert counts.max() <= 2


def test_mmr_returns_k_items_even_when_every_channel_is_capped():
    vectors = _orthogonal_blocks(3)
    relevance = np.linspace(1.0, 0.5, 6).astype(np.float32)
    channels = np.array(["only"] * 6)
    picks, _ = mmr_select(relevance, vectors, k=5, lambda_=1.0,
                          channel_of=channels, max_per_channel=2)
    assert len(picks) == 5, "cap must relax rather than truncate the page"


def test_freshness_multiplier_is_bounded_and_monotone():
    fresh = freshness_multiplier(np.array([0.0]), halflife=21.0, weight=0.08)[0]
    old = freshness_multiplier(np.array([500.0]), halflife=21.0, weight=0.08)[0]
    assert fresh == pytest.approx(1.08)
    assert old == pytest.approx(1.0, abs=1e-6)
    assert fresh > old


# ------------------------------------------------------- reproducibility
def test_build_is_byte_identical_with_a_pinned_reference_date():
    """The README claims reproducibility; this asserts it.

    The catalog dates publish times relative to "now" and the simulator places
    sessions in a window ending at "now". Both originally called utcnow(), so
    two runs with the same seed produced DIFFERENT data -- quietly falsifying
    the claim. Pinning project.reference_date makes the pipeline deterministic
    across machines and across days.
    """
    from recsys.config import load_config
    from recsys.data.simulator import simulate

    pinned = "2026-01-15"
    sim_cfg = load_config().simulator.model_copy(update={"n_users": 40})

    def build():
        catalog, topics, _ = generate_catalog(
            n_videos=250, n_channels=15, seed=42, reference_date=pinned)
        log = simulate(catalog, topics, sim_cfg, seed=42, verbose=False,
                       reference_date=pinned).interactions
        return catalog, log

    cat_a, log_a = build()
    cat_b, log_b = build()
    pd.testing.assert_frame_equal(cat_a, cat_b)
    pd.testing.assert_frame_equal(log_a, log_b)


def test_reference_date_actually_shifts_the_timeline():
    """Guards against the pin being silently ignored."""
    from recsys.clock import reference_now
    early = generate_catalog(n_videos=80, n_channels=8, seed=1,
                             reference_date="2026-01-15")[0]
    late = generate_catalog(n_videos=80, n_channels=8, seed=1,
                            reference_date="2026-06-01")[0]
    assert late["published_at"].max() > early["published_at"].max()
    assert reference_now("2026-01-15") == pd.Timestamp("2026-01-15")
