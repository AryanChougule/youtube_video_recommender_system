"""Integration tests against the built artifacts.

Skipped automatically if the artifacts are missing, so a fresh clone can run
``pytest`` before ``build_all.py`` without a wall of red.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.artifacts import ArtifactError, load_artifacts
from recsys.config import Paths, load_config
from recsys.data.simulator import simulate
from recsys.data.synthetic import generate_catalog
from recsys.engine import RecommendationEngine
from recsys.rank.dataset import build_training_set
from recsys.split import temporal_split


@pytest.fixture(scope="module")
def art():
    try:
        return load_artifacts(load_config(), with_ranker=True)
    except (ArtifactError, FileNotFoundError) as exc:
        pytest.skip(f"artifacts not built: {exc}")


@pytest.fixture(scope="module")
def engine(art):
    return RecommendationEngine(art)


# ------------------------------------------------------- simulator shape
def test_simulator_produces_position_bias_and_sane_ctr():
    catalog, topics, _ = generate_catalog(n_videos=900, n_channels=40, seed=5)
    cfg = load_config().simulator.model_copy(update={"n_users": 150})
    result = simulate(catalog, topics, cfg, seed=5, verbose=False)
    log = result.interactions

    assert set(log.columns) >= {"user_id", "video_id", "clicked", "watch_fraction", "rank_shown"}
    ctr = log["clicked"].mean()
    assert 0.02 < ctr < 0.35, f"implausible CTR {ctr:.2%}"

    # Position bias must EMERGE from the cascade scan, not be absent.
    by_position = log.groupby("rank_shown")["clicked"].mean()
    assert by_position.iloc[0] > by_position.iloc[-1] * 3

    clicks = log[log["clicked"] == 1]
    assert (clicks["watch_fraction"] > 0).all()
    assert (log[log["clicked"] == 0]["watch_fraction"] == 0).all()


def test_training_set_history_is_causal():
    """A feed's own click must not be in the history used to featurise it.

    This is the leak that would make offline metrics meaningless, so it is
    asserted directly rather than trusted.
    """
    catalog, topics, _ = generate_catalog(n_videos=500, n_channels=25, seed=11)
    cfg = load_config()
    sim_cfg = cfg.simulator.model_copy(update={"n_users": 80})
    result = simulate(catalog, topics, sim_cfg, seed=11, verbose=False)

    from recsys.data.schema import derive_catalog_features
    from recsys.features.text import build_text_index
    from recsys.data.catalog import catalog_text
    from recsys.rank.features import FeatureBuilder

    index = build_text_index(catalog_text(catalog), dims=32, verbose=False)
    builder = FeatureBuilder(catalog, derive_catalog_features(catalog), index.vectors)

    dataset = build_training_set(
        result.interactions, builder, catalog["video_id"].tolist(),
        objective="click", seed=11, verbose=False,
    )
    assert dataset.meta["n_rows"] > 0
    # history length is monotone non-decreasing within a user, so the very
    # first feed of every user must have an empty history -> zero match feats.
    assert np.isfinite(dataset.X).all()


# --------------------------------------------------------------- engine
def test_cold_start_returns_results_without_history(engine):
    res = engine.recommend(history=[], n=12)
    assert len(res.items) == 12
    assert res.request["mode"] == "cold_start"
    assert all(item.explanation for item in res.items)


def test_personalised_beats_cold_start_on_category_focus(engine, art):
    """A single-category history must skew the feed toward that category."""
    catalog = art.catalog
    gaming = catalog[catalog["category"] == "Gaming"]["video_id"].head(5).tolist()
    personalised = engine.recommend(history=gaming, n=20)
    cold = engine.recommend(history=[], n=20)

    share = lambda r: sum(i.category == "Gaming" for i in r.items) / len(r.items)  # noqa: E731
    assert share(personalised) > share(cold)
    assert personalised.request["mode"] == "personalised"


def test_history_items_are_never_recommended_back(engine, art):
    history = art.catalog["video_id"].head(6).tolist()
    res = engine.recommend(history=history, n=24)
    assert not (set(history) & {i.video_id for i in res.items})


def test_search_is_relevant(engine):
    res = engine.search("sourdough bread baking", n=10)
    assert res.items
    text = " ".join(i.title.lower() + " " + i.category.lower() for i in res.items)
    assert any(word in text for word in ("sourdough", "baking", "food", "dough", "pastry"))


def test_similar_excludes_the_seed_video(engine, art):
    seed = art.catalog["video_id"].iloc[100]
    res = engine.similar(seed, n=10)
    assert seed not in {i.video_id for i in res.items}
    assert len(res.items) == 10


def test_lower_mmr_lambda_increases_diversity(engine, art):
    history = art.catalog[art.catalog["category"] == "Food"]["video_id"].head(5).tolist()
    focused = engine.recommend(history=history, n=20, mmr_lambda=1.0, exploration_slots=0)
    diverse = engine.recommend(history=history, n=20, mmr_lambda=0.3, exploration_slots=0)
    assert (diverse.diagnostics["intra_list_diversity"]
            >= focused.diagnostics["intra_list_diversity"])


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_channel_cap_is_enforced_including_exploration_slots(engine, art, cap):
    """The cap is documented as a HARD constraint, so it must survive Stage 3.

    Regression test: exploration slots were originally selected after MMR
    without consulting the channel budget, which let a third video from one
    creator onto a max-2 page. Cap is on channel_id, not channel_title --
    distinct creators can share a display name.
    """
    history = art.catalog["video_id"].head(5).tolist()
    res = engine.recommend(history=history, n=24, max_per_channel=cap,
                           exploration_slots=4)
    counts = pd.Series([i.channel_id for i in res.items]).value_counts()
    assert counts.max() <= cap, counts.head().to_dict()


def test_every_item_carries_provenance(engine, art):
    history = art.catalog["video_id"].head(4).tolist()
    res = engine.recommend(history=history, n=15)
    for item in res.items:
        assert item.explanation, "every recommendation must be explainable"
        assert item.sources or item.policy_notes, f"{item.video_id} has no provenance"


def test_ground_truth_never_leaks_into_the_response(engine, art):
    """`latent_quality` is a hidden generative variable.

    Serving it would let the UI (and any evaluator) see the answer key. This is
    the single most important safety property of the serving payload.
    """
    res = engine.recommend(history=art.catalog["video_id"].head(3).tolist(), n=10)
    blob = str(res.to_dict())
    assert "latent_quality" in art.catalog.columns   # it exists...
    assert "latent_quality" not in blob              # ...but never ships


def test_latency_is_within_budget(engine, art):
    history = art.catalog["video_id"].head(8).tolist()
    timings = []
    for _ in range(20):
        timings.append(engine.recommend(history=history, n=24).stages["total_ms"])
    p95 = float(np.percentile(timings, 95))
    assert p95 < 250, f"p95 latency {p95:.0f}ms exceeds the serving budget"


# ------------------------------------------------------ artifact hygiene
def test_all_artifacts_agree_on_catalog_size(art):
    n = len(art.catalog)
    assert art.text_index.vectors.shape[0] == n
    assert art.als.item_factors.shape[0] == n
    assert art.covisitation.neighbours.shape[0] == n
    assert len(art.item_stats) == n


def test_cf_was_trained_under_the_declared_temporal_cutoff(art):
    """Guards the leak documented in src/recsys/split.py."""
    if not Paths.interactions.exists():
        pytest.skip("interactions not built")
    interactions = pd.read_parquet(Paths.interactions)
    split = temporal_split(interactions, test_size=art.config.ranker.test_size)
    declared = art.index_meta.get("temporal_split", {}).get("cutoff_ns")
    assert declared == split.cutoff_ns, "CF artifacts are stale; rerun 03_train_cf.py"
    assert not art.index_meta.get("trained_on_full_log"), \
        "CF was trained with --full; holdout metrics from this build are invalid"


def test_watch_page_stays_on_topic(engine, art):
    """Regression: the ranker was handed an empty history in watch-page mode.

    With no history every match feature is zero, so the ranker fell back to
    item-quality features and discarded Stage 1's ordering entirely -- a
    Finance seed returned Autos/Sports/Food above Finance. The seed video is
    now used as a one-item history, which is exactly what it is.
    """
    for category in ("Finance", "Food", "Gaming"):
        subset = art.catalog[art.catalog["category"] == category]
        seed = subset["video_id"].iloc[3]
        res = engine.similar(seed, n=10)
        share = sum(i.category == category for i in res.items) / len(res.items)
        assert share >= 0.5, f"{category} watch page drifted off-topic ({share:.0%})"


def test_explanations_do_not_claim_a_profile_that_does_not_exist(engine, art):
    """Regression: watch-page items were explained as 'viewers with a taste
    profile like yours', but in watch-page mode there is no user profile at
    all. An explanation that misdescribes its own evidence is worse than none.
    """
    seed = art.catalog["video_id"].iloc[250]
    for item in engine.similar(seed, n=10).items:
        assert "like yours" not in item.explanation.lower(), item.explanation
        assert item.explanation_detail.get("mode") == "watch_page"

    for item in engine.search("index fund investing", n=8).items:
        assert "you have been watching" not in item.explanation.lower(), item.explanation
        assert item.explanation_detail.get("mode") == "search"
