"""Typed configuration loaded from ``config.yaml``.

Everything the pipeline does is driven from a single YAML file so that a run is
reproducible from ``config.yaml`` + ``project.seed`` alone.  Pydantic gives us
validation and clear errors when a key is mistyped, which is a lot friendlier
than a ``KeyError`` three modules deep in the build.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/recsys/config.py -> src/recsys -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths:
    """Canonical locations for data and build artifacts."""

    root = PROJECT_ROOT
    config = PROJECT_ROOT / "config.yaml"
    data = PROJECT_ROOT / "data"
    raw = PROJECT_ROOT / "data" / "raw"
    processed = PROJECT_ROOT / "data" / "processed"
    artifacts = PROJECT_ROOT / "artifacts"
    docs = PROJECT_ROOT / "docs"
    static = PROJECT_ROOT / "src" / "recsys" / "api" / "static"

    # processed data
    catalog = processed / "catalog.parquet"
    interactions = processed / "interactions.parquet"
    users = processed / "users.parquet"
    data_meta = processed / "data_meta.json"

    # latent ground truth -- used ONLY by the simulator and by evaluation.
    # Never exposed through the API: leaking it would make every metric a lie.
    gt_item_topics = processed / "gt_item_topics.npy"
    gt_user_topics = processed / "gt_user_topics.npy"
    gt_session_intent = processed / "gt_session_intent.parquet"

    # build artifacts
    item_vectors = artifacts / "item_vectors.npy"
    text_encoder = artifacts / "text_encoder.joblib"
    text_backend = artifacts / "text_backend.json"
    cooccurrence = artifacts / "item_cooc.npz"
    als_user_factors = artifacts / "als_user_factors.npy"
    als_item_factors = artifacts / "als_item_factors.npy"
    ranker = artifacts / "ranker.joblib"
    multitask_ranker = artifacts / "multitask_ranker.joblib"
    item_stats = artifacts / "item_stats.parquet"
    # pandas-free serving bundle (see recsys.catalog_view)
    serving_npz = artifacts / "serving_catalog.npz"
    serving_json = artifacts / "serving_catalog.json"
    index_meta = artifacts / "index_meta.json"
    eval_report = artifacts / "evaluation.json"

    @classmethod
    def ensure(cls) -> None:
        for p in (cls.raw, cls.processed, cls.artifacts, cls.docs):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class ProjectCfg(BaseModel):
    name: str = "ReelRank"
    seed: int = 42
    reference_date: str | None = None


class YouTubeApiCfg(BaseModel):
    api_key_env: str = "YOUTUBE_API_KEY"
    queries_per_category: int = 6
    max_results_per_query: int = 50


class KaggleCfg(BaseModel):
    csv_glob: str = "data/raw/*videos.csv"


class SyntheticCfg(BaseModel):
    n_videos: int = 6000
    n_channels: int = 420


class CatalogCfg(BaseModel):
    source: Literal["auto", "youtube_api", "kaggle", "synthetic"] = "auto"
    target_size: int = 6000
    youtube_api: YouTubeApiCfg = Field(default_factory=YouTubeApiCfg)
    kaggle: KaggleCfg = Field(default_factory=KaggleCfg)
    synthetic: SyntheticCfg = Field(default_factory=SyntheticCfg)


class SimulatorCfg(BaseModel):
    n_users: int = 4000
    sessions_per_user_mean: float = 6.0
    min_session_len: int = 3
    max_session_len: int = 22
    popularity_bias: float = 0.55
    position_bias_decay: float = 0.82
    exploration_rate: float = 0.12
    persona_drift: float = 0.05
    completion_noise: float = 0.18
    min_interactions_per_user: int = 5
    # Session intent: the difference between "what this person likes" and
    # "what they want right now".
    intent_rate: float = 0.45            # share of sessions with a focused intent
    intent_offpersona_rate: float = 0.25  # of those, how many are genuinely new
    intent_strength_a: float = 6.0        # Beta(a, b) -> mean a/(a+b) = 0.67
    intent_strength_b: float = 3.0


class FeaturesCfg(BaseModel):
    text_backend: Literal["auto", "tfidf_svd", "sentence_transformers"] = "auto"
    svd_dims: int = 256
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_max_features: int = 60000
    tfidf_ngram_max: int = 2


class RecallCfg(BaseModel):
    content_k: int = 150
    cf_item_k: int = 150
    als_k: int = 150
    channel_k: int = 60
    trending_k: int = 60
    max_candidates: int = 400
    rrf_k: int = 60
    covisit_damping: float = 0.5
    weights: Dict[str, float] = {
        "content_history": 1.0,
        "cf_covisit": 1.15,
        "cf_als": 1.10,
        "channel": 0.75,
        "trending": 0.45,
    }


class AlsCfg(BaseModel):
    factors: int = 96
    regularization: float = 0.05
    alpha: float = 30.0
    iterations: int = 18


class RankerCfg(BaseModel):
    model: Literal["hgb", "logistic", "lightgbm"] = "hgb"
    objective: Literal["watch_time", "click"] = "watch_time"
    use_ips: bool = True
    negatives_per_positive: int = 4
    random_negatives_per_positive: int = 2
    test_size: float = 0.2
    learning_rate: float = 0.08
    max_iter: int = 350
    multitask: bool = True
    objective_weights: Dict[str, float] = {
        "click": 0.10, "long_watch": 0.25, "completion": 0.15,
        "liked": 0.15, "satisfied": 0.40, "dismissed": -0.20,
    }


class PolicyCfg(BaseModel):
    mmr_lambda: float = 0.72
    # Session-intent blending. Default 0.0 -- MEASURED not to help, because the
    # profile's recency decay already acts as a session model. Kept as a knob so
    # the Recommendation Lab can demonstrate the negative result rather than
    # hide it. See src/recsys/intent.py.
    intent_alpha_scale: float = 0.0
    max_per_channel: int = 2
    freshness_halflife_days: float = 21.0
    freshness_weight: float = 0.08
    exploration_slots: int = 2


class EvaluationCfg(BaseModel):
    k_values: List[int] = [5, 10, 20]
    protocol: Literal["leave_last_n", "temporal"] = "leave_last_n"
    n_holdout: int = 2
    max_eval_users: int = 1500
    catalog_sample_for_coverage: int = 2000


class ServingCfg(BaseModel):
    default_n: int = 24
    cache_size: int = 512
    host: str = "0.0.0.0"
    port: int = 7860


class Config(BaseModel):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    catalog: CatalogCfg = Field(default_factory=CatalogCfg)
    simulator: SimulatorCfg = Field(default_factory=SimulatorCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    recall: RecallCfg = Field(default_factory=RecallCfg)
    als: AlsCfg = Field(default_factory=AlsCfg)
    ranker: RankerCfg = Field(default_factory=RankerCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)
    evaluation: EvaluationCfg = Field(default_factory=EvaluationCfg)
    serving: ServingCfg = Field(default_factory=ServingCfg)


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml`` (cached).

    Any value may be overridden from the environment with a ``RECSYS__``
    prefix and ``__`` as the nesting separator, e.g.::

        RECSYS__SIMULATOR__N_USERS=500 python scripts/build_all.py

    which is handy for smoke tests and for tuning a deployed container without
    rebuilding the image.
    """
    cfg_path = Path(path) if path else Paths.config
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    for key, value in os.environ.items():
        if not key.startswith("RECSYS__"):
            continue
        parts = [p.lower() for p in key[len("RECSYS__"):].split("__")]
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    return Config(**raw)


def get_seed() -> int:
    return load_config().project.seed
