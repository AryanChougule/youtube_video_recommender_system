"""Load every build artifact into ready-to-use objects.

One loader shared by the training scripts, the evaluator and the API. Loading
logic that lives in three places drifts in three directions; this is the same
anti-skew reasoning as the shared FeatureBuilder.

Cheap integrity checks run at load time. A silent row-order mismatch between
the catalog and the factor matrices would not crash -- it would just recommend
the wrong videos forever, which is far worse than a loud failure on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .catalog_view import CatalogView
from .config import Config, Paths, load_config
from .features.text import TextIndex
from .rank.features import FeatureBuilder
from .rank.multitask import MultiTaskRanker
from .rank.ranker import Ranker
from .recall.cf import CoVisitation, ImplicitALS
from .recall.content import ContentRecall
from .recall.heuristic import ChannelRecall, TrendingRecall
from .serving import load_serving_models


class ArtifactError(RuntimeError):
    """Raised when artifacts are missing or inconsistent with the catalog."""


@dataclass
class Artifacts:
    config: Config
    catalog: CatalogView          # pandas-free; see recsys.catalog_view
    item_stats: CatalogView
    text_index: TextIndex
    content: ContentRecall
    covisitation: CoVisitation
    als: ImplicitALS
    trending: TrendingRecall
    channel: ChannelRecall
    features: FeatureBuilder
    ranker: Optional[Ranker]
    multitask: Optional[object]      # MultiTaskRanker, when trained
    index_meta: dict
    data_meta: dict

    # id <-> row index
    video_index: dict[str, int]
    video_ids: np.ndarray

    @property
    def n_items(self) -> int:
        return len(self.catalog)

    def idx(self, video_id: str) -> int:
        try:
            return self.video_index[video_id]
        except KeyError as exc:
            raise KeyError(f"unknown video_id {video_id!r}") from exc


def _require(path, what: str):
    if not path.exists():
        raise ArtifactError(
            f"missing {what}: {path}\nRun `python scripts/build_all.py` first."
        )
    return path


def load_artifacts(cfg: Config | None = None, with_ranker: bool = True,
                   verbose: bool = False) -> Artifacts:
    cfg = cfg or load_config()

    catalog = CatalogView.load(_require(Paths.serving_npz, "serving catalog"),
                               _require(Paths.serving_json, "serving catalog json"))
    item_stats = catalog          # stats live in the same bundle
    item_vectors = np.load(_require(Paths.item_vectors, "item vectors"))
    # The NumPy bundle carries the query encoder and every tree, so serving
    # imports neither scikit-learn nor joblib. See recsys.serving.trees for the
    # deployment failure that motivated it.
    models = load_serving_models(
        _require(Paths.serving_models_npz, "serving models"),
        _require(Paths.serving_models_json, "serving models meta"),
    )
    encoder = models.text
    cooc = np.load(_require(Paths.cooccurrence, "co-visitation"))
    user_factors = np.load(_require(Paths.als_user_factors, "ALS user factors"))
    item_factors = np.load(_require(Paths.als_item_factors, "ALS item factors"))
    index_meta = json.loads(_require(Paths.index_meta, "index meta").read_text(encoding="utf-8"))
    data_meta = json.loads(Paths.data_meta.read_text(encoding="utf-8")) if Paths.data_meta.exists() else {}

    n = len(catalog)
    # Every artifact is indexed by catalog row order. Assert it rather than
    # trusting it -- a mismatch is invisible at runtime and catastrophic.
    for name, array in (("item_vectors", item_vectors),
                        ("item_stats", item_stats),
                        ("als_item_factors", item_factors),
                        ("covisit_neighbours", cooc["neighbours"])):
        if len(array) != n:
            raise ArtifactError(
                f"{name} has {len(array)} rows but the catalog has {n}. "
                "Artifacts are stale -- rerun `python scripts/build_all.py`."
            )
    if index_meta.get("first_video_id") and index_meta["first_video_id"] != catalog["video_id"][0]:
        raise ArtifactError(
            "catalog row order changed since the models were trained. "
            "Rerun `python scripts/build_all.py`."
        )

    text_index = TextIndex(
        vectors=item_vectors, encoder=encoder,
        backend=str(index_meta.get("text_backend", "tfidf_svd")),
        dims=int(item_vectors.shape[1]), meta={},
    )
    covisitation = CoVisitation(
        neighbours=cooc["neighbours"], scores=cooc["scores"], popularity=cooc["popularity"],
    )
    als = ImplicitALS.from_arrays(
        user_factors, item_factors,
        alpha=float(index_meta.get("als_alpha", cfg.als.alpha)),
        regularization=float(index_meta.get("als_regularization", cfg.als.regularization)),
    )

    ranker: Optional[Ranker] = None
    if with_ranker and models.ranker is not None:
        ranker = Ranker.from_numpy(models.ranker, models.feature_names)
    elif with_ranker:
        raise ArtifactError(
            f"no ranker in {Paths.serving_models_npz}\n"
            "Run `python scripts/04_train_ranker.py`."
        )

    multitask = None
    if with_ranker and models.multitask:
        multitask = MultiTaskRanker.from_numpy(
            models.multitask, models.multitask_weights, models.feature_names)

    features = FeatureBuilder(
        catalog=catalog, item_stats=item_stats, item_vectors=item_vectors,
        covisitation=covisitation, als=als,
    )

    artifacts = Artifacts(
        config=cfg, catalog=catalog, item_stats=item_stats, text_index=text_index,
        content=ContentRecall(text_index), covisitation=covisitation, als=als,
        trending=TrendingRecall(item_stats, halflife_days=cfg.policy.freshness_halflife_days),
        channel=ChannelRecall(catalog, item_stats),
        features=features, ranker=ranker, multitask=multitask,
        index_meta=index_meta, data_meta=data_meta,
        video_index={str(v): i for i, v in enumerate(catalog["video_id"])},
        video_ids=catalog["video_id"].to_numpy(),
    )
    if verbose:
        print(f"  [artifacts] {n:,} videos, {item_vectors.shape[1]}-d text vectors, "
              f"{item_factors.shape[1]} ALS factors, ranker={'yes' if ranker else 'no'}")
    return artifacts
