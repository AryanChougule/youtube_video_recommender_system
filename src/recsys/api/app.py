"""FastAPI service: REST API + the YouTube-style testing UI.

Serving design
--------------
* **Artifacts load once at startup**, not per request. Cold-loading 6MB of
  vectors on every call would dominate the latency budget.
* **Stateless.** Watch history lives in the browser and is posted with each
  request. No sessions, no database, so the container scales horizontally and
  an evaluator can hard-refresh without losing anything they cannot rebuild in
  two clicks.
* **Every response carries its own explanation.** The brief asks evaluators to
  understand WHY a recommendation appeared, so provenance (which recall source,
  at what rank, what the ranker scored) ships with the payload rather than
  living in a log the evaluator cannot see.
"""

from __future__ import annotations

import json
import os
import time

# BLAS threads must be pinned before numpy is imported anywhere. Serving does
# many tiny linear-algebra ops (a 96x96 ALS fold-in per request); with default
# threading that single solve costs 3.6ms instead of 51us.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from ..artifacts import ArtifactError, load_artifacts  # noqa: E402
from ..config import Paths, load_config  # noqa: E402
from ..engine import RecommendationEngine  # noqa: E402

app = FastAPI(
    title="ReelRank - YouTube-style Recommender",
    description="Two-stage video recommendation: candidate generation -> ranking -> policy",
    version="1.0.0",
)

STATE: dict = {"engine": None, "error": None, "loaded_at": None}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    history: list[str] = Field(default_factory=list,
                               description="video_ids watched, oldest first")
    watch_weights: list[float] | None = Field(
        default=None, description="watch fraction per history item (0-1)")
    seed_video: str | None = None
    query: str | None = None
    n: int = Field(default=24, ge=1, le=60)
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    exploration_slots: int | None = Field(default=None, ge=0, le=10)
    max_per_channel: int | None = Field(default=None, ge=0, le=10)
    # Recommendation Lab: choose the OBJECTIVE per request. The multi-task
    # heads are fixed at training time, so switching the objective costs one
    # weighted sum -- no retraining, no redeploy.
    objective_weights: dict[str, float] | None = None
    intent_alpha_scale: float | None = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _startup() -> None:
    started = time.time()
    try:
        cfg = load_config()
        artifacts = load_artifacts(cfg, with_ranker=True, verbose=True)
        STATE["engine"] = RecommendationEngine(artifacts)
        STATE["loaded_at"] = time.time()
        print(f"[api] artifacts loaded in {time.time() - started:.1f}s "
              f"({artifacts.n_items:,} videos)")
    except Exception as exc:  # noqa: BLE001
        # Start anyway and report the problem through /api/health, so a broken
        # deploy shows a diagnosable page instead of a bare 500.
        #
        # Deliberately catches Exception, not just ArtifactError. A serverless
        # platform that prunes "unused" dependency files can break unpickling
        # with ModuleNotFoundError long before any of our own checks run --
        # which is exactly what happened on Vercel, and a 502 with no body is
        # far harder to diagnose than a health endpoint that names the cause.
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[api] FAILED to load artifacts: {STATE['error']}")


def engine() -> RecommendationEngine:
    if STATE["engine"] is None:
        raise HTTPException(503, detail=STATE["error"] or "artifacts not loaded")
    return STATE["engine"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    ready = STATE["engine"] is not None
    out: dict = {"status": "ok" if ready else "degraded", "ready": ready}
    if STATE["error"]:
        out["error"] = STATE["error"]
    if ready:
        art = STATE["engine"].art
        out["catalog_size"] = art.n_items
        out["data_source"] = art.data_meta.get("source")
        out["has_ranker"] = art.ranker is not None
    return out


@app.get("/api/meta")
def meta() -> dict:
    """System description for the UI's 'How it works' panel."""
    art = engine().art
    cfg = art.config
    payload = {
        "project": cfg.project.name,
        "catalog": {
            "size": art.n_items,
            "channels": int(art.catalog["channel_id"].nunique()),
            "categories": sorted(set(art.catalog["category"].tolist())),
            "source": art.data_meta.get("source"),
            "latent_topics_origin": art.data_meta.get("latent_topics_origin"),
        },
        "data": {k: art.data_meta.get(k) for k in
                 ("n_users", "n_impressions", "n_clicks", "ctr",
                  "clicks_per_user", "matrix_density")},
        "models": {
            "text_backend": art.text_index.backend,
            "text_dims": art.text_index.dims,
            "als_factors": art.index_meta.get("als_factors"),
            "als_alpha": art.index_meta.get("als_alpha"),
            "covisitation_damping": art.index_meta.get("covisitation_damping"),
            "ranker": art.ranker.model_name if art.ranker else None,
            "ranker_features": len(art.ranker.feature_names) if art.ranker else 0,
        },
        "policy": {
            "mmr_lambda": cfg.policy.mmr_lambda,
            "max_per_channel": cfg.policy.max_per_channel,
            "exploration_slots": cfg.policy.exploration_slots,
            "freshness_halflife_days": cfg.policy.freshness_halflife_days,
        },
        "recall_weights": cfg.recall.weights,
        "objectives": {
            "available": list(cfg.ranker.objective_weights),
            "default_weights": cfg.ranker.objective_weights,
            "multitask_trained": art.multitask is not None,
            "per_task_auc": (getattr(art.multitask, "metrics", None).per_task_auc
                             if art.multitask is not None else {}),
        },
        "intent": {
            "alpha_scale_default": cfg.policy.intent_alpha_scale,
            "note": ("Blending is OFF by default: measured not to improve ranking "
                     "because the profile's recency decay already acts as a session "
                     "model. Detection still powers the 'current focus' label."),
        },
    }
    for path, key in ((Paths.eval_report, "evaluation"),
                      (Paths.artifacts / "ranker_report.json", "ranker_report"),
                      (Paths.artifacts / "intent_evaluation.json", "intent_evaluation"),
                      (Paths.artifacts / "objective_evaluation.json", "objective_evaluation")):
        if path.exists():
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict:
    return engine().recommend(
        history=req.history, watch_weights=req.watch_weights,
        seed_video=req.seed_video, query=req.query, n=req.n,
        mmr_lambda=req.mmr_lambda, exploration_slots=req.exploration_slots,
        max_per_channel=req.max_per_channel,
        objective_weights=req.objective_weights,
        intent_alpha_scale=req.intent_alpha_scale,
    ).to_dict()


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), n: int = Query(24, ge=1, le=60)) -> dict:
    return engine().search(q, n=n).to_dict()


@app.get("/api/similar/{video_id}")
def similar(video_id: str, n: int = Query(12, ge=1, le=40)) -> dict:
    eng = engine()
    if video_id not in eng.art.video_index:
        raise HTTPException(404, detail=f"unknown video_id {video_id}")
    return eng.similar(video_id, n=n).to_dict()


@app.get("/api/video/{video_id}")
def video(video_id: str) -> dict:
    eng = engine()
    if video_id not in eng.art.video_index:
        raise HTTPException(404, detail=f"unknown video_id {video_id}")
    idx = eng.art.idx(video_id)
    row = eng.art.catalog.row(idx)
    return {
        "video_id": str(row["video_id"]), "title": str(row["title"]),
        "channel_id": str(row["channel_id"]), "channel_title": str(row["channel_title"]),
        "category": str(row["category"]), "tags": str(row["tags"]).split("|"),
        "description": str(row["description"]),
        "view_count": int(row["view_count"]), "like_count": int(row["like_count"]),
        "comment_count": int(row["comment_count"]),
        "duration_seconds": int(row["duration_seconds"]),
        "published_at": str(row["published_at"])[:10],
        "age_days": round(float(eng.age_days[idx]), 1),
        "thumbnail_url": str(row["thumbnail_url"]),
    }


@app.get("/api/personas")
def personas() -> dict:
    """Pre-built watch histories so evaluators can see personalisation in one click.

    Built at request time from the live catalog rather than hard-coded ids, so
    they keep working when the catalog is rebuilt or swapped for real YouTube
    data.
    """
    art = engine().art
    catalog = art.catalog
    rng = np.random.default_rng(art.config.project.seed)

    def pick(category: str, n: int = 4) -> list[str]:
        subset = catalog[catalog["category"] == category]
        if subset.empty:
            return []
        # Mid-popularity videos: the very top would make every persona look
        # like the popularity baseline.
        ranked = subset.nlargest(min(120, len(subset)), "view_count")
        mid = ranked.iloc_slice(max(0, len(ranked) - 80), len(ranked))
        take = min(n, len(mid))
        if take <= 0:
            return []
        chosen = rng.choice(len(mid), size=take, replace=False)
        ids = mid["video_id"].values
        return [str(ids[int(i)]) for i in chosen]

    definitions = [
        ("cold_start", "Brand new viewer", "No history at all - shows the cold-start path", []),
        ("gamer", "Gaming enthusiast", "Deep in one vertical", pick("Gaming", 5)),
        ("foodie", "Home cook", "Food only - watch the filter bubble form", pick("Food", 5)),
        ("engineer", "Developer", "Science & Technology", pick("Science & Technology", 5)),
        ("split_taste", "Split interests", "Food AND Gaming - tests multi-interest handling",
         pick("Food", 3) + pick("Gaming", 3)),
        ("eclectic", "Eclectic viewer", "Five different categories - hardest case",
         [v for c in ["Music", "Travel", "Finance", "Education", "Autos & Vehicles"]
          for v in pick(c, 1)]),
    ]
    out = []
    for key, name, description, ids in definitions:
        out.append({
            "key": key, "name": name, "description": description,
            "video_ids": ids,
            "videos": [
                {"video_id": v,
                 "title": str(catalog["title"][art.idx(v)]),
                 "category": str(catalog["category"][art.idx(v)])}
                for v in ids
            ],
        })
    return {"personas": out}


@app.get("/api/catalog/sample")
def catalog_sample(n: int = Query(24, ge=1, le=60), category: str | None = None) -> dict:
    """Browsable slice of the catalog, for picking a starting video."""
    art = engine().art
    catalog = art.catalog
    if category:
        catalog = catalog[catalog["category"] == category]
        if catalog.empty:
            return {"items": []}
    elif len(catalog) == 0:
        return {"items": []}
    rng = np.random.default_rng(int(time.time()) % 100000)
    pool = art.catalog.top_by("view_count", min(600, len(catalog)),
                              subset=getattr(catalog, "index", None))
    take = min(n, len(pool))
    chosen = pool[rng.choice(len(pool), size=take, replace=False)] if take else []
    return {"items": [
        {"video_id": str(r["video_id"]), "title": str(r["title"]),
         "channel_title": str(r["channel_title"]), "category": str(r["category"]),
         "view_count": int(r["view_count"]), "like_count": int(r["like_count"]),
         "duration_seconds": int(r["duration_seconds"]),
         "published_at": str(r["published_at"])[:10],
         "thumbnail_url": str(r["thumbnail_url"])}
        for r in (art.catalog.row(int(i)) for i in chosen)
    ]}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

if Paths.static.exists():
    app.mount("/static", StaticFiles(directory=str(Paths.static)), name="static")


@app.get("/")
def index():
    page = Paths.static / "index.html"
    if not page.exists():
        return JSONResponse({"detail": "UI not built; API is at /docs"}, status_code=404)
    return FileResponse(page)


def main() -> None:
    import uvicorn
    cfg = load_config()
    uvicorn.run(app, host=cfg.serving.host,
                port=int(os.environ.get("PORT", cfg.serving.port)))


if __name__ == "__main__":
    main()
