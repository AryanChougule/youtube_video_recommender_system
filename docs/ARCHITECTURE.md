# Architecture

## Problem statement

Given a viewer — identified only by what they have watched in this browser session — select
~24 videos from a 6,000-item catalog that maximise **expected watch time**, subject to the
page being diverse, containing fresh material, and remaining explainable. Serve in under
100 ms with no GPU.

Three sub-problems fall out of that, and they have genuinely different shapes:

1. **Retrieval** — 6,000 candidates must become ~400, fast, without losing the good ones.
2. **Ranking** — those ~400 must be ordered precisely, which can afford real computation.
3. **Page construction** — the best *set* is not the top-N items scored independently.

Conflating these is the most common architectural mistake in recommender systems. A single
model that tries to do all three either cannot scale (too expensive per item) or cannot be
precise (too cheap per item).

## The two-stage pattern

This follows Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations*
(RecSys 2016), with an explicit third policy stage.

```
                         ┌───────────────────────────┐
   request ──────────────►  history / seed / query   │
   {history[], query?}   └────────────┬──────────────┘
                                      │
   ═══════════ STAGE 1 · CANDIDATE GENERATION ═══════════   budget: ~5 ms
                                      │
        ┌────────────┬────────────┬───┴────────┬────────────┐
        ▼            ▼            ▼            ▼            ▼
   content      co-visitation    ALS       channel      trending
   TF-IDF+SVD   S·Sᵀ counts    implicit    affinity    velocity·decay
   256-d cosine  damped norm   96 factors   grouped     freshness
        │            │            │            │            │
        └────────────┴─────┬──────┴────────────┴────────────┘
                           ▼
              Reciprocal Rank Fusion   (fuses RANK, not score)
                           │
                      ~400 candidates
                           │
   ═══════════════ STAGE 2 · RANKING ═══════════════════════  budget: ~30 ms
                           │
              FeatureBuilder  →  19 features per (user, item)
                           │      shared by training AND serving
              HistGradientBoosting, positives weighted by watch time
                           │
                  score = odds = P/(1−P) ≈ E[watch time]
                           │
   ═══════════════ STAGE 3 · POLICY ════════════════════════  budget: ~10 ms
                           │
        freshness boost → MMR (category-aware) + channel cap
                        → reserved exploration slots
                           │
                     24 items + provenance
                           ▼
                  JSON  {items, stages, diagnostics}
```

## Why each recall source exists

Each one covers a hole the others leave. This table is the justification for the whole Stage 1
design — if two rows had the same failure mode, one of them should be deleted.

| Source | Retrieves by | Fails when |
|---|---|---|
| Content (TF-IDF+SVD) | semantic text similarity | filter bubble; cannot cross vocabulary gaps |
| Co-visitation | same-session co-watch | needs ≥2 co-watches; 2,282 of 6,000 items have no neighbours |
| ALS | latent factor affinity | needs a trained catalog; weak on cold items |
| Channel affinity | creator loyalty | definitionally incapable of discovery |
| Trending | velocity × freshness | not personalised at all |

Measured cross-category escape rate (top-10 neighbours leaving the seed's category, while
staying on-topic by ground-truth similarity):

| | escapes category | still on-topic |
|---|---|---|
| Content | 1.5% | 0.904 |
| Co-visitation | 38.6% | 0.544 |
| **ALS** | **12.3%** | **0.782** |

That is the filter bubble, quantified. Content essentially never leaves the category.
ALS is the sweet spot — it crosses categories 8× more often than content while staying
genuinely on-topic, recovering the latent "bridge" topics **from behaviour alone, never
having seen the text**.

## Serving

**Stateless.** Watch history lives in the browser and is posted with every request. No
sessions, no database. The container scales horizontally and a hard refresh loses nothing.

**Load once.** All artifacts (6 MB of vectors, 4 MB of factors, the ranker) load at startup
in ~1.3 s. Nothing touches disk per request.

**Cold users are a solved problem, cold items are not.** A brand-new user gets a latent vector
via ALS **fold-in** — one 96×96 closed-form solve (~51 µs) against frozen item factors, no
retraining. A brand-new *item* has no factors and no co-visitation neighbours, and is reachable
only through content similarity and trending. That asymmetry is inherent to collaborative
filtering.

**Measured latency** (p50, 6k catalog, 24 results):

| Stage | ms | share |
|---|---|---|
| Stage 1 recall | ~3 | 14% |
| Stage 2 ranking | ~12 | 55% |
| Stage 3 policy | ~5 | 23% |
| **total** | **~22** | |

Ranking dominates, as it should — that is where the budget is deliberately spent.

### Why brute-force nearest neighbours

6,000 items × 256 dims = 1.5 M multiply-adds, which NumPy completes in well under a
millisecond. An approximate index (FAISS/HNSW) starts paying for itself around 10⁶ items and
would cost exactness plus a heavyweight dependency. Use the simple thing until the numbers say
otherwise; the numbers here say otherwise at roughly 100× the current catalog size.

## Train / serve separation

The single most important rule in the codebase:

> **Encode offline. Serve with a dot product.**

The TF-IDF+SVD pipeline is fitted at build time and written to `artifacts/`. Serving is pure
NumPy. This is why the Docker image is ~450 MB rather than ~2.5 GB, and why the optional
`sentence-transformers` upgrade costs nothing at serving time — its embeddings become the same
`.npy` matrix.

The corresponding hazard is **train/serve skew**: training and serving computing "the same"
feature slightly differently. The defence is structural — one `FeatureBuilder`
([`src/recsys/rank/features.py`](../src/recsys/rank/features.py)) used by both paths, and every
feature computable from a `(history, candidate)` pair alone so nothing depends on having
executed retrieval.

## Artifact integrity

Every artifact is indexed by **catalog row order**. If that ever drifts, the system does not
crash — it silently recommends the wrong videos forever, which is far worse. So
[`artifacts.py`](../src/recsys/artifacts.py) asserts at load time that every matrix has the
same row count as the catalog and that the first/last video ids still match what was recorded
at training time, and refuses to start otherwise.

`index_meta.json` also records the temporal cutoff the CF models were fitted under; the ranker
script refuses to run if its own cutoff disagrees. That check exists because the mismatch it
guards against silently invalidated an entire evaluation once already.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/recommend` | main entry: history / seed / query → ranked items + provenance |
| `GET /api/search?q=` | semantic search, personalised by history |
| `GET /api/similar/{id}` | watch-page rail, no personalisation |
| `GET /api/video/{id}` | full metadata for the detail view |
| `GET /api/personas` | prebuilt histories, generated live from the catalog |
| `GET /api/catalog/sample` | browsable slice for picking a starting video |
| `GET /api/meta` | model config + evaluation report, powers "How it works" |
| `GET /api/health` | readiness; reports artifact errors instead of dying |

Every recommendation response carries `stages` (per-stage timings, candidate counts per source)
and `diagnostics` (intra-list diversity, novelty, distinct categories/channels). The brief asks
evaluators to understand *why* a recommendation appeared, so provenance ships in the payload
rather than living in a server log they cannot see.
