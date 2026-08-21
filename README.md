# ReelRank — a YouTube-style video recommendation system

A three-stage recommender (candidate generation → ranking → policy) built from scratch,
with a YouTube-inspired testing UI that shows **why** every recommendation appeared.

**Live demo:** _<add your Hugging Face Space URL here>_ · **Docs:** [`docs/`](docs/) · **Course notes:** [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  6,000 videos                                                            │
│         │                                                                │
│  ┌──────▼──────────────────────────────────────────────┐                 │
│  │ STAGE 1 · CANDIDATE GENERATION      cheap, recall    │  ~3 ms          │
│  │  content embeddings · co-visitation · ALS ·          │                 │
│  │  channel affinity · trending    ── fused with RRF ── │                 │
│  └──────┬──────────────────────────────────────────────┘                 │
│         │ ~400 candidates                                                │
│  ┌──────▼──────────────────────────────────────────────┐                 │
│  │ STAGE 2 · RANKING                expensive, precision│  ~12 ms         │
│  │  gradient-boosted model, 19 features                 │                 │
│  │  odds ≈ E[watch time], not P(click)                  │                 │
│  └──────┬──────────────────────────────────────────────┘                 │
│         │ ~24 items                                                      │
│  ┌──────▼──────────────────────────────────────────────┐                 │
│  │ STAGE 3 · POLICY                    what a PAGE needs│  ~5 ms          │
│  │  MMR diversity · channel cap · freshness ·           │                 │
│  │  reserved exploration slots                          │                 │
│  └──────┬──────────────────────────────────────────────┘                 │
│         ▼   YouTube-style UI + "Why this video?" panel     p50 ≈ 22 ms    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Read this first: the interactions are simulated

There is **no public dataset of real YouTube watch histories**, and there never will be —
watch history is among the most re-identifying data a platform holds (the Netflix Prize
dataset was de-anonymised within weeks, and the sequel was cancelled).

So this project makes an explicit split:

| Component | Source | Real? |
|---|---|---|
| Video catalog (titles, channels, tags, views, likes, durations) | synthetic generator, or **real YouTube Data API v3 / Kaggle** with one config change | configurable |
| User watch logs | **simulated** from a documented behavioural model | **no** |
| Algorithms (ALS, co-visitation, LTR, MMR) | implemented from scratch | yes |

The simulator is not "fake data" — it is a hypothesis about user behaviour written in code
([`src/recsys/data/simulator.py`](src/recsys/data/simulator.py)), containing a cascade click
model, position bias, popularity bias, per-user mainstream-ness, duration preference and
taste drift. Its advantage over real logs is that **the latent structure is known**, which
lets us ask a question real data cannot answer: *did the model recover genuine preference
structure, or just memorise popularity?*

**What the metrics in this repo mean:** they show the algorithms recover the latent structure
that generated the data. They are **not** a prediction of real-world YouTube performance.
No offline metric ever is. See [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
python scripts/build_all.py
```

```bash
python -m uvicorn recsys.api.app:app --app-dir src --port 7860
```

Then open <http://localhost:7860>. The full build takes ~7 minutes. For a 1-minute smoke build
use `python scripts/build_all.py --quick`.

**Reproducibility:** the catalog dates publish times relative to "now", and the simulator places
sessions in a window ending at "now" — so by default a rebuild on a different day produces
different (statistically equivalent) data. Set `project.reference_date` in `config.yaml` to pin
that instant and the whole pipeline becomes **byte-identical** across machines and across days
(asserted by `test_build_is_byte_identical_with_a_pinned_reference_date`). It is left unpinned by
default because a live demo should have "trending" mean trending *now*.

**To use real YouTube data instead of synthetic:** get a free
[YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started) key, then

```bash
YOUTUBE_API_KEY=your_key python scripts/build_all.py --source youtube_api
```

Nothing else changes — every source is normalised to the same schema at the boundary
([`src/recsys/data/schema.py`](src/recsys/data/schema.py)).

---

## Results

Measured on a **global temporal split** (train on the past, test on the future), 800 held-out
users, 6,000-video catalog. Every model upstream of the ranker respects the same cutoff.

### The headline finding: the standard offline protocol is invalid here

The usual recipe — hide future clicks, retrieve from the full catalog, report NDCG — silently
measures **the logging policy**, not the recommender. We proved it rather than asserting it,
by scoring an **oracle** built from the simulator's own generative parameters:

| Full-catalog retrieval | NDCG@10 |
|---|---|
| popularity baseline | **0.0218** |
| **ORACLE** (true user personas + true hidden video quality) | 0.0198 ❌ |

**The data-generating process itself loses to a popularity list.** A user can only click what
they were shown; the incumbent policy was popularity-heavy, so a better recommender is
*penalised* for surfacing something the user never got the chance to click. Any conclusion
drawn from full-catalog NDCG on logged data is unsafe.

### Protocol A — re-ranking logged impressions (counterfactually valid)

Re-orders only the videos users were **actually shown**, so every label is observed.
This is the offline protocol that best predicts online lift.

| Scorer | top-1 | NDCG | MRR |
|---|---|---|---|
| `[ref] shown position` † | 0.7308 | 0.8908 | 0.8531 |
| **LEARNED RANKER (19 features)** | **0.2240** | **0.5787** | **0.4473** |
| content only | 0.2108 | 0.5704 | 0.4364 |
| CF — ALS only | 0.1818 | 0.5425 | 0.4018 |
| CF — co-visitation only | 0.1478 | 0.5032 | 0.3532 |
| popularity | 0.1385 | 0.4998 | 0.3482 |
| random | 0.1202 | 0.4854 | 0.3302 |

† *Not a competing model.* Position **causes** clicks under a cascade click model, so this
row measures the size of position bias — the ceiling on what re-ranking could ever be worth —
not video quality.

### Protocol B — 1 held-out positive vs 100 sampled negatives

| Scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.4842** | 0.2469 | 24.17 |
| **LEARNED RANKER** | 0.4581 | 0.2368 | **22.12** |
| CF — ALS only | 0.3387 | 0.1784 | 32.88 |
| popularity | 0.2536 | 0.1354 | 36.90 |
| CF — co-visitation only | 0.1494 | 0.0821 | 48.15 |
| random | 0.0929 | 0.0419 | 52.13 |

The hybrid ranker wins on Protocol A and takes the best mean rank on Protocol B.
Sampled metrics are known to flatter simpler models ([Krichene & Rendle, KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403226)),
which is why both are reported and neither alone.

### Beyond accuracy

| Strategy | coverage | Gini (exposure) | novelty (bits) | intra-list diversity | p50 latency |
|---|---|---|---|---|---|
| popularity | 0.005 | 0.997 | 9.17 | 0.938 | 0.9 ms |
| content only | 0.760 | 0.565 | 12.77 | 0.531 | 0.6 ms |
| CF — ALS only | 0.490 | 0.777 | 11.33 | 0.844 | 0.4 ms |
| **FULL pipeline** | **0.466** | 0.809 | 11.87 | 0.713 | 22.4 ms |

The popularity baseline reaches **0.5% catalog coverage with a Gini of 0.997** — it shows
essentially the same 30 videos to everyone. That is what accuracy metrics alone will not tell you,
and why coverage/Gini/novelty are treated as first-class here.

### Ranker

`AUC 0.697` · `watch-time-weighted AUC 0.790` · `within-feed top-1 29.8%` (random 14.3%) ·
824k training rows · 19 features · trained on cross-fitted CF features.
Top features by permutation importance: `content_sim_profile`, `category_affinity`,
`log_duration_min`, `log_views`, `engagement_rate`.

---

## Three bugs worth reading about

Real engineering happened here. Each of these is documented in the source where it lives.

**1 · Leakage through a stacked model** → [`src/recsys/rank/crossfit.py`](src/recsys/rank/crossfit.py)
The first ranker scored **AUC 0.957** and `als_score` carried the entire model. ALS had been
trained on the full log, so item factors encoded the very clicks the ranker was asked to
predict. Enforcing one global temporal cutoff dropped it to 0.578 — *below its own best single
feature*, because training rows were still in-sample for ALS (0.856 in-sample vs 0.584 held out).
Fixed with **user-fold cross-fitting**: 0.697, and the model finally beats every input.
*Generalises past this repo:* any stacked model's upstream features must be out-of-fold.

**2 · The evaluation protocol, not the model** → [`src/recsys/counterfactual.py`](src/recsys/counterfactual.py)
Covered above. The oracle control is the part most tutorials skip.

**3 · A 70× performance bug that was not in my code** → [`src/recsys/recall/cf.py`](src/recsys/recall/cf.py)
ALS took 649 s. Profiling showed a single 96×96 `np.linalg.solve` costing **3.65 ms** — about
0.16 GFLOPS. The cause was OpenBLAS spawning and synchronising a thread team for a matrix far
too small to amortise it. Pinning to one thread: **51 µs**, and ALS finished in **18 s**.

---

## Repository map

```
config.yaml              every tunable knob; the build is reproducible from this + the seed
Dockerfile               ~450MB image, no torch — models are NumPy/sklearn or hand-written
scripts/
  build_all.py           one command, five stages
  01_build_data.py       catalog + simulated watch log
  02_build_features.py   TF-IDF → SVD text vectors, item stats
  03_train_cf.py         co-visitation + implicit ALS (respects the temporal cutoff)
  04_train_ranker.py     learning-to-rank, cross-fitted
  05_evaluate.py         baselines, ablation, and both counterfactual protocols
  06_validate_on_movielens.py  the same CF code on 100k REAL human ratings
  07_ablate_text_fields.py     which text fields to index (measured, not guessed)
  08_diagnose_ranker.py        is the ranker weak, or is the task near its noise ceiling?
  09_test_cases.py             runs every scenario in docs/TEST_CASES.md, live
src/recsys/
  data/       schema · topics · synthetic · simulator · youtube_api · kaggle_loader
  features/   text.py           TF-IDF+SVD (LSA), optional sentence-transformers
  recall/     content · cf (ALS + co-visitation, from scratch) · heuristic · blend (RRF)
  rank/       features · dataset (causal replay) · crossfit · ranker
  policy/     rerank.py         MMR, channel cap, freshness, exploration slots
  engine.py   orchestration + explanations
  evaluate.py · counterfactual.py · metrics.py · split.py
  api/        FastAPI + the YouTube-style UI (vanilla JS, no build step)
tests/        40 tests; the interesting ones are regressions for real bugs
```

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | system design, data flow, serving, latency budget |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | every algorithm and why it was chosen over the alternative |
| [DATASET.md](docs/DATASET.md) | data sources, the simulator, and its parameters |
| [EVALUATION.md](docs/EVALUATION.md) | protocols, metrics, full results, and why offline ≠ online |
| [TEST_CASES.md](docs/TEST_CASES.md) | worked success **and failure** scenarios |
| [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | every assumption made, and what breaks if it is wrong |
| [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | decisions with the rejected alternatives |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | what this system cannot do |
| [COMPARISON.md](docs/COMPARISON.md) | benchmarked against real YouTube |
| [FUTURE_WORK.md](docs/FUTURE_WORK.md) | what I would build next, in priority order |
| [LEARNING_NOTES.md](docs/LEARNING_NOTES.md) | the full course: theory behind every module |
| [DEPLOY.md](docs/DEPLOY.md) | Hugging Face Spaces, Docker, and alternatives |

## Tech stack

Python 3.10+ · NumPy · pandas · scikit-learn · SciPy · FastAPI · Uvicorn · vanilla JS · Docker

**No `implicit`, no `faiss`, no `lightgbm`, no PyTorch.** ALS, the co-visitation index, the
nearest-neighbour search and MMR are all hand-written — which is the point pedagogically, and
also keeps the deployed image at ~450 MB instead of ~2.5 GB.

## License

MIT
