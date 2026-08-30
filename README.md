# ReelRank — a YouTube-style video recommendation system

A three-stage recommender (candidate generation → ranking → policy) built from scratch,
with a YouTube-inspired testing UI that shows **why** every recommendation appeared.

**Live demo:** <https://reelrank-one.vercel.app> · **Docs:** [`docs/`](docs/) · **Course notes:** [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md)

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

Measured on a **global temporal split** (train on the past, test on the future), 1,500 held-out
users, 6,000-video catalog. Every model upstream of the ranker respects the same cutoff.

### The headline finding: full-catalog NDCG is the wrong metric here

The usual recipe — hide future clicks, retrieve from the full catalog, report
NDCG — is dominated by retrieval breadth and popularity, not ranking quality.
The cleanest evidence is that **adding the learned ranker moves the two families
of metric in opposite directions**:

| | full-catalog NDCG@10 | Protocol A top-1 |
|---|---|---|
| Stage 1 recall only | **0.0125** | 0.1840 † |
| + learned ranker (shipped) | 0.0107 ↓ | **0.1930** ↑ |

† *content-only, the strongest single Stage-1 signal.*

Two metrics disagreeing about the same change means one of them is wrong **for
this purpose**. The full-catalog metric rewards casting a wide net; a popularity
list scores 0.0121, beating content-based retrieval (0.0090) outright.

An **oracle** built from the simulator's own generative parameters scores 0.0161
— so the metric is not pure noise, and there is real headroom our models do not
capture. But the ranking that wins on it is not the ranking that wins on
observed labels, which is why the counterfactual protocols below are the ones
quoted.

> On an earlier build (before latent clickbait was added to the generator) the
> oracle scored *below* the popularity baseline outright. Both versions point the
> same way; the current numbers are the ones that reproduce.

### Protocol A — re-ranking logged impressions (counterfactually valid)

Re-orders only the videos users were **actually shown**, so every label is observed.
This is the offline protocol that best predicts online lift.

| Scorer | top-1 | NDCG | MRR |
|---|---|---|---|
| `[ref] shown position` ‡ | 0.7713 | 0.9086 | 0.8768 |
| **LEARNED RANKER (19 features)** | **0.1930** | **0.5562** | **0.4183** |
| content only | 0.1840 | 0.5514 | 0.4119 |
| CF — ALS only | 0.1622 | 0.5259 | 0.3804 |
| random | 0.1415 | 0.5018 | 0.3507 |
| popularity | 0.1323 | 0.4949 | 0.3417 |
| CF — co-visitation only | 0.1280 | 0.4928 | 0.3392 |

‡ *Not a competing model.* Position **causes** clicks under a cascade click model, so this
row measures the size of position bias — the ceiling on what re-ranking could ever be worth —
not video quality. Note popularity and co-visitation now score *below random* here: on a
per-page basis, "show the globally popular thing" is actively worse than chance.

### Protocol B — 1 held-out positive vs 100 sampled negatives

| Scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.3703** | **0.1821** | 30.91 |
| **LEARNED RANKER** | 0.3567 | 0.1776 | **28.49** |
| CF — ALS only | 0.2792 | 0.1484 | 37.24 |
| popularity | 0.2665 | 0.1484 | 36.29 |
| CF — co-visitation only | 0.1428 | 0.0758 | 48.75 |
| random | 0.0999 | 0.0450 | 51.27 |

The hybrid ranker wins Protocol A outright and takes the best mean rank on Protocol B, where
content-only edges it on HR@10.
Sampled metrics are known to flatter simpler models ([Krichene & Rendle, KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403226)),
which is why both are reported and neither alone.

### Beyond accuracy

| Strategy | coverage | Gini (exposure) | novelty (bits) | intra-list diversity | p50 latency |
|---|---|---|---|---|---|
| popularity | 0.004 | 0.997 | 9.69 | 0.863 | 0.7 ms |
| content only | 0.890 | 0.509 | 12.78 | 0.536 | 0.5 ms |
| CF — ALS only | 0.572 | 0.764 | 11.41 | 0.855 | 0.3 ms |
| **FULL pipeline** | **0.564** | 0.798 | 12.04 | 0.718 | 15.0 ms |

The popularity baseline reaches **0.5% catalog coverage with a Gini of 0.997** — it shows
essentially the same 30 videos to everyone. That is what accuracy metrics alone will not tell you,
and why coverage/Gini/novelty are treated as first-class here.

### Ranker

`AUC 0.663` · `watch-time-weighted AUC 0.755` · `within-feed top-1 25.8%` (random 14.3%) ·
878k training rows · 19 features · trained on cross-fitted CF features.
Top features by permutation importance: `log_duration_min`, `category_affinity`,
`content_sim_profile`, `log_views`, `engagement_rate`.
Plus six multi-objective heads — see [INTENT_AND_OBJECTIVES.md](docs/INTENT_AND_OBJECTIVES.md).

---

## Two product hypotheses, tested rather than assumed

Full write-up in [INTENT_AND_OBJECTIVES.md](docs/INTENT_AND_OBJECTIVES.md).

**Session intent** — *"what does this person want right now?"* rather than
*"what do they generally like?"*. Blending a session vector into the query helps
exactly the cohort it should (**+7.5%** on focused, off-persona sessions) and
hurts on browsing sessions (**−3.4%**). Net across all sessions: **−0.1%**.

The explanation is the payoff. The profile is already recency-weighted with a
half-life of 8 positions, so `cos(profile, session vector) = 0.803` — **recency
decay is already a soft session model.** Remove it and session blending suddenly
works (+2.6% on a uniform-mean profile). So: shipped for *explainability* (the
UI names your current focus), with blending off by default and exposed as a
slider so the negative result is demonstrable.

**Multi-objective ranking** — six calibrated heads (click / long-watch /
completion / liked / satisfied / dismissed), combined by weights chosen **per
request**. Completion@1 nearly doubles (0.0055 → 0.0100, **+82%**) and the
Recommendation Lab can switch the system's objective with no retraining.

But it does *not* reduce clickbait exposure, and the reason is the useful part:

| target | GBDT R² from all item features |
|---|---|
| latent quality | **0.6355** |
| latent clickbait | **−0.1122** |

**Clickbait is invisible to the feature set**, so no ranker can optimise it away.
A multi-objective ranker can only trade off objectives it can *observe* — which
is precisely why YouTube collects user surveys instead of inferring satisfaction
from engagement metadata. That makes the next step a *feature* problem, not a
model problem.

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
  10_evaluate_intent.py        does session intent improve ranking? (no -- and why)
  11_evaluate_objectives.py    CTR vs watch-time vs satisfaction vs multi-objective
src/recsys/
  data/       schema · topics · synthetic · simulator · youtube_api · kaggle_loader
  features/   text.py           TF-IDF+SVD (LSA), optional sentence-transformers
  recall/     content · cf (ALS + co-visitation, from scratch) · heuristic · blend (RRF)
  rank/       features · dataset (causal replay) · crossfit · ranker
  policy/     rerank.py         MMR, channel cap, freshness, exploration slots
  intent.py   session intent detection (explanation; blending measured off)
  rank/multitask.py  six calibrated heads, weights chosen per request
  engine.py   orchestration + explanations
  evaluate.py · counterfactual.py · metrics.py · split.py
  api/        FastAPI + the YouTube-style UI (vanilla JS, no build step)
tests/        45 tests; the interesting ones are regressions for real bugs
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
| [INTENT_AND_OBJECTIVES.md](docs/INTENT_AND_OBJECTIVES.md) | session intent + multi-objective ranking: two hypotheses, tested |
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
