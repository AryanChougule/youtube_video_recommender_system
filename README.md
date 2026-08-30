# ReelRank — Multi-Objective YouTube-Style Recommendation System

A three-stage video recommender (candidate generation → ranking → policy), built from
scratch, that ranks against **six objectives instead of one** and shows *why* every
recommendation appeared.

| | |
|---|---|
| **Live demo** | **<https://reelrank-one.vercel.app>** — no setup, no login |
| **Repository** | <https://github.com/AryanChougule/youtube_video_recommender_system> |
| **Docs** | [`docs/`](docs/) · [full course notes](docs/LEARNING_NOTES.md) |
| **Tests** | 61 passing (`python -m pytest tests/ -q`) |

---

## Try it in 60 seconds

No installation. Open the [live demo](https://reelrank-one.vercel.app) and:

1. **Click a persona** in the left sidebar — *Gaming enthusiast*, *Home cook*, *Split
   interests*. The feed re-ranks immediately and every card explains itself.
2. **Click the 💡 line on any card** → the *"Why this video?"* panel opens, showing which
   recall source proposed it and at what rank, all six objective probabilities with their
   weights and contributions, the ranker score, and which policy rules fired.
3. **Click "Max satisfaction"** in the Recommendation Lab → the objective changes and the
   feed re-ranks live, with **no retraining**. Compare against "Max engagement".
4. **Drag the MMR slider to 1.0** → watch diversity collapse into a monoculture. The
   controls are real, not decorative.
5. **Search `brown butter`** (works) then **`machine learning`** (nothing in this catalog
   matches) → the second says so explicitly instead of faking results. See [F10](docs/TEST_CASES.md#f10--out-of-vocabulary-search-retrieves-nothing).

---

## What this project does

Given a viewer's watch history, ReelRank returns a ranked, diversified page of videos and
an explanation for each one. It implements the two-stage architecture YouTube described in
[Covington, Adams & Sargin (RecSys 2016)](https://research.google/pubs/pub45530/) — cheap
high-recall candidate generation, then an expensive precision ranker — plus a third policy
stage for the things a *page* needs that a per-item score cannot express.

Every algorithm is written from scratch: ALS, the co-visitation index, RRF fusion,
nearest-neighbour search, MMR. **No `implicit`, no `faiss`, no `lightgbm`, no PyTorch.**

## Why multi-objective ranking?

A recommender optimised for clicks learns to produce clickbait, because clickbait is what
maximises clicks. Optimising watch time instead is a real improvement — and still not
enough. Measured on this project's own generated log:

```
top-decile clickbait : watch 0.425, satisfied 42%
bottom-half clickbait: watch 0.438, satisfied 73%
```

**Watch time is nearly blind to clickbait** (correlation +0.01): it holds attention slightly
*and* draws in lower-affinity viewers, and the two effects cancel. Satisfaction is not blind
to it at all. So a watch-time-only objective cannot tell a genuinely good video from one
that kept you waiting for a payoff that never came.

That gap is the entire justification for six heads. If watch time and satisfaction agreed,
this would be complexity for its own sake.

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

**What the metrics here mean:** they show the algorithms recover the latent structure that
generated the data. They are **not** a prediction of real-world YouTube performance. No
offline metric ever is. See [EVALUATION.md](docs/EVALUATION.md).

---

## Key features

| Feature | Where |
|---|---|
| Two-stage retrieval + ranking, plus a policy stage | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Five recall sources fused by Reciprocal Rank Fusion | [`recall/`](src/recsys/recall/) |
| Six-objective ranking, weights chosen **per request** | [`rank/multitask.py`](src/recsys/rank/multitask.py) |
| Per-item explanations covering all three stages | [`engine.py`](src/recsys/engine.py) |
| MMR diversity, channel caps, freshness, exploration slots | [`policy/rerank.py`](src/recsys/policy/rerank.py) |
| Cold-start for new **users** via ALS fold-in (~51 µs) | [`recall/cf.py`](src/recsys/recall/cf.py) |
| Session-intent detection | [`intent.py`](src/recsys/intent.py) |
| Counterfactual evaluation with an oracle control | [`counterfactual.py`](src/recsys/counterfactual.py) |
| Leakage-free training via user-fold cross-fitting | [`rank/crossfit.py`](src/recsys/rank/crossfit.py) |
| NumPy-only serving runtime (no scikit-learn at inference) | [`serving/`](src/recsys/serving/) |

---

## System architecture

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
│  │  gradient-boosted models, 19 features                │                 │
│  │  odds ≈ E[watch time] · six objective heads          │                 │
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

Stage 1 answers *"which 400 of 6,000 videos are worth scoring?"* — five sources, each with a
different failure mode, fused by rank rather than score (scores from ALS and TF-IDF are not
comparable; ranks are). Stage 2 answers *"of those, which are best for this person?"*
Stage 3 answers *"what makes a good page?"* — a question no per-item score can answer, since
relevance is a property of an item and diversity is a property of a set.

Full detail, including the latency budget: [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Recommendation pipeline

```
user history + candidate item
            ↓
   19 engineered features           shared FeatureBuilder, train and serve
            ↓
   6 independent HistGradientBoostingClassifier heads
            ↓
   6 calibrated probabilities       P(click), P(long_watch), P(completion),
                                    P(liked), P(satisfied), P(dismissed)
            ↓
   objective weighting              value = Σ wₖ · Pₖ(x)      weights per request
            ↓
   final ranking score
            ↓
   policy re-rank → ranked page
```

### Multi-objective ranking

Six binary classifiers over **one shared feature matrix**:

| head | predicts | default weight |
|---|---|---|
| `click` | did they open it | +0.10 |
| `long_watch` | watched ≥ 50% | +0.25 |
| `completion` | watched ≥ 90% | +0.15 |
| `liked` | explicit positive | +0.15 |
| `satisfied` | the survey-like signal | **+0.40** |
| `dismissed` | bounced almost immediately | **−0.20** |

These weights are an **engineering choice, not YouTube's numbers** — YouTube has never
published theirs. They encode a product stance: a click is worth little on its own, a
satisfying watch is worth most, and bouncing is actively bad.

**Why one shared feature matrix?** All six questions are asked about the same
(user, item) pair, so the same 19 features are sufficient statistics for all of them.
Building six matrices would cost six times as much for identical numbers, and Stage 2 has a
~12 ms budget.

**Why independently trained?** Each head fits its own label with its own positive rate,
which ranges from 14.1% (`click`) to 0.44% (`dismissed`). Joint training would need a
shared loss and a weighting between tasks — one more thing to tune, with no mechanism to
share representation in a GBDT anyway.

**Why calibrated probabilities?** Because the weights must be *interpretable*. `0.40 ×
P(satisfied)` only means something if `P(satisfied)` is a probability. An uncalibrated
margin cannot be meaningfully multiplied by a weight or compared across heads — and the
whole Recommendation Lab depends on an evaluator being able to reason about what moving a
slider should do.

**How dismissal enters:** with a negative weight, so a high `P(dismissed)` subtracts. It is
the only head that can *demote*, which is what makes the objective a genuine trade-off
rather than a sum of goods.

**The trade-off, stated honestly.** Six independent GBDTs are simpler, faster and far more
interpretable than a shared-trunk multi-gate Mixture-of-Experts
([Zhao et al., 2019](https://dl.acm.org/doi/10.1145/3298689.3346997)), which is what a
platform at YouTube's scale would use. The cost is real: independent heads **cannot share
representation**, so correlated tasks re-learn the same structure six times. That is
wasteful, not wrong. At 6k items with one simulated label stream per outcome, an MoE would
be capacity we cannot feed — this is a deliberate choice for the data and scope available,
not a claim to be state of the art. Upgrade path: [FUTURE_WORK.md](docs/FUTURE_WORK.md).

Full write-up: [INTENT_AND_OBJECTIVES.md](docs/INTENT_AND_OBJECTIVES.md).

---

## Dataset

6,000 videos · 3,831 users · 1,142,312 impressions · 124,097 clicks · CTR 10.86% ·
27,043 sessions · matrix density 0.54%.

The catalog has 40 latent micro-topics mapped onto 13 overlapping categories, with 15
bridge topics that belong to more than one — which is what makes cross-category
recommendation possible at all. Two hidden generative variables, `latent_quality` and
`latent_clickbait`, drive behaviour and are **structurally excluded** from the serving
bundle so leaking them is impossible rather than merely forbidden.

Swap in real data with one flag — every source normalises to the same schema
([`data/schema.py`](src/recsys/data/schema.py)):

```bash
YOUTUBE_API_KEY=your_key python scripts/build_all.py --source youtube_api
```

Full detail: [DATASET.md](docs/DATASET.md).

## Feature engineering

19 features across four families, built by a **single shared `FeatureBuilder`** used by both
training and serving — the standard defence against train/serve skew.

| Family | Examples |
|---|---|
| Content match | `content_sim_profile`, `content_sim_max_hist`, `content_sim_recent` |
| Collaborative | `als_score`, `covisit_score` (both cross-fitted) |
| Affinity | `category_affinity`, `channel_affinity`, `channel_seen` |
| Item quality / context | `log_views`, `engagement_rate`, `like_rate`, `log_duration_min`, `is_short`, `log_age_days`, `duration_fit`, `views_vs_user` |

Top five by permutation importance: `log_duration_min` (0.0506), `category_affinity`
(0.0451), `content_sim_profile` (0.0424), `log_views` (0.0109), `engagement_rate` (0.0089).

## Model architecture

| Component | Model | Notes |
|---|---|---|
| Text embeddings | TF-IDF → TruncatedSVD (LSA), 256-d | fitted offline, served as a matrix |
| Item-item CF | co-visitation, `C = SᵀS`, popularity-damped | hand-written sparse matmul |
| User-item CF | implicit ALS, 96 factors, 18 iterations | hand-written; fold-in for cold users |
| Ranker | HistGradientBoostingClassifier, 101 trees | watch-time-weighted → odds ≈ E[watch time] |
| Objective heads | 6 × HistGradientBoostingClassifier | shared features, independent fits |
| Fusion | Reciprocal Rank Fusion, K=60 | fuses rank, not score |
| Diversity | MMR with category-aware similarity | Carbonell & Goldstein, 1998 |

---

## Evaluation

Measured on a **global temporal split** (train on the past, test on the future), 1,500
held-out users, 6,000-video catalog. Every model upstream of the ranker respects the same
cutoff. Full detail and protocol definitions: [EVALUATION.md](docs/EVALUATION.md).

### Classification metrics (do we separate the labels?)

`AUC 0.6627` · `watch-time-weighted AUC 0.7553` · `log loss 0.4026` ·
`within-feed top-1 25.8%` (random 14.3%) · 877,630 rows · 19 features · 4-fold cross-fitted.

### Ranking metrics (does the right video reach the top of a page?)

**Protocol A — re-ranking logged impressions.** Re-orders only videos users were actually
shown, so every label is observed. This is the protocol that best predicts online lift.

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
row measures the size of position bias — the ceiling on what re-ranking could be worth.
Note popularity and co-visitation score *below random*: per page, "show the globally popular
thing" is actively worse than chance.

**Protocol B — 1 held-out positive vs 100 sampled negatives.**

| Scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.3703** | **0.1821** | 30.91 |
| **LEARNED RANKER** | 0.3567 | 0.1776 | **28.49** |
| CF — ALS only | 0.2792 | 0.1484 | 37.24 |
| popularity | 0.2665 | 0.1484 | 36.29 |
| CF — co-visitation only | 0.1428 | 0.0758 | 48.75 |
| random | 0.0999 | 0.0450 | 51.27 |

The hybrid wins Protocol A outright and takes the best mean rank on Protocol B, where
content-only edges it on HR@10. Sampled metrics are known to flatter simpler models
([Krichene & Rendle, KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403226)), which is
why both are reported and neither alone.

### Why AUC alone is not enough — the clearest result in the project

Each objective head, used **alone** as a ranker:

| head | AUC | top-1 | NDCG | MRR |
|---|---|---|---|---|
| `satisfied` | 0.7376 | **0.1982** | **0.5636** | **0.4272** |
| `click` | 0.6627 | 0.1933 | 0.5565 | 0.4187 |
| `liked` | 0.7436 | 0.1895 | 0.5559 | 0.4176 |
| `long_watch` | 0.8124 | 0.1858 | 0.5524 | 0.4134 |
| `completion` | 0.9154 | 0.1768 | 0.5389 | 0.3969 |
| `dismissed` | **0.9154** | **0.1005** | 0.4667 | 0.3061 |

**`dismissed` has the joint-highest AUC and by far the worst ranking — 0.1005 top-1, below
random (0.1415).** It separates its rare label (0.44% positive) almost perfectly while being
nearly useless for ordering a page. The metrics ask different questions: AUC asks "across
all pairs, does a positive outrank a negative?", which a confident majority-class predictor
aces at 0.44% prevalence. Top-1 asks "on this page of ~8 candidates, is the item you put
first the one they engaged with?" **A recommender is only ever graded on pages.** Reporting
AUC alone would have made `dismissed` look like the best head in the system.

### Objective comparison

| configuration | top-1 | NDCG | MRR | NDCG(satisfaction) | completion@1 | clickbait@1 |
|---|---|---|---|---|---|---|
| A. CTR-optimised | 0.1983 | 0.5618 | 0.4250 | 0.3857 | 0.0055 | 0.2057 |
| B. Watch-time optimised | 0.1933 | 0.5565 | 0.4187 | 0.3801 | 0.0035 | 0.2086 |
| C. Satisfaction-only | 0.1973 | **0.5643** | **0.4280** | **0.3897** | 0.0095 | **0.2026** |
| D. Multi-objective (shipped) | 0.1963 | 0.5638 | 0.4273 | 0.3893 | **0.0100** | 0.2044 |

Multi-objective ranking **nearly doubles completion@1 (0.0055 → 0.0100, +82%)** at
essentially no cost in top-1, and makes the objective switchable per request. What it does
*not* do is reduce clickbait exposure — see [F9](docs/TEST_CASES.md) and *Limitations*.

### Beyond accuracy

| Strategy | coverage | Gini (exposure) | novelty (bits) | intra-list diversity | p50 latency |
|---|---|---|---|---|---|
| popularity | 0.004 | 0.997 | 9.69 | 0.863 | 0.7 ms |
| content only | 0.890 | 0.509 | 12.78 | 0.536 | 0.5 ms |
| CF — ALS only | 0.572 | 0.764 | 11.41 | 0.855 | 0.3 ms |
| **FULL pipeline** | **0.564** | 0.798 | 12.04 | 0.718 | 15.0 ms |

The popularity baseline reaches **0.5% catalog coverage with a Gini of 0.997** — essentially
the same 30 videos for everyone. Accuracy metrics alone will never tell you that.

### Latency

p50 ≈ 22 ms end-to-end (recall ~3 ms, ranking ~12 ms, policy ~5 ms), p95 < 30 ms, one CPU
core, 6,000 items. A test fails the build if p95 exceeds 250 ms. In production on Vercel,
warm requests measure ~170 ms including network round-trip from the edge.

### The headline caveat: full-catalog NDCG is the wrong metric here

Adding the learned ranker moves the two families of metric in **opposite directions**:

| | full-catalog NDCG@10 | Protocol A top-1 |
|---|---|---|
| Stage 1 recall only | **0.0125** | 0.1840 |
| + learned ranker (shipped) | 0.0107 ↓ | **0.1930** ↑ |

Two metrics disagreeing about the same change means one is wrong *for this purpose*. The
full-catalog metric rewards casting a wide net: a popularity list scores 0.0121, beating
content-based retrieval (0.0090) outright. An **oracle** built from the simulator's own
generative parameters scores 0.0161, so the metric is not pure noise — but the ranking that
wins on it is not the ranking that wins on observed labels, which is why the counterfactual
protocols are the ones quoted.

---

## Test cases

Ten success scenarios and ten failure scenarios, each with inputs, observed behaviour, and
an explanation — [TEST_CASES.md](docs/TEST_CASES.md). Every one is executable:
`python scripts/09_test_cases.py`.

| Success | | Failure | |
|---|---|---|---|
| S1 | Cold start, no history | F1 | Cold **items** — the structural CF hole |
| S2 | Single-interest viewer | F2 | Single-video history is not a taste |
| S3 | Multi-interest viewer | F3 | Contradictory 6-category history |
| S4 | Semantic search | F4 | The TF-IDF template trap |
| S5 | Watch page, "more like this" | F5 | Extreme policy setting, λ = 0.0 |
| S6 | Channel affinity + hard cap | F6 | Popularity beats the hybrid offline |
| S7 | Diversity control is real | F7 | Filter bubble from one interest |
| S8 | Cross-category bridge | F8 | Session-intent blending does not help |
| S9 | Session intent detected | F9 | Multi-objective cannot reduce clickbait |
| S10 | Objective switchable live | F10 | Out-of-vocabulary search retrieves nothing |

## Explainability

Every recommendation carries an explanation covering all three stages. Live in the demo via
the 💡 line on any card:

```
Stage 1  content_history  rank #11  score 0.6324
         cf_als           rank #50  score 0.2365
         channel          rank #20  score 0.9831

Stage 2  objective        P        weight   contribution
         satisfied        33.5%    +1.00    +0.3348
         long_watch       30.3%    +0.15    +0.0455
         liked             4.4%    +0.20    +0.0087
         completion        2.5%    +0.25    +0.0061
         dismissed         0.0%    -0.40    -0.0001
         click            42.6%    +0.00    +0.0000
                                   value score  0.3950

Stage 3  policy actions — no adjustment; ranked purely on relevance
```

The explanations are **mode-aware**: a watch page says *"Because you watched X"*, a cold
start says *"Trending in Gaming"*, and the system never claims a taste profile it does not
have. That last point is enforced by a test, because an earlier version did exactly that.

## YouTube comparison

Full write-up in [COMPARISON.md](docs/COMPARISON.md): what was deliberately reproduced
(two-stage architecture, watch time over clicks, co-visitation for "related videos",
fold-in, freshness, position-bias correction), where this diverges and why, where ReelRank
is arguably *better* (it explains itself, and exposes its policy controls), what it cannot
do at YouTube's scale, and what would come next.

---

## Deployment architecture

**Training and serving have different dependency sets, on purpose.**

| | file | contains | used by |
|---|---|---|---|
| Build / training | `requirements-build.txt` | pandas, scikit-learn, SciPy, PyArrow + the serving set | `scripts/*.py`, tests |
| Serving / inference | `requirements.txt` | NumPy, FastAPI, Uvicorn, Pydantic, PyYAML | the deployed app |

The last build stage ([`scripts/12_export_serving.py`](scripts/12_export_serving.py))
converts the fitted trees and the TF-IDF/SVD encoder into plain arrays, so **the deployed
app imports NumPy and the standard library and nothing else.**

Four reasons, in the order they mattered:

1. **It removes a fragile pickle.** A pickled `HistGradientBoostingClassifier` reaches for a
   Cython class whose `__module__` is the bare string `_loss`, so unpickling needs a
   top-level `import _loss` to resolve. That works locally by accident of import order. In
   production it failed outright with `ModuleNotFoundError: No module named '_loss'`.
2. **Smaller footprint.** Dropping scikit-learn + SciPy + joblib (~200 MB) and pandas
   (~55 MB) took the bundle from 339 MB — over the serverless limit, which is what caused
   the platform to prune files and break the import in the first place — to comfortably under.
3. **More portable inference.** No version-matched unpickling.
4. **Faster cold start.** Less to import.

**The conversion is verified, not assumed** — the exporter refuses to write a bundle that
disagrees with scikit-learn by more than `1e-6`:

| model | max &#124;sklearn − numpy&#124; | checked on |
|---|---|---|
| ranker (101 trees) | 5.551e-17 | 4,000 rows, 5% NaN to exercise missing-value routing |
| 6 objective heads | 4.337e-19 … 1.110e-16 | as above |
| text encoder | 1.192e-07 | all 6,000 catalog documents |
| query vectors | 1.490e-08 | top-3 rankings identical |

Trees agree to machine precision because tree inference is exact arithmetic on the same
thresholds; the encoder's 1.2e-07 is float32 rounding in the SVD components, four orders of
magnitude below the gap between adjacent candidates.

[`tests/test_serving_deps.py`](tests/test_serving_deps.py) enforces this: it blocks pandas,
scikit-learn, SciPy, joblib and PyArrow at the import hook, then runs a real recommendation
and a real search. Full detail, including the Vercel routing fix: [DEPLOY.md](docs/DEPLOY.md).

---

## Engineering decisions

Every decision with its rejected alternative: [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).
Three that came from real failures:

**1 · Leakage through a stacked model** → [`rank/crossfit.py`](src/recsys/rank/crossfit.py)
The first ranker scored **AUC 0.957** and `als_score` carried the entire model. ALS had been
trained on the full log, so item factors encoded the very clicks the ranker was asked to
predict. Enforcing one global temporal cutoff dropped it to 0.578 — *below its own best
single feature*, because training rows were still in-sample for ALS (0.856 in-sample vs
0.584 held out). Fixed with **user-fold cross-fitting**: 0.663, and the model finally beats
every input. *Generalises past this repo:* any stacked model's upstream features must be
out-of-fold.

**2 · The evaluation protocol, not the model** → [`counterfactual.py`](src/recsys/counterfactual.py)
Covered above. The oracle control is the part most tutorials skip.

**3 · A 70× performance bug that was not in my code** → [`recall/cf.py`](src/recsys/recall/cf.py)
ALS took 649 s. Profiling showed a single 96×96 `np.linalg.solve` costing **3.65 ms** —
about 0.16 GFLOPS. OpenBLAS was spawning and synchronising a thread team for a matrix far
too small to amortise it. Pinning to one thread: **51 µs**, and ALS finished in **18 s**.

### Two hypotheses tested rather than assumed

**Session intent** helps exactly the cohort it should (**+7.5%** on focused, off-persona
sessions) and hurts on browsing sessions (**−3.4%**). Net: **−0.1%**. The explanation is the
payoff — the profile is already recency-weighted with a half-life of 8 positions, so
`cos(profile, session vector) = 0.803`. **Recency decay is already a soft session model.**
Remove it and session blending suddenly works (+2.6%). Shipped for *explainability*, with
blending off by default and exposed as a slider so the negative result is demonstrable.

**Multi-objective ranking** buys completion and controllability, but cannot reduce clickbait:

| target | GBDT R² from all item features |
|---|---|
| `latent_quality` | **+0.6355** (visible) |
| `latent_clickbait` | **−0.1122** (invisible) |

**No ranker can optimise an objective absent from its inputs.** That makes the next step a
*feature* problem, not a model problem — which is precisely why YouTube runs user surveys
rather than inferring satisfaction from engagement metadata.

## Limitations

Full list: [LIMITATIONS.md](docs/LIMITATIONS.md). The ones that matter most:

- **Interactions are simulated.** Every metric measures recovery of known latent structure, not real-world performance.
- **Clickbait is invisible to the feature set** (R² = −0.11), so multi-objective ranking cannot reduce it.
- **Cold items** have no collaborative signal — only content and freshness reach them (F1).
- **Lexical search only.** Out-of-vocabulary queries retrieve nothing; the system now says so rather than faking it (F10).
- **Pointwise ranking**, not pairwise/listwise — chosen so the score stays calibrated and blendable; LambdaMART would likely win on NDCG.
- **No online evaluation.** No A/B test, no interleaving, no real users. Offline metrics never settle this.
- **Single-node scale.** Brute-force nearest-neighbour is correct at 6k items and wrong at 10⁶.

## Future improvements

Priority order in [FUTURE_WORK.md](docs/FUTURE_WORK.md): title-vs-content mismatch features
(to make clickbait visible at all), a two-tower retrieval model with ANN, sequential
modelling (GRU4Rec/SASRec), pairwise learning-to-rank, contextual bandits for the
exploration slots, and an online A/B harness.

---

## Project structure

```
config.yaml              every tunable knob; the build is reproducible from this + the seed
Dockerfile               container build; artifacts baked in at image build time
vercel.json  api/        serverless deployment (the live demo)
requirements.txt         SERVING deps (NumPy only)
requirements-build.txt   BUILD deps (adds pandas, scikit-learn, SciPy)
scripts/
  build_all.py           one command, six stages
  01_build_data.py       catalog + simulated watch log
  02_build_features.py   TF-IDF → SVD text vectors, item stats
  03_train_cf.py         co-visitation + implicit ALS (respects the temporal cutoff)
  04_train_ranker.py     learning-to-rank, cross-fitted + the six objective heads
  05_evaluate.py         baselines, ablation, both counterfactual protocols
  12_export_serving.py   convert models to NumPy arrays for serving
  06..11                 MovieLens validation, ablations, diagnostics, test cases,
                         intent evaluation, objective evaluation
src/recsys/
  data/       schema · topics · synthetic · simulator · youtube_api · kaggle_loader
  features/   text.py           TF-IDF+SVD (LSA), optional sentence-transformers
  recall/     content · cf (ALS + co-visitation, from scratch) · heuristic · blend (RRF)
  rank/       features · dataset (causal replay) · crossfit · ranker · multitask
  policy/     rerank.py         MMR, channel cap, freshness, exploration slots
  serving/    trees · text_encoder · export    NumPy-only inference runtime
  catalog_view.py  pandas-free catalog for serving
  intent.py   session intent detection (explanation; blending measured off)
  engine.py   orchestration + explanations
  clock.py · evaluate.py · counterfactual.py · metrics.py · split.py
  api/        FastAPI + the YouTube-style UI (vanilla JS, no build step)
tests/        61 tests; the interesting ones are regressions for real bugs
```

## Local setup

**Building needs `requirements-build.txt`** — plain `requirements.txt` is the serving set and
deliberately excludes pandas and scikit-learn.

```bash
pip install -r requirements-build.txt
```

```bash
python scripts/build_all.py
```

```bash
python -m uvicorn recsys.api.app:app --app-dir src --port 7860
```

Then open <http://localhost:7860>. The full build takes ~7 minutes; `--quick` does a
1-minute smoke build. Or skip all of it:

```bash
docker build -t reelrank . && docker run -p 7860:7860 reelrank
```

Run the tests with `python -m pytest tests/ -q` (needs the build set).

## How to use

| Action | UI | API |
|---|---|---|
| Cold-start feed | open the page | `POST /api/recommend {"history": [], "n": 24}` |
| Personalised feed | click a persona | `POST /api/recommend {"history": ["id1", "id2"]}` |
| Change objective | Recommendation Lab presets | `POST /api/recommend {"objective_weights": {"satisfied": 1.0}}` |
| Explain a recommendation | click the 💡 line | read `items[].explanation_detail` |
| Search | search bar | `GET /api/search?q=brown+butter&n=10` |
| Watch page | click a video | `GET /api/similar/{video_id}` |
| Health | — | `GET /api/health` |
| Model + data metadata | "How it works" | `GET /api/meta` |

## Reproducibility

| Stage | Command | Output |
|---|---|---|
| 1 · Data | `python scripts/01_build_data.py` | catalog + simulated log (`data/processed/`) |
| 2 · Features | `python scripts/02_build_features.py` | text vectors, item stats |
| 3 · CF | `python scripts/03_train_cf.py` | co-visitation index, ALS factors |
| 4 · Training | `python scripts/04_train_ranker.py` | ranker + six objective heads |
| 5 · Evaluation | `python scripts/05_evaluate.py` | `artifacts/evaluation.json` |
| 6 · Export | `python scripts/12_export_serving.py` | `artifacts/serving_models.npz` |
| All | `python scripts/build_all.py` | everything above, in order |

**What is fully reproducible:** everything. There is no private or external data in the
default path — the catalog and the watch log are both generated from `config.yaml` plus
`project.seed`, so a fresh clone plus one command reproduces every number in these docs.

**One caveat, stated precisely:** the catalog dates publish times relative to "now", and the
simulator places sessions in a window ending at "now", so a rebuild on a different day
produces different — statistically equivalent — data. Set `project.reference_date` in
`config.yaml` to pin that instant and the pipeline becomes **byte-identical** across machines
and across days (asserted by `test_build_is_byte_identical_with_a_pinned_reference_date`).
It is left unpinned by default because a live demo should have "trending" mean trending *now*.

**Optional real data:** with a free
[YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started) key, the
catalog comes from the real API instead. The *interactions* remain simulated either way —
that is a property of the world, not of this repo. See [`.env.example`](.env.example).

**Committed artifacts:** the serving bundle (~28 MB) is committed so the deployment needs no
build step and a reviewer can run the app without training anything. The fitting outputs
(`*.joblib`, parquet) are gitignored — they are regenerated by `build_all.py`.

---

## Technical assignment coverage

| Requirement | Implementation | Status |
|---|---|---|
| Working recommendation system | Three-stage engine, 5 recall sources, 6-objective ranker, policy re-rank | ✅ |
| Deployed testing UI, no local setup | <https://reelrank-one.vercel.app> — no login | ✅ |
| Shows inputs | Persona picker, editable watch history, search, live policy sliders | ✅ |
| Shows outputs | YouTube-style feed with rank, score, category, source badges | ✅ |
| Shows **why** | "Why this video?" panel: all three stages + six objectives + raw payload | ✅ |
| Problem statement & use case | [README](#what-this-project-does) · [ARCHITECTURE.md](docs/ARCHITECTURE.md) | ✅ |
| Approach & architecture | [ARCHITECTURE.md](docs/ARCHITECTURE.md) | ✅ |
| Methodology | [METHODOLOGY.md](docs/METHODOLOGY.md) — every algorithm and its rejected alternative | ✅ |
| Dataset | [DATASET.md](docs/DATASET.md) — sources, simulator, parameters | ✅ |
| Technologies used | [Tech stack](#tech-stack) | ✅ |
| Assumptions | [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) — each with what breaks if wrong | ✅ |
| Design decisions | [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | ✅ |
| Evaluation methodology | [EVALUATION.md](docs/EVALUATION.md) — classification *and* ranking metrics, 2 counterfactual protocols, oracle control | ✅ |
| Successful test cases | S1–S10, [TEST_CASES.md](docs/TEST_CASES.md), executable | ✅ |
| **Failure** test cases | F1–F10, [TEST_CASES.md](docs/TEST_CASES.md), with cause and fix path | ✅ |
| Limitations | [LIMITATIONS.md](docs/LIMITATIONS.md) | ✅ |
| Future improvements | [FUTURE_WORK.md](docs/FUTURE_WORK.md) | ✅ |
| GitHub repository | <https://github.com/AryanChougule/youtube_video_recommender_system> | ✅ |
| Deployment link | <https://reelrank-one.vercel.app> | ✅ |
| **Bonus** — platform comparison | [COMPARISON.md](docs/COMPARISON.md) | ✅ |
| **Bonus** — YouTube-inspired UI | Dark theme, thumbnail grid, watch page, search, category chips | ✅ |

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
| [DEPLOY.md](docs/DEPLOY.md) | Vercel, Docker, Hugging Face Spaces, and the serving architecture |

## Tech stack

**Serving:** Python 3.10+ · NumPy · FastAPI · Uvicorn · vanilla JS (no build step)
**Build/training:** the above plus pandas · scikit-learn · SciPy · PyArrow
**Deployment:** Vercel serverless · Docker · Hugging Face Spaces

**No `implicit`, no `faiss`, no `lightgbm`, no PyTorch.** ALS, the co-visitation index, the
nearest-neighbour search, RRF and MMR are all hand-written — which is the point
pedagogically, and also keeps the deployed function small enough to be serverless.

## License

MIT
