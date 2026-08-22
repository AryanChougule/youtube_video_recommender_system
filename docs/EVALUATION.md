# Evaluation

## Protocol

**Global temporal split.** One wall-clock cutoff at the 80th percentile of impression
timestamps. Train on the past, test on the future.

Not a random split (which lets a user's Friday click train the model scored on their Monday
click), and not a per-user leave-one-out (which still lets *other* users' futures inform the
model). Only a global temporal split matches how the system actually runs.

**Every model respects the same cutoff.** Co-visitation, ALS, and the ranker are all fitted on
pre-cutoff data only, enforced in [`src/recsys/split.py`](../src/recsys/split.py) and asserted
at load time. This matters more than it sounds — see *Finding 1*.

```
2026-05-23 ─────────────────── 2026-08-03 12:21 ─────────────── 2026-08-21
│              TRAIN (879,814 impressions)     │  TEST (219,954)         │
│  co-visitation · ALS · ranker fitted here    │  metrics measured here  │
```

**Graded relevance = watch fraction**, not a binary click. A fully-watched video counts more
than an abandoned one, so the evaluation stays aligned with the training objective instead of
quietly reverting to clicks.

**Evaluation users:** 1,500, each with ≥3 pre-cutoff clicks and ≥1 held-out click. Held-out items
the user had already watched before the cutoff are excluded — that is a rewatch, not a
prediction task, and the engine deliberately filters history.

---

## Finding 1 — the leakage that made the first model look excellent

The first ranker scored **AUC 0.957**, and permutation importance said `als_score` accounted
for essentially all of it.

That was not skill. ALS had been trained on the *full* interaction log, so its item factors
were fitted using the very clicks the ranker was being asked to predict. The user side was
causal (we fold in from history); the **item** side had seen the future.

| Fix applied | AUC | within-feed top-1 |
|---|---|---|
| (none — ALS on full log) | 0.957 ❌ | — |
| Global temporal cutoff for all models | 0.578 | 25.9% |
| + user-fold cross-fitting of CF features | 0.643 | 32.7% |
| + mixed negative sampling + relative user features | **0.697** | **28.9%** ¹ |

¹ *Random baseline changed from 20% to 14.3% when random negatives were added, so 28.9% is a
larger lift (2.02× random) than 32.7% was (1.64×). The first three rows were measured during
development and are reported as the debugging record; only the final row corresponds to the
artifacts shipped here.*

The intermediate 0.578 is the instructive one: **the model was worse than its own best single
feature** (content similarity alone scored 0.641). That is the diagnostic signature of a leaking
upstream feature — `als_score` measured AUC 0.856 on training rows but 0.584 held out, so the
ranker learned to trust it far more than it deserved.

**The general rule:** in a multi-stage system, any upstream model's output used as a downstream
feature must be **out-of-fold**, exactly as in stacking ensembles. A temporal cutoff alone is
not sufficient, because training rows are still in-sample for the upstream model.

`scripts/04_train_ranker.py` now prints a warning if the trained model fails to beat its best
single input — the check that would have caught this immediately.

---

## Finding 2 — the standard offline protocol does not measure the recommender

The usual recipe: hide each user's future clicks, ask the model to retrieve them from the full
catalog, report NDCG. Run it and you get this:

| Full-catalog retrieval | NDCG@10 | recall@20 |
|---|---|---|
| **popularity** | **0.0178** | 0.0342 |
| CF — ALS only | 0.0176 | 0.0256 |
| hybrid recall only | 0.0167 | 0.0277 |
| FULL pipeline | 0.0120 | 0.0250 |
| content only | 0.0108 | 0.0240 |
| random | 0.0015 | 0.0036 |

A trivial popularity list beats everything. The tempting conclusion is that the recommender is
broken. Before accepting it, we built a control that only a simulator makes possible:

> **Score an ORACLE** — the exact generative model that produced the data, using the true user
> personas, the true hidden video quality, and the true duration preferences.

| | NDCG@10 |
|---|---|
| popularity | 0.0178 |
| **ORACLE (true generative parameters)** | **0.0169** |

**The data-generating process itself loses to a popularity list.** If the true model cannot win,
the metric is not measuring model quality.

### Why

A user can only click what they were shown. The logging policy is popularity-heavy, so held-out
clicks concentrate on popular videos — 26% of them are in the catalog's top 10% by views. A
popularity ranker scores well by **reproducing the old policy**, while a genuinely better
recommender is *penalised* for surfacing something the user never had the chance to click.

This is not an artifact of our simulator. It is the central pathology of offline recommender
evaluation on logged data, and it is why the field increasingly reports counterfactual
estimators. Most tutorials report the biased number without knowing it.

We checked the obvious alternative explanation — that popularity simply wins for mainstream
users — and it does not hold: popularity beats personalisation even in the **niche** cohort
(mainstream-ness < 0.35): 0.0168 vs 0.0104 when that split was measured.

---

## The two protocols we actually trust

### Protocol A — re-ranking logged impressions

Re-order only the videos users were **actually shown**. Every label is observed; no
counterfactual guessing. This answers the real product question — *given the same page, would we
have put the right video on top?* — and is the offline protocol that best predicts online lift.

6,000 test-period feeds of 8 items each, one click per feed:

| Scorer | top-1 | NDCG | MRR |
|---|---|---|---|
| `[ref]` shown position † | 0.7365 | 0.8920 | 0.8549 |
| **LEARNED RANKER (19 features)** | **0.2090** | **0.5707** | **0.4366** |
| content only | 0.2025 | 0.5671 | 0.4320 |
| CF — ALS only | 0.1755 | 0.5400 | 0.3982 |
| CF — co-visitation only | 0.1375 | 0.4943 | 0.3419 |
| popularity | 0.1338 | 0.4963 | 0.3438 |
| random | 0.1305 | 0.4933 | 0.3400 |

† **Not a competing model.** Under a cascade click model, position *causes* the click — a scorer
that knows the slot has access to the label's mechanism. This row measures the size of position
bias (the ceiling on what re-ranking could ever be worth), not video quality. Comparing a
content model against it would be a category error.

The hybrid ranker leads every genuine model, and is 1.60× random.

### Protocol B — 1 positive vs 100 sampled negatives

Isolates pure ranking ability from the "find the needle the old policy hid" retrieval problem.

| Scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.4753** | 0.2389 | 23.85 |
| **LEARNED RANKER** | 0.4679 | **0.2410** | **21.34** |
| CF — ALS only | 0.3384 | 0.1748 | 33.05 |
| popularity | 0.2608 | 0.1429 | 36.96 |
| CF — co-visitation only | 0.1455 | 0.0798 | 48.21 |
| random | 0.0926 | 0.0433 | 51.29 |

Sampled metrics have their own documented bias — they systematically flatter weaker models
([Krichene & Rendle, KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403226)) — which is
visible here: content-only edges the hybrid on raw HR@10 while losing on both NDCG@10 and
mean rank. That is why both
protocols are reported and neither is quoted alone.

---

## Beyond-accuracy metrics

Accuracy alone will happily reward a system that shows everyone the same 20 viral videos.

| Strategy | coverage | Gini | novelty (bits) | intra-list diversity |
|---|---|---|---|---|
| random | 0.994 | 0.249 | 12.86 | 0.946 |
| popularity | **0.005** | **0.997** | 9.17 | 0.938 |
| trending | 0.003 | 0.997 | 16.60 | 0.948 |
| content only | 0.878 | 0.526 | 12.77 | **0.533** |
| CF — ALS only | 0.563 | 0.767 | 11.30 | 0.847 |
| hybrid recall only | 0.735 | 0.665 | 11.74 | 0.711 |
| **FULL pipeline** | 0.571 | 0.799 | 11.97 | 0.717 |

Two things this table shows that NDCG cannot:

- **The popularity baseline is a terrible product.** 0.5% catalog coverage, Gini 0.997 — it
  shows essentially the same 30 videos to every user forever. It "wins" on the biased accuracy
  metric while being useless.
- **Content-only has the worst intra-list diversity (0.533)** — the filter-bubble signature. It
  retrieves accurately and repetitively.

### Definitions

| Metric | Formula | Reads as |
|---|---|---|
| Coverage | \|∪ recommended\| / \|catalog\| | how much of the corpus is reachable |
| Gini | inequality of exposure counts | is exposure concentrated? |
| Novelty | mean −log₂ p(i) | how obscure the picks are |
| Intra-list diversity | mean pairwise (1 − cos) | how varied one page is |
| Serendipity | relevant AND not in the popularity top-K | value beyond the obvious |

## Latency

p50 ≈ 22 ms end-to-end (recall ~3 ms, ranking ~12 ms, policy ~5 ms), p95 < 30 ms, on one CPU
core with 6,000 items. A test asserts p95 < 250 ms so a regression fails CI rather than
production.

## Policy sensitivity

| MMR λ | NDCG@10 | coverage | intra-list diversity |
|---|---|---|---|
| 0.50 | 0.0122 | 0.436 | 0.749 |
| **0.72** | 0.0120 | 0.401 | 0.717 |
| 0.90 | 0.0126 | 0.380 | 0.695 |
| 1.00 | **0.0135** | 0.377 | 0.681 |

Accuracy varies by ~11% across the whole λ range while coverage moves by 16% and diversity by
10%. λ = 1.0 does edge the others on NDCG here — but on the **biased** full-catalog metric,
which Finding 2 shows is not a safe basis for a decision. Given that, shipping λ = 0.72 buys a
visibly less repetitive page for an accuracy difference that this protocol cannot reliably
measure. It is a judgement call, and it is exposed as a live slider in the UI for exactly that
reason.

---

## What these numbers do and do not mean

**They do** show that the algorithms recover the latent preference structure that generated the
data, that each pipeline stage earns its place, and that the implementations are correct
(cross-checked on real MovieLens ratings — see [DATASET.md](DATASET.md)).

**They do not** predict real-world YouTube performance. The interactions are simulated; real
viewers have moods, social context, multi-device sessions, and taste that shifts for reasons no
simulator encodes.

**The only way to actually know is an online A/B test**, measuring long-run satisfaction rather
than session watch time. Offline metrics are a filter for obviously-bad ideas, not a substitute
for that. Anyone claiming otherwise has not looked hard enough at their own protocol — as
Finding 2 demonstrates.

## Reproducing

```bash
python scripts/05_evaluate.py --users 1500    # full evaluation → artifacts/evaluation.json
python scripts/06_validate_on_movielens.py    # real-data validation of the algorithms
python scripts/08_diagnose_ranker.py          # oracle ceiling analysis
python scripts/07_ablate_text_fields.py       # text-field ablation
python -m pytest tests/ -q                    # 40 tests
```
