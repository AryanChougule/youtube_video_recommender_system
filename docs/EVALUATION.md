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
| **ORACLE** (true generative parameters) | **0.0161** | — |
| hybrid recall only (Stage 1) | 0.0125 | 0.0222 |
| popularity | 0.0121 | 0.0267 |
| CF — ALS only | 0.0119 | 0.0198 |
| FULL pipeline (Stage 1+2+3) | 0.0107 | 0.0191 |
| content only | 0.0090 | 0.0167 |
| random | 0.0010 | 0.0022 |

Two things are wrong here, and they are different problems.

**First, the metric is popularity-dominated.** A trivial popularity list (0.0121) beats
content-based retrieval (0.0090) and matches ALS. Users can only click what the logging policy
showed them, and that policy was popularity-heavy, so reproducing it scores well.

**Second — and this is the decisive evidence — adding the learned ranker moves the two families
of metric in OPPOSITE directions:**

| | full-catalog NDCG@10 | Protocol A top-1 |
|---|---|---|
| Stage 1 recall only | **0.0125** | 0.1840 † |
| + learned ranker | 0.0107 ↓ | **0.1930** ↑ |

† *content-only, the strongest single Stage-1 signal.*

Two metrics disagreeing about the same change means one of them is wrong **for this purpose**.
Full-catalog NDCG rewards casting a wide net over the catalog; Protocol A measures whether the
right video went on top of a page the user actually saw. Only the second is the product question.

The **oracle** scores 0.0161 — above everything else — so the metric is not pure noise, and
there is genuine headroom our models do not capture. But it is not measuring what we are trying
to improve.

> On an earlier build (before latent clickbait entered the generator) the oracle scored *below*
> the popularity baseline outright — an even starker version of the same point. Both builds
> agree on the conclusion; these are the numbers that reproduce today.

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
| `[ref]` shown position † | 0.7713 | 0.9086 | 0.8768 |
| **LEARNED RANKER (19 features)** | **0.1930** | **0.5562** | **0.4183** |
| content only | 0.1840 | 0.5514 | 0.4119 |
| CF — ALS only | 0.1622 | 0.5259 | 0.3804 |
| random | 0.1415 | 0.5018 | 0.3507 |
| popularity | 0.1323 | 0.4949 | 0.3417 |
| CF — co-visitation only | 0.1280 | 0.4928 | 0.3392 |

Note that **popularity and co-visitation now score below random**. Per page, "show the globally
popular thing" is actively worse than chance — the exact opposite of what the full-catalog
metric concluded about the same scorer.

† **Not a competing model.** Under a cascade click model, position *causes* the click — a scorer
that knows the slot has access to the label's mechanism. This row measures the size of position
bias (the ceiling on what re-ranking could ever be worth), not video quality. Comparing a
content model against it would be a category error.

The hybrid ranker leads every genuine model, and is 1.36× random.

### Protocol B — 1 positive vs 100 sampled negatives

Isolates pure ranking ability from the "find the needle the old policy hid" retrieval problem.

| Scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.3703** | **0.1821** | 30.91 |
| **LEARNED RANKER** | 0.3567 | 0.1776 | **28.49** |
| CF — ALS only | 0.2792 | 0.1484 | 37.24 |
| popularity | 0.2665 | 0.1484 | 36.29 |
| CF — co-visitation only | 0.1428 | 0.0758 | 48.75 |
| random | 0.0999 | 0.0450 | 51.27 |

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
| random | 0.994 | 0.250 | 12.84 | 0.945 |
| popularity | **0.004** | **0.997** | 9.69 | 0.863 |
| trending | 0.003 | 0.997 | 16.68 | 0.953 |
| content only | 0.890 | 0.509 | 12.78 | **0.536** |
| CF — ALS only | 0.572 | 0.764 | 11.41 | 0.855 |
| hybrid recall only | 0.756 | 0.645 | 11.84 | 0.721 |
| **FULL pipeline** | 0.564 | 0.798 | 12.04 | 0.718 |

Two things this table shows that NDCG cannot:

- **The popularity baseline is a terrible product.** 0.5% catalog coverage, Gini 0.997 — it
  shows essentially the same 30 videos to every user forever. It "wins" on the biased accuracy
  metric while being useless.
- **Content-only has the worst intra-list diversity (0.536)** — the filter-bubble signature. It
  retrieves accurately and repetitively.

### Definitions

| Metric | Formula | Reads as |
|---|---|---|
| Coverage | \|∪ recommended\| / \|catalog\| | how much of the corpus is reachable |
| Gini | inequality of exposure counts | is exposure concentrated? |
| Novelty | mean −log₂ p(i) | how obscure the picks are |
| Intra-list diversity | mean pairwise (1 − cos) | how varied one page is |
| Serendipity | relevant AND not in the popularity top-K | value beyond the obvious |

## Per-objective ranking metrics

AUC alone was not enough — it measures separation on a fixed label, not whether
the right video reached the top of a real page. Both are now reported
(`python scripts/11_evaluate_objectives.py`):

| objective | top-1 | NDCG | MRR | NDCG(satisfaction) |
|---|---|---|---|---|
| A. CTR-optimised | 0.1972 | 0.5613 | 0.4243 | 0.3853 |
| B. Watch-time optimised | 0.1930 | 0.5562 | 0.4183 | 0.3800 |
| C. Satisfaction-only | 0.1972 | **0.5641** | **0.4278** | **0.3893** |
| D. Multi-objective (shipped) | 0.1967 | 0.5637 | 0.4272 | 0.3890 |

*NDCG(satisfaction) re-grades the identical ranking by whether the click was
satisfying rather than by watch fraction.*

And each head used **alone** as a ranker, which is where AUC and ranking
quality come apart most sharply:

| head | top-1 | NDCG | MRR | AUC (fit) |
|---|---|---|---|---|
| satisfied | **0.1973** | **0.5633** | **0.4268** | 0.7376 |
| click | 0.1930 | 0.5562 | 0.4183 | 0.6627 |
| liked | 0.1898 | 0.5560 | 0.4177 | 0.7436 |
| long_watch | 0.1852 | 0.5519 | 0.4128 | 0.8124 |
| completion | 0.1773 | 0.5390 | 0.3970 | 0.9154 |
| dismissed | 0.1010 | 0.4669 | 0.3064 | **0.9154** |

**`dismissed` has the highest AUC and the worst ranking.** It separates its rare
label (0.44% positive) almost perfectly while being nearly useless for ordering a
page. Reporting AUC alone would have made it look like the best head in the
system.

## Latency

p50 ≈ 22 ms end-to-end (recall ~3 ms, ranking ~12 ms, policy ~5 ms), p95 < 30 ms, on one CPU
core with 6,000 items. A test asserts p95 < 250 ms so a regression fails CI rather than
production.

## Policy sensitivity

| MMR λ | NDCG@10 | coverage | intra-list diversity |
|---|---|---|---|
| 0.50 | 0.0102 | 0.438 | 0.749 |
| **0.72** | **0.0107** | 0.398 | 0.718 |
| 0.90 | 0.0103 | 0.368 | 0.700 |
| 1.00 | 0.0099 | 0.365 | 0.688 |

The shipped λ = 0.72 happens to top this sweep, but the spread (0.0099–0.0107) is small and the
metric is the biased one, so this is not the reason to ship it. The reason is that coverage moves
20% and intra-list diversity 9% across the range: **diversity here is close to free**. It remains
a judgement call, which is why it is a live slider in the UI.

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
python -m pytest tests/ -q                    # 45 tests
```
