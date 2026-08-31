# Evaluation

## Metric coverage at a glance

Every metric measured, where it is reported, and what it is computed by. Everything in the
first group is written to `artifacts/evaluation.json` at k = 5, 10 and 20 by
[`src/recsys/metrics.py`](../src/recsys/metrics.py).

| Metric | Status | Where reported |
|---|---|---|
| **Precision@k** | measured | [Accuracy in full](#accuracy-metrics-in-full--precision-recall-map-ndcg-hit-rate-serendipity) |
| **Recall@k** | measured | [Accuracy in full](#accuracy-metrics-in-full--precision-recall-map-ndcg-hit-rate-serendipity) |
| **NDCG@k** | measured | [Protocol A](#protocol-a--re-ranking-logged-impressions), [Protocol B](#protocol-b--1-positive-vs-100-sampled-negatives), [Accuracy in full](#accuracy-metrics-in-full--precision-recall-map-ndcg-hit-rate-serendipity) |
| **MAP@k** | measured | [Accuracy in full](#accuracy-metrics-in-full--precision-recall-map-ndcg-hit-rate-serendipity) |
| **Diversity** (intra-list) | measured | [Beyond-accuracy](#beyond-accuracy-metrics) |
| **Coverage** | measured | [Beyond-accuracy](#beyond-accuracy-metrics) |
| **Novelty** (self-information) | measured | [Beyond-accuracy](#beyond-accuracy-metrics) |
| **Latency** (p50/p95/mean) | measured | [Latency](#latency) |
| **User satisfaction** | measured, via a simulated survey label | [User satisfaction](#user-satisfaction-metrics) |
| **Business metrics** | *proxies only* — real ones need real users | [Business metrics](#business-metrics) |
| Hit rate@k, MRR | measured | Protocols A and B |
| Serendipity@k | measured | [Accuracy in full](#accuracy-metrics-in-full--precision-recall-map-ndcg-hit-rate-serendipity) |
| Gini (exposure fairness) | measured | [Beyond-accuracy](#beyond-accuracy-metrics), [Business metrics](#business-metrics) |
| AUC, watch-time-weighted AUC, log loss | measured | [Per-objective](#per-objective-ranking-metrics) |
| Top-1 within feed | measured | [Protocol A](#protocol-a--re-ranking-logged-impressions) |

Two of these need justification rather than a number, and both are given one in their own
section: **business metrics** cannot be measured offline without inventing them, and
**user satisfaction** is measured against a *simulated* survey signal, not real people.

The single most important thing on this page is not any individual metric. It is
[Finding 2](#finding-2--the-standard-offline-protocol-does-not-measure-the-recommender):
the standard full-catalog protocol, on which precision, recall and MAP are usually quoted,
**does not measure recommendation quality on logged data** — and this project demonstrates
that with an oracle control rather than asserting it.

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
| + mixed negative sampling + relative user features | **0.6627** | **25.8%** ¹ |

¹ *Only the final row corresponds to the artifacts shipped here — it is read directly from
[`artifacts/ranker_report.json`](../artifacts/ranker_report.json), which is committed, so it
can be checked without retraining: `auc 0.6627`, `weighted_auc 0.7553`,
`within_feed_top1 0.2575` against a random baseline of `0.1429`. The first three rows were
measured during development, before the six objective heads were added and the ranker
retrained; they are the debugging record and are not reproducible from the current
artifacts. Note the random baseline itself changed from 20% to 14.3% when random negatives
were added, so 25.8% is a 1.80× lift where 32.7% had been 1.64×.*

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
| **ORACLE** (true generative parameters) ‡ | **0.0165** | — |
| hybrid recall only (Stage 1) | 0.0125 | 0.0222 |
| popularity | 0.0121 | 0.0267 |
| CF — ALS only | 0.0119 | 0.0198 |
| FULL pipeline (Stage 1+2+3) | 0.0106 | 0.0187 |
| content only | 0.0090 | 0.0167 |
| random | 0.0010 | 0.0022 |

‡ *From `python scripts/13_oracle_control.py`, which needs the simulator's hidden
generative variables and so runs outside stage 5. Its baselines come from
`counterfactual.make_scorers`, which builds the user profile slightly differently
from the stage-5 baselines — there, popularity scores 0.0121 and content-only 0.0095. The oracle's margin
over both is the point, and it holds either way.*

Two things are wrong here, and they are different problems.

**First, the metric is popularity-dominated.** A trivial popularity list (0.0121) beats
content-based retrieval (0.0090) and matches ALS. Users can only click what the logging policy
showed them, and that policy was popularity-heavy, so reproducing it scores well.

**Second — and this is the decisive evidence — adding the learned ranker moves the two families
of metric in OPPOSITE directions:**

| | full-catalog NDCG@10 | Protocol A top-1 |
|---|---|---|
| Stage 1 recall only | **0.0125** | 0.1840 † |
| + learned ranker | 0.0107 ↓ | **0.1933** ↑ |

† *content-only, the strongest single Stage-1 signal.*

Two metrics disagreeing about the same change means one of them is wrong **for this purpose**.
Full-catalog NDCG rewards casting a wide net over the catalog; Protocol A measures whether the
right video went on top of a page the user actually saw. Only the second is the product question.

The **oracle** scores 0.0165 — above everything else — so the metric is not pure
noise, and there is genuine headroom our models do not capture. But it is not measuring what we
are trying to improve. Reproduce it with `python scripts/13_oracle_control.py`.

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

| scorer | top-1 | NDCG | MRR |
|---|---|---|---|
| `[ref] shown position` | 0.7713 | 0.9086 | 0.8768 |
| **LEARNED RANKER** | **0.1933** | **0.5565** | **0.4187** |
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

| scorer | HR@10 | NDCG@10 | mean rank |
|---|---|---|---|
| content only | **0.3703** | **0.1821** | 30.91 |
| **LEARNED RANKER** | 0.3586 | 0.1780 | **28.33** |
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

## Accuracy metrics in full — precision, recall, MAP, NDCG, hit rate, serendipity

Every accuracy metric is computed at k = 5, 10 and 20 by
[`src/recsys/metrics.py`](../src/recsys/metrics.py) and persisted to
`artifacts/evaluation.json`. At k = 10, on the full-catalog protocol:

| Strategy | P@10 | R@10 | MAP@10 | NDCG@10 | HR@10 | Serendipity@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0151 | 0.0164 | 0.0078 | 0.0121 | 0.1367 | 0.0007 |
| CF — ALS only | 0.0107 | 0.0104 | 0.0042 | 0.0119 | 0.1013 | 0.0100 |
| hybrid recall only | 0.0123 | 0.0111 | 0.0044 | 0.0125 | 0.1100 | 0.0115 |
| **FULL pipeline** | 0.0104 | 0.0110 | 0.0046 | 0.0106 | 0.0947 | 0.0100 |
| content only | 0.0083 | 0.0080 | 0.0035 | 0.0090 | 0.0767 | 0.0083 |
| CF — co-visitation only | 0.0066 | 0.0065 | 0.0022 | 0.0065 | 0.0627 | 0.0066 |
| random | 0.0013 | 0.0011 | 0.0005 | 0.0010 | 0.0133 | 0.0013 |

**Read this table together with Finding 2, not on its own.** Popularity tops precision,
recall and MAP here — and it is the worst product in the table, at 0.5% catalog coverage
with a Gini of 0.997. That is not a paradox; it is the point. Full-catalog retrieval
metrics reward matching the logging policy, and the logging policy was popularity-biased.
The counterfactual protocols above are the ones that answer the product question.

The one column where the ranking matches intuition is **serendipity** — relevant *and*
non-obvious. Popularity scores 0.0007, essentially zero by construction, because nothing
it recommends is a surprise. The full pipeline scores 0.0100, roughly 14x higher.

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

## User-satisfaction metrics

The brief lists user satisfaction as a metric to consider. It is normally the hardest one
to get, because satisfaction is not an engagement signal — it is the thing engagement is a
*proxy* for, and the two come apart exactly where it matters. YouTube solves this by asking
people directly, in surveys.

This project cannot ask anyone, so the simulator emits a survey-like `satisfied` label from
a separate generative path (affinity, quality, watch fit, and a strong negative clickbait
term), and it is measured as a first-class objective rather than inferred from watch time:

| configuration | engagement@1 | **satisfaction@1** | NDCG(satisfaction) | completion@1 | clickbait@1 |
|---|---|---|---|---|---|
| A. CTR-optimised | 0.1983 | 0.1473 | 0.3857 | 0.0055 | 0.2057 |
| C. Satisfaction-only | 0.1973 | **0.1462** | **0.3897** | 0.0095 | **0.2026** |
| D. Multi-objective (shipped) | 0.1963 | 0.1453 | 0.3893 | **0.0100** | 0.2044 |

`NDCG(satisfaction)` re-grades the identical ranking by whether the click was *satisfying*
rather than by watch fraction — the same ordering, a different question asked of it.

**The honest reading:** satisfaction@1 barely moves between objectives (0.1473 to
0.1453). Completion@1 nearly doubles. Clickbait exposure does not move at all. A
satisfaction *label* is not enough on its own — the ranker also needs features that
correlate with it, and this feature set does not have them ([F9](TEST_CASES.md)).

---

## Business metrics

Real business metrics — revenue, retention, DAU, session length, subscriber growth —
**require real users**, and every one of them is a longitudinal measurement that no offline
protocol can produce. Claiming them here would be fabrication, so this section states what
*is* measurable and what would have to be instrumented.

**Measurable now, as leading indicators:**

| Business concern | Proxy measured here | Value | Why it is the right proxy |
|---|---|---|---|
| Inventory utilisation | catalog coverage | 0.562 vs popularity's 0.004 | a catalog nobody sees earns nothing; 0.4% coverage strands 99.6% of the library |
| Creator-ecosystem health | Gini of exposure | 0.799 vs popularity's 0.997 | winner-take-all exposure drives mid-tail creators off a platform |
| Brand / trust risk | clickbait@1 | 0.2044 | bait converts short-term clicks into long-term distrust |
| Content depth | completion@1 | 0.0100 (+82% vs CTR-optimised) | finished videos, not opened ones, indicate delivered value |
| Serving cost | p50 latency | ~17 ms, one CPU core | cost per request is a real constraint at scale |

**Not measurable without an online system,** and stated as such rather than approximated:
click-through lift, watch-time-per-session, retention/churn, revenue per user, subscriber
conversion. These need an A/B or interleaving harness against live traffic — the highest
priority item in [FUTURE_WORK.md](FUTURE_WORK.md), and the reason
[Finding 2](#finding-2--the-standard-offline-protocol-does-not-measure-the-recommender)
matters: offline metrics disagree with each other here, so only an online experiment
settles which change is actually better.

**Gini deserves a caveat.** The full pipeline's 0.799 is
better than popularity's 0.997, and still not defensible for a real
creator ecosystem. It is reported because hiding it would be the dishonest choice, and it is
on the roadmap as an explicit exposure-fairness constraint rather than a hoped-for side
effect of diversity re-ranking.

---

## Per-objective ranking metrics

AUC alone was not enough — it measures separation on a fixed label, not whether
the right video reached the top of a real page. Both are now reported
(`python scripts/11_evaluate_objectives.py`):

| objective | top-1 | NDCG | MRR | NDCG(satisfaction) | completion@1 | clickbait@1 |
|---|---|---|---|---|---|---|
| A. CTR-optimised | 0.1983 | 0.5618 | 0.4250 | 0.3857 | 0.0055 | 0.2057 |
| B. Watch-time optimised | 0.1933 | 0.5565 | 0.4187 | 0.3801 | 0.0035 | 0.2086 |
| C. Satisfaction-only | 0.1973 | **0.5643** | **0.4280** | **0.3897** | 0.0095 | **0.2026** |
| D. Multi-objective (shipped) | 0.1963 | 0.5638 | 0.4273 | 0.3893 | **0.0100** | 0.2044 |

*NDCG(satisfaction) re-grades the identical ranking by whether the click was
satisfying rather than by watch fraction.*

And each head used **alone** as a ranker, which is where AUC and ranking
quality come apart most sharply:

| head | top-1 | NDCG | MRR | AUC (fit) |
|---|---|---|---|---|
| satisfied | **0.1982** | **0.5636** | **0.4272** | 0.7376 |
| click | 0.1933 | 0.5565 | 0.4187 | 0.6627 |
| liked | 0.1895 | 0.5559 | 0.4176 | 0.7436 |
| long_watch | 0.1858 | 0.5524 | 0.4134 | 0.8124 |
| completion | 0.1768 | 0.5389 | 0.3969 | 0.9154 |
| dismissed | 0.1005 | 0.4667 | 0.3061 | **0.9154** |

**`dismissed` has the highest AUC and the worst ranking.** It separates its rare
label (0.44% positive) almost perfectly while being nearly useless for ordering a
page: `AUC 0.9154` against `top-1 0.1005`, which is *below random* (0.1415) on
Protocol A. Reporting AUC alone would have made it look like the best head in
the system.

The reason is that the two metrics ask different questions. AUC asks "across all
pairs, does a positive outrank a negative?" — and when only 0.44% of rows are
positive, a head that confidently pushes almost everything to "not dismissed"
scores superbly. Top-1 asks "on this specific page of ~8 candidates, is the item
you put first the one they engaged with?" A page is not a random pair, and a
recommender only ever gets graded on pages.

Both columns come from `artifacts/objective_evaluation.json` (`_per_head` and
`_per_head_auc`), regenerated by `python scripts/11_evaluate_objectives.py`, so
every number in this table is checkable without rerunning training.

## Latency

p50 ≈ 17 ms end-to-end (recall ~2 ms, ranking ~7 ms, policy ~4 ms), p95 < 20 ms, on one CPU
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
python -m pytest tests/ -q                    # 61 tests
```
