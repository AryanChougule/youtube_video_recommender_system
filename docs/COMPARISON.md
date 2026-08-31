# ReelRank vs real YouTube

The bonus challenge: benchmark this system against the product that inspired it.

Sources for YouTube's side: Davidson et al., *The YouTube Video Recommendation System*
(RecSys 2010); Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations*
(RecSys 2016); Zhao et al., *Recommending What Video to Watch Next: A Multitask Ranking System*
(RecSys 2019); Beutel et al., *Fairness in Recommendation Ranking through Pairwise Regularization*
(KDD 2019); and YouTube's public engineering/policy blog posts.

---

## Similarities — what I deliberately reproduced

### 1 · The two-stage architecture

Both split retrieval from ranking, for the same reason: you cannot afford a rich model over the
whole corpus.

| | YouTube (2016 paper) | ReelRank |
|---|---|---|
| Corpus | ~10⁹ videos | 6 × 10³ |
| Candidate generation | deep net → ANN over learned embeddings | 5 retrievers fused with RRF |
| Candidates passed on | ~hundreds | ~400 |
| Ranking | deep net, hundreds of features | GBDT, 19 features |
| Final page | ~dozens | 24 |

### 2 · Watch time as the objective, not clicks

This is the most important thing I copied. YouTube switched from CTR to watch time in 2012 after
click optimisation produced clickbait. I implement their exact weighted-logistic trick — weight
positives by watch seconds so the learned **odds** approximate `E[watch time]`
([METHODOLOGY.md](METHODOLOGY.md#21-the-objective-watch-time-not-clicks)).

### 3 · Co-visitation as the workhorse for "related videos"

The 2010 paper describes normalised co-visitation counts as YouTube's related-videos engine.
Mine is the same idea with a tunable popularity damping term.

### 4 · Fold-in for users the model has never seen

YouTube's candidate generator embeds a user from their recent history at serve time. My ALS
fold-in does the same thing in closed form — one 96×96 solve, ~51 µs, no retraining.

### 5 · Explicit freshness handling

The 2016 paper feeds "age of video" as a feature and sets it to zero at serving so the model does
not inherit the training window's recency bias. I handle it as an explicit policy multiplier
instead — same motivation, simpler mechanism.

### 6 · Position bias treated as a first-class problem

YouTube feeds position as a training feature and zeroes it at inference. I use inverse
propensity scoring instead. Both accept the same premise: **naive training on logged clicks
learns "things at the top get clicked."**

### 7 · The UI

Dark theme, 12px card radius, thumbnail-first grid, duration badge bottom-right, channel avatar,
"views · age" metadata line, chip row for categories, search in the header. Close enough that an
evaluator can judge the *recommendations* rather than decode a new layout.

---

## Differences — where I diverged, and why

| Dimension | YouTube | ReelRank | Why |
|---|---|---|---|
| **Models** | Deep neural nets end-to-end | Matrix factorisation + GBDT | 3,831 users cannot train a deep net. MF is the right capacity for this data size, and the maths is inspectable. |
| **Objectives** | Multi-task (Zhao 2019): watch time, likes, shares, dismissals, survey responses, with a Mixture-of-Experts head and a shallow tower for position bias | Single objective (watch time) | Multi-task needs multiple label streams. I have one. |
| **Signals** | Audio/video content, thumbnails, ASR transcripts, watch-time curves, subscriptions, survey responses, demographics | Text metadata + simulated watch logs | Availability. |
| **Explanations** | Minimal ("Recommended for you") | Every item names its recall source, rank, ranker score and policy actions | An evaluation harness has the opposite requirement to a consumer product: it must be legible. |
| **Diversity** | Learned, and partly a policy/regulatory matter | Explicit MMR with a live λ slider | Transparency over sophistication. |
| **Exploration** | Large-scale bandits with real traffic | 2 reserved slots, ε-greedy | No traffic to learn from. |
| **Safety** | Borderline-content classifiers, authoritative sources for news/health, age gating | **None** | Out of scope, and a genuine gap ([L15](LIMITATIONS.md)). |
| **Freshness** | Minutes | Rebuild cycle | Static artifacts. |
| **Serving** | Distributed, sharded ANN, sub-100 ms at planetary scale | Single process, 17 ms, 6k items | Different problem. |
| **Feedback loop** | Continuous online retraining | Batch, offline | No live traffic. |

### The most consequential difference: multi-task ranking

YouTube's 2019 system predicts **many** objectives at once — engagement (clicks, watch time) and
satisfaction (likes, dismissals, surveys) — and combines them with hand-tuned weights, using a
Multi-gate Mixture-of-Experts to stop the objectives fighting each other. They also add a
*shallow tower* fed with position and device, explicitly to absorb selection bias.

That architecture exists because watch time alone is a **flawed proxy for satisfaction** — the
exact weakness documented as [L7](LIMITATIONS.md).

ReelRank now closes part of that gap and is explicit about the part it does not. It predicts the
same six-ish objective families — click, long-watch, completion, liked, satisfied, dismissed —
and combines them with weights, which are chosen **per request** rather than hand-tuned once, so
an evaluator can change the system's objective live.

What is still genuinely different:

| | YouTube (2019) | ReelRank |
|---|---|---|
| Objectives | many, engagement + satisfaction | six, engagement + satisfaction |
| Architecture | shared-trunk **Multi-gate MoE** | **six independent GBDTs** |
| Representation sharing | yes — gates route shared experts per task | **no** — each head re-learns the same structure |
| Weights | hand-tuned, fixed at serving | chosen per request, exposed in the UI |
| Selection bias | a shallow tower fed position + device | IPS weights + a position feature at training time |
| Satisfaction label | real user **surveys** | a simulated survey-like signal |

The MoE is the right answer at their scale: it lets correlated tasks share representation while
gating away destructive interference. Six independent GBDTs cannot share anything, so correlated
tasks re-learn the same structure six times — wasteful, not wrong, and the correct trade at 6k
items with one simulated label stream per outcome. The honest summary is that the *product*
capability is reproduced and the *architecture* is deliberately simpler.

The deeper remaining gap is the label, not the model. YouTube asks people whether they were
satisfied. Inferring satisfaction from engagement metadata has a hard ceiling, which this project
measures directly: clickbait is invisible to every served feature (R² = −0.11), so no objective
weighting reduces it ([F9](TEST_CASES.md)).

---

## Where ReelRank is arguably better

Not many places, but they are real, and they come from having different constraints rather than
better engineering.

1. **Transparency.** Every recommendation exposes its full provenance. YouTube cannot do this at
   scale, and arguably would not want to.
2. **Honest evaluation.** I built an oracle control and used it to show that the standard offline
   protocol is invalid on logged data. Most production teams measure this only when a launch
   surprises them.
3. **Live policy controls.** An evaluator can move λ from 0.72 to 0.0 and watch the page fall
   apart. Exposing the failure surface is a deliberate choice ([D15](DESIGN_DECISIONS.md)).
4. **Reproducibility.** `config.yaml` + a seed regenerates every artifact byte-identically.
5. **No dark patterns.** No autoplay pressure, no engagement-maximising notifications, no
   infinite scroll. Not a technical achievement — a scope one.

---

## Current limitations relative to YouTube

Beyond the model gaps above:

- **Scale**: 6 × 10³ vs 10⁹ items — five orders of magnitude, which changes which algorithms are
  even admissible (brute force → ANN, batch ALS → streaming).
- **No multimodal understanding**: YouTube sees frames, audio and transcripts. I see titles and tags.
- **No social graph**: subscriptions, shares, and comment communities are strong signals I ignore.
- **No context**: time of day, device, location, session length all matter and are unused.
- **No sequence model**: my user is a bag of items with recency decay, not a trajectory.
- **No safety layer**: the single largest responsible-AI gap.

---

## Areas for improvement

Ordered by expected value per unit of work:

1. **Break the monoculture** ([L2](LIMITATIONS.md)) — reserve slots for the best candidate
   *outside* the dominant category rather than using a global relevance floor. Small change,
   large product effect.
2. **Sequential modelling** — a GRU4Rec/SASRec-style encoder over the watch sequence, replacing
   the recency-weighted mean.
3. **Two-tower retrieval with a content-conditioned item tower** — fixes cold items structurally
   (an item's embedding comes from its metadata, so it exists before anyone watches it).
4. **Thompson sampling** for exploration instead of ε-greedy.
5. **Neural text embeddings** as default, once bundle size is not the binding constraint.
6. **Title-vs-content mismatch features** — the one change that would make clickbait *visible*
   to the ranker at all. Currently R² = −0.11 from every served feature, so no objective
   weighting can touch it ([F9](TEST_CASES.md)). This is a feature problem, not a model problem.

> **Already done since this document was first written:** multi-task ranking now ships — six
> calibrated heads (click, long-watch, completion, liked, satisfied, dismissed) over one shared
> feature matrix, with weights chosen per request, and the simulator emits dismissal and
> survey-like satisfaction signals. See [INTENT_AND_OBJECTIVES.md](INTENT_AND_OBJECTIVES.md).
> It raises completion@1 from 0.0055 to 0.0100 (+82%) and makes the objective switchable live
> in the UI. It does **not** reduce clickbait exposure, for the reason in item 6.

---

---

## What a production YouTube-scale system would additionally require

Everything above is about *quality*. This section is about what changes when the
system is real — serving billions of items to billions of people, continuously.
None of it is implemented here, and most of it is not implementable at this
scale; the point of listing it is to be clear about where the boundary is.

**Retrieval at 10⁹ items.** Brute-force nearest-neighbour is *correct* at 6,000
items — 1.5M multiply-adds, well under a millisecond — and inadmissible at 10⁹.
Production needs a sharded ANN index (ScaNN/HNSW), an embedding-refresh pipeline,
and a serving tier that survives index rebuilds without dropping traffic. That is
an infrastructure project, not a modelling one.

**Real-time feedback.** Here, a click updates the feed on the next request because
everything is recomputed from scratch in ~17 ms against static artifacts. At scale
the profile lives in a feature store, the co-visitation counts arrive from a
streaming aggregation (minutes, not the nightly batch this repo does), and the
freshness window for a trending video is measured in minutes. Getting "this video
went viral an hour ago" into recommendations is a data-freshness problem, and it
is most of the engineering.

**Online learning and continuous training.** This model is trained once against a
frozen temporal split. Production retrains continuously — often incrementally —
because the catalog and the audience both drift under it. That brings training/
serving skew monitoring, feature drift detection, automatic rollback, and the
question of what to do when a retrain is *worse*, which offline metrics will not
reliably tell you.

**Contextual bandits for exploration.** The exploration slots here are ε-greedy
with a relevance floor: simple, and it does not learn from what exploration
discovers. A production system runs Thompson sampling or LinUCB so exploration
cost is repaid, and treats the exploration budget itself as a tuned parameter.

**A/B experimentation as the actual arbiter.** Nothing offline settles whether a
change is good — this repo demonstrates that directly, since two valid-looking
metric families [disagree about the sign](EVALUATION.md) of adding the ranker.
Production needs live experiments, interleaving for ranking comparisons,
sequential testing so experiments can stop early, guardrail metrics, and long-horizon
holdouts to catch changes that raise engagement this week and lose users next
quarter.

**Cold-start on all three axes.** New *users* are solved here (ALS fold-in, ~51 µs).
New *items* are not — collaborative signal is structurally unavailable until
somebody watches, which a content-conditioned two-tower model fixes ([F1](TEST_CASES.md)).
New *contexts* — a user's first session on a new device, in a new country — are not
modelled at all.

**Multimodal understanding.** YouTube sees frames, audio and transcripts. This sees
titles, tags and descriptions. That gap is why the clickbait ceiling here is a
feature problem: the mismatch between a thumbnail's promise and a video's content
is not expressible in the data available.

**Safety, fairness and policy.** The largest gap, and it is not a ranking problem.
Production needs a borderline-content classifier, authority signals for topics where
misinformation is costly, creator-exposure fairness constraints (the current Gini of
0.799 would not be defensible in a real creator ecosystem), age-appropriateness gating,
and per-user filter-bubble tracking over time rather than per page. A recommender at
scale is a distribution system with consequences, and none of that follows from
optimising a value score.

**Operational scale.** Multi-region serving, graceful degradation when a recall
source is down (this system already degrades to fused recall order without a
ranker — the right instinct, at toy scale), capacity planning against diurnal
traffic, and cost-per-request budgets that make a 12 ms ranker a real constraint
rather than a comfortable one.

---

## What I would build next, with more time

**Week 1 — fix the bubble and the cold-item hole.**
Per-category exploration floors (L2), and a two-tower retrieval model whose item tower reads
metadata so a brand-new upload has an embedding on day zero (L4). These are the two failures
that most change what the product *feels* like.

**Week 2 — sequence and multi-task.**
Replace the bag-of-items profile with a small transformer over the last 50 watches; extend the
ranker to predict watch time, completion and like-rate jointly with a shared trunk. Extend the
simulator to emit dismissal and survey signals so satisfaction can be modelled separately from
engagement — which is precisely the gap between my system and YouTube's 2019 one.

**Week 3 — make the evaluation trustworthy.**
Implement an interleaving harness and a proper counterfactual off-policy estimator (SNIPS,
doubly-robust). Given that I have already shown full-catalog NDCG to be invalid here, this is
where the real methodological value lies. Add a simulated A/B harness that replays the logging
policy against a new policy, so "would this have been better?" becomes answerable offline with
stated uncertainty.

**Week 4 — the responsible-AI layer.**
A quality/authority classifier, exposure fairness constraints across creators (current Gini
0.799 is not defensible for a creator ecosystem), and a filter-bubble metric tracked per user
over time rather than per page. If this were going in front of real people, this work would move
to week 1.
