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
| **Models** | Deep neural nets end-to-end | Matrix factorisation + GBDT | 3,803 users cannot train a deep net. MF is the right capacity for this data size, and the maths is inspectable. |
| **Objectives** | Multi-task (Zhao 2019): watch time, likes, shares, dismissals, survey responses, with a Mixture-of-Experts head and a shallow tower for position bias | Single objective (watch time) | Multi-task needs multiple label streams. I have one. |
| **Signals** | Audio/video content, thumbnails, ASR transcripts, watch-time curves, subscriptions, survey responses, demographics | Text metadata + simulated watch logs | Availability. |
| **Explanations** | Minimal ("Recommended for you") | Every item names its recall source, rank, ranker score and policy actions | An evaluation harness has the opposite requirement to a consumer product: it must be legible. |
| **Diversity** | Learned, and partly a policy/regulatory matter | Explicit MMR with a live λ slider | Transparency over sophistication. |
| **Exploration** | Large-scale bandits with real traffic | 2 reserved slots, ε-greedy | No traffic to learn from. |
| **Safety** | Borderline-content classifiers, authoritative sources for news/health, age gating | **None** | Out of scope, and a genuine gap ([L15](LIMITATIONS.md)). |
| **Freshness** | Minutes | Rebuild cycle | Static artifacts. |
| **Serving** | Distributed, sharded ANN, sub-100 ms at planetary scale | Single process, 22 ms, 6k items | Different problem. |
| **Feedback loop** | Continuous online retraining | Batch, offline | No live traffic. |

### The most consequential difference: multi-task ranking

YouTube's 2019 system predicts **many** objectives at once — engagement (clicks, watch time) and
satisfaction (likes, dismissals, surveys) — and combines them with hand-tuned weights, using a
Multi-gate Mixture-of-Experts to stop the objectives fighting each other. They also add a
*shallow tower* fed with position and device, explicitly to absorb selection bias.

That architecture exists because watch time alone is a **flawed proxy for satisfaction** — the
exact weakness I document as [L7](LIMITATIONS.md). Mine is a one-objective system, which is a
real capability gap and not just a scale difference.

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
3. **Multi-task ranking** — predict watch time *and* like-rate *and* completion, then combine.
   Needs richer labels, which the simulator can already produce.
4. **Two-tower retrieval with a content-conditioned item tower** — fixes cold items structurally
   (an item's embedding comes from its metadata, so it exists before anyone watches it).
5. **Thompson sampling** for exploration instead of ε-greedy.
6. **Neural text embeddings** as default, once image size is not the binding constraint.

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
