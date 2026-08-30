# Future work

Priority order, with the reasoning. Each item names the limitation it closes.

---

## P0 — fix the filter bubble

**Closes:** [L2](LIMITATIONS.md) · [F7](TEST_CASES.md#f7--filter-bubble--a-single-interest-history-yields-a-monoculture)

A 5-video Gaming history produces a 100% Gaming page. Diagnosed precisely: cross-category
candidates are retrieved, but the exploration slots' **global** relevance floor (top half of all
candidates) removes exactly them.

**Change:** apply the relevance floor *per category*, and reserve at least one slot for the
best-scoring candidate outside the dominant category.

```python
# recsys/policy/rerank.py -- replace the global floor
dominant = mode(category_of[picks])
outside  = pool[category_of[pool] != dominant]
floor    = np.quantile(relevance[outside], 0.5)   # floor WITHIN the outside group
```

Cheap, local, and it changes what the product feels like more than anything else on this list.
**Effort: hours.**

---

## P1 — content-conditioned two-tower retrieval

**Closes:** [L4](LIMITATIONS.md) cold items · [L11](LIMITATIONS.md) scale

Replace ALS with a two-tower model: a user tower over watch history, an item tower over
**metadata** (text embedding + category + channel + duration). Trained with in-batch softmax
plus mixed negatives.

The structural win: an item's embedding is a function of its metadata, so **a brand-new upload
has a usable vector before anyone watches it**. That closes the cold-item hole properly instead
of papering over it with a content-similarity fallback. It also gives one embedding space to put
in an ANN index, which is the migration path past 10⁶ items.

**Effort: 2–3 days.** Adds torch to the build (not the serving image — embeddings still ship
as `.npy`).

---

## P2 — sequential user modelling

**Closes:** [L8](LIMITATIONS.md) · [L6](LIMITATIONS.md)

The user is currently a recency-weighted bag of items. Real sessions have direction: trailer →
review → tutorial is a trajectory. A GRU4Rec or SASRec encoder over the last ~50 watches would
capture both session intent and multi-interest structure without the centroid-dilution problem
that a mean vector has by construction.

Also fixes the multi-interest dilution more cleanly than the max-over-history feature currently
patching it.

**Effort: 2–3 days.**

---

## P3 — ~~multi-task ranking~~ **DONE** → make clickbait visible

**Closes:** [L7](LIMITATIONS.md) · the main capability gap vs YouTube 2019

**Shipped.** Six calibrated heads (click, long-watch, completion, liked, satisfied, dismissed)
over one shared feature matrix, combined by weights chosen **per request**; the simulator emits
dismissals and a survey-like satisfaction signal. Completion@1 rose 0.0055 → 0.0100 (+82%) and
the objective is switchable live in the UI ([INTENT_AND_OBJECTIVES.md](INTENT_AND_OBJECTIVES.md)).

**What it did not fix, and what to do instead.** Multi-objective ranking did *not* reduce
clickbait exposure (0.2057 → 0.2044). The diagnosis is the useful part:

| target | GBDT R² from every served feature |
|---|---|
| `latent_quality` | **+0.6355** (visible) |
| `latent_clickbait` | **−0.1122** (invisible) |

**No ranker can optimise an objective absent from its inputs.** So the next step is a *feature*
problem, not a model problem:

1. **Title-vs-content mismatch** — cosine distance between the title embedding and the
   description/tag embedding. A title promising something the body does not deliver is the
   textual signature of clickbait, and it is computable from data already held.
2. **Early-abandon rate** — the fraction of viewers who leave in the first 10 seconds,
   conditioned on having clicked. Bait attracts clicks and loses them immediately.
3. **Title-form features** — all-caps ratio, superlative count, curiosity-gap patterns.

The honest test is whether R² on `latent_clickbait` moves off the floor. If it does, the
existing objective weights will act on it with no model change at all.

**A shared trunk remains the architectural upgrade.** Six independent GBDTs cannot share
representation, so correlated tasks re-learn the same structure six times. A Multi-gate MoE
(Zhao et al., 2019) fixes that — but it needs a neural net and more label volume than one
simulated stream per outcome provides, so it is not worth doing before the feature work above.

**Effort: 1–2 days** for the features; the MoE is a larger project.

---

## P4 — pairwise / listwise ranking

**Closes:** [L13](LIMITATIONS.md)

LambdaMART optimises NDCG directly and would likely beat the pointwise model on ranking metrics.
The reason it is not P0: the watch-time weighting depends on the score being *calibrated*
(expected watch seconds), and Stage 3 blends that score with freshness and diversity terms. A
pairwise margin has no such interpretation.

The honest experiment is to run both and compare on Protocol A, accepting that the winner may
need Stage 3 restructured.

**Effort: 1 day.**

---

## P5 — Thompson sampling for exploration

**Closes:** [L14](LIMITATIONS.md)

Maintain a Beta posterior per item (or per item-cluster) over engagement rate; sample from it
when filling exploration slots. Explores in proportion to **uncertainty** rather than uniformly
over novelty, which is strictly more sample-efficient.

Needs an online feedback loop to be worth much, so it pairs naturally with P7.

**Effort: 1 day.**

---

## P6 — trustworthy off-policy evaluation

**Closes:** [L3](LIMITATIONS.md)

Given that this project already demonstrated full-catalog NDCG to be invalid on logged data,
this is where the real methodological value sits:

- **SNIPS / doubly-robust estimators** for off-policy value, with confidence intervals.
- **A simulated A/B harness** — replay the logging policy against a new policy through the same
  user model, so "would this have been better?" is answerable with stated uncertainty rather
  than vibes.
- **Interleaving** (team-draft) for the online case.

**Effort: 2–3 days.**

---

## P7 — online learning loop

**Closes:** [L12](LIMITATIONS.md) · [L17](LIMITATIONS.md)

Incremental catalog indexing, periodic ALS refresh with warm start, streaming co-visitation
counts, and a feedback channel from the UI so the deployed demo learns from evaluator clicks.
Turns the demo from a static artifact into a live system — and makes P5 meaningful.

**Effort: 3–4 days.**

---

## P8 — responsible-AI layer

**Closes:** [L15](LIMITATIONS.md) · [L16](LIMITATIONS.md)

- Quality / authority classification, with boosting for authoritative sources on news and health.
- Exposure-fairness constraints across creators — the current Gini of 0.799 is not defensible for
  a creator ecosystem.
- A **per-user, longitudinal** filter-bubble metric: how much has this person's topic entropy
  narrowed over 30 days? Page-level diversity does not capture drift over time.
- Dismissal / "not interested" handling as a negative signal.

If this system were going in front of real people, this moves to P0. It sits here only because
the deliverable is an evaluation harness.

**Effort: 1 week+.**

---

## Smaller items

| Item | Closes | Effort |
|---|---|---|
| Neural text embeddings as default | [L5](LIMITATIONS.md) template trap | 1 h (already a config flag) |
| FAISS/HNSW index behind the `RecallResult` interface | [L11](LIMITATIONS.md) | 0.5 day |
| Learned RRF weights instead of hand-set | — | 0.5 day |
| Per-user diversity personalisation (some people *want* a monoculture) | [L2](LIMITATIONS.md) | 1 day |
| Context features: time of day, device, session length | [L9](LIMITATIONS.md) | 1 day |
| Redis-backed history for cross-device continuity | [L9](LIMITATIONS.md) | 0.5 day |
| Template n-gram mining and down-weighting | [L5](LIMITATIONS.md) | 0.5 day |
| Prometheus metrics + structured request logging | [L17](LIMITATIONS.md) | 0.5 day |
