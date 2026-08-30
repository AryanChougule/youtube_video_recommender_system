---
title: ReelRank - YouTube-style Recommender
emoji: 📺
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Two-stage video recommender with explainable results
---

# ReelRank

A YouTube-style video recommendation system: **candidate generation → ranking → policy**,
built from scratch, with a testing UI that shows *why* every recommendation appeared.

Click a persona in the sidebar, or click any video to "watch" it and see the feed re-rank live.
Every card has a **"Why this video?"** panel showing which recall source proposed it, at what
rank, what the ranker scored it, and which policy rules fired.

## The three stages

| Stage | What it does | Budget |
|---|---|---|
| **1 · Candidate generation** | 5 retrievers (content embeddings, co-visitation, implicit ALS, channel affinity, trending) fused with Reciprocal Rank Fusion | ~3 ms |
| **2 · Ranking** | Gradient-boosted model over 19 features; positives weighted by watch time so the odds estimate *expected watch time*, not click probability | ~12 ms |
| **3 · Policy** | MMR diversity, hard channel cap, freshness boost, reserved exploration slots | ~5 ms |

Stage 2 also runs **six calibrated objective heads** — click, long-watch, completion, liked,
satisfied and dismissed — over the same feature matrix, combined by weights chosen *per
request*. The **Recommendation Lab** in the sidebar changes those weights live, so the
system's objective can be switched from engagement-maximising to satisfaction-maximising with
no retraining. Every card's "Why this video?" panel shows all six probabilities with their
weights and contributions.

## Important: the interactions are simulated

No public dataset of real YouTube watch histories exists — watch history is among the most
re-identifying data a platform holds. Video metadata is real-or-synthetic (configurable), but
**user behaviour comes from an explicit, documented simulator** with a cascade click model,
position bias, popularity bias and taste drift.

The metrics demonstrate that the algorithms recover the latent structure that generated the
data. They are **not** a prediction of real-world YouTube performance. The same algorithms are
separately validated on MovieLens-100k (real human ratings), where they beat a popularity
baseline by 1.9×.

Full write-up, including a demonstration that the *standard* offline evaluation protocol is
invalid on logged data, is in the source repository.

## Try these

- **Cold start** → clear the history; the system correctly says "Trending in X" rather than
  faking personalisation.
- **Split interests** → 3 Food + 3 Gaming; watch both survive.
- **MMR λ → 0** → break the page on purpose and see relevance collapse.
- **Search "sourdough bread baking"** → none of the top hits contain the word "sourdough".
