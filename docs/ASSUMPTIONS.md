# Assumptions

Every assumption the system rests on, why it was made, and **what breaks if it is wrong**.
Assumptions that are never written down are the ones that quietly become bugs.

---

## About the data

### A1 · Simulated interactions are a valid stand-in for real ones — *for demonstrating algorithms*

**Why:** no public YouTube watch-history dataset exists ([DATASET.md](DATASET.md)).

**If wrong:** every metric in this repo describes the simulator rather than reality. This is
mitigated but not eliminated — the same algorithms are validated on real MovieLens ratings,
where ALS beats popularity by 1.9×, so the *implementations* are sound even if the *data* is not
real.

**Scope of the claim:** the metrics show the algorithms recover latent structure. They are
explicitly **not** a prediction of real-world performance.

### A2 · Users have 2–5 stable-ish interests, drifting slowly

**Why:** matches observed behaviour in published recommender literature; a dense persona would
make every user look average and destroy the CF signal.

**If wrong:** if real taste is far more volatile, our recency half-life (8 positions) is too
long and the system will lag behind users. If it is far more stable, we are over-weighting
recency and under-using long-run history.

### A3 · Watch fraction is a good proxy for satisfaction

**Why:** it is the strongest signal available without asking, and it is what YouTube itself
moved to after click optimisation produced clickbait.

**If wrong — and it is partly wrong.** Watch time rewards long videos and does not distinguish
*engaged* from *unable-to-look-away*. YouTube itself moved past raw watch time to "valued watch
time" using survey data. We have no such signal, so the objective over-rewards long-form
content — visible in `log_duration_min` being a top-3 ranker feature.

### A4 · Video metadata is honest

**Why:** we take titles, tags and categories at face value.

**If wrong:** creators optimise metadata for discovery, and tag-stuffing is common. Content
recall would surface tag-spammed videos. Real YouTube runs abuse detection; we run none.

### A5 · The catalog is static within a session

**Why:** artifacts load once at startup.

**If wrong:** real catalogs grow continuously. New uploads are invisible until the next build.
Production would need incremental indexing (see [FUTURE_WORK.md](FUTURE_WORK.md)).

### A6 · Duration can be imputed from the category median (Kaggle source only)

**Why:** the trending export omits `contentDetails`, and duration drives watch-time and the
duration-fit feature.

**If wrong:** the duration-fit feature becomes noise for that data source, and watch-time
targets are miscalibrated. Only affects `--source kaggle`; the API and synthetic sources carry
true durations.

---

## About the modelling

### A7 · A 96-dimensional latent space is enough

**Why:** 96 factors over 6,000 items and 3,831 users, at 0.54% density. More factors would
overfit sparse data.

**If wrong:** a larger catalog needs more capacity. `als.factors` is a config knob; the cost is
$O(f^2)$ per solve, so 256 factors would be ~7× slower.

### A8 · Position bias follows a cascade model with γ = 0.82

**Why:** it is the standard model (Craswell et al., 2008) and it reproduces the CTR-by-position
curve we observe.

**If wrong:** IPS weights are miscalibrated and position bias is only partly corrected. Real
systems estimate propensities by randomising results; we know them because we generated them —
which makes this assumption *true by construction here* and *unverified in reality*.

### A9 · Unwatched ≠ disliked

**Why:** the foundational assumption of implicit-feedback CF, encoded as confidence
$c_{ui} = 1 + \alpha r_{ui}$.

**If wrong:** it is not wrong — this is one of the few assumptions here that is unambiguously
correct. Treating unwatched as negative is the single most common beginner error in the field.

### A10 · Latent topic structure is recoverable from co-watch behaviour

**Why:** the entire premise of collaborative filtering.

**Verified:** ALS neighbours cross category boundaries 16.4% of the time while staying on-topic
(ground-truth similarity 0.743, vs 0.907 for content which crosses only 1.5%) — it recovered the
bridges without ever seeing the text.

### A11 · The ranker's odds approximate expected watch time

**Why:** Covington et al. (2016); the derivation is in [METHODOLOGY.md](METHODOLOGY.md).

**If wrong:** the approximation degrades when the positive rate is high (it assumes
$N - k \approx N$). Our positive rate is 14.1%, comfortably inside the regime.

---

## About the product

### A12 · Diversity is worth a small accuracy cost

**Why:** accuracy varies by ~11% across the whole λ range while coverage moves 16% — and the
accuracy metric doing the ranking there is the *biased* full-catalog one, which
[EVALUATION.md](EVALUATION.md) shows is not a safe basis for a decision.

**If wrong:** if users genuinely want a monoculture, λ = 1.0 is correct. Only an A/B test can
settle it; it is exposed as a live slider precisely because it is a product judgement, not a
technical one.

### A13 · Freshness matters on YouTube in a way it does not on Netflix

**Why:** the catalog turns over hourly; a 3-year-old video is rarely what someone wants now.

**If wrong:** the freshness boost is capped at +8%, so being wrong is cheap by construction.

### A14 · Explanations improve trust

**Why:** the brief requires evaluators to understand *why* a recommendation appeared, and
explanation is well-established as a trust lever.

**If wrong:** explanations can also *reduce* trust when they expose creepy inference
("because you watched…" applied to sensitive topics). We show provenance for evaluation
purposes; a consumer product would need care here.

### A15 · A stateless browser-held history is acceptable

**Why:** no accounts, no database, trivially horizontally scalable, and privacy-friendly —
nothing about a viewer is persisted server-side.

**If wrong:** it prevents long-term personalisation and cross-device continuity, which are
central to real YouTube. Deliberate for a demo, wrong for a product.

---

## Assumptions I deliberately did **not** make

- **That offline metrics predict online performance.** Explicitly disclaimed; the oracle
  experiment shows why.
- **That more features means a better ranker.** Four features were *removed* after measurement.
- **That the neural embedding model is better.** Tested; TF-IDF+SVD kept as default.
- **That the title is the most important text field.** Measured; the hypothesis lost.
- **That my code was the reason ALS was slow.** Profiled; it was BLAS thread contention.
