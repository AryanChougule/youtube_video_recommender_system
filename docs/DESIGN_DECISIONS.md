# Key design decisions

Each decision, the alternatives rejected, and the reason. A decision without a rejected
alternative is not a decision — it is a default.

---

## D1 · Two-stage architecture instead of one model

**Chosen:** cheap high-recall candidate generation → expensive high-precision ranking → policy.

**Rejected:** a single model scoring the whole catalog.

**Why:** you cannot run a 19-feature gradient-boosted model over 6,000 items in 20 ms, let alone
over millions. Splitting lets each stage optimise a different objective — recall then precision
— and lets us spend 55% of the latency budget where it buys the most.

**Cost:** a mistake in Stage 1 is unrecoverable. The ranker can only reorder what it is given.

---

## D2 · Real metadata + simulated interactions

**Chosen:** honest hybrid, loudly documented.
**Rejected:** MovieLens relabelled as videos; a content-only system.

**Why:** MovieLens has no channels, no watch-time, no Shorts, no freshness dynamics — the things
that make video recommendation *video* recommendation. Content-only deletes collaborative
filtering, which is the intellectually interesting half.

**Cost:** the metrics cannot claim real-world validity. Mitigated by validating the same
algorithms on real MovieLens ratings.

---

## D3 · Implement ALS, co-visitation, ANN search and MMR from scratch

**Chosen:** ~400 lines of NumPy/SciPy.
**Rejected:** `implicit`, `faiss`, `lightgbm`.

**Why:** three reasons, in order of weight. (1) The maths is the deliverable — a reviewer can
read the confidence-weighting and the $Y^\top Y$ trick directly. (2) Dependency weight: the
deployed image is ~450 MB instead of ~2.5 GB. (3) At 6,000 items, brute-force search is
sub-millisecond, so FAISS would buy nothing and cost exactness.

**Cost:** would not survive 10⁶ items unchanged. The migration path is documented, and the
`RecallResult` interface is designed so swapping in FAISS touches one class.

---

## D4 · TF-IDF + SVD as the default text backend

**Chosen:** classical LSA, 256 dims.
**Rejected:** sentence-transformers as default (kept as a one-line opt-in).

**Why:** on 6k short, keyword-dense documents the neural gain is small, while it adds ~2 GB and
requires torch at query time for free-text search. Knowing when *not* to use the fancier model
is part of the job.

**Cost:** paraphrase handling is weaker, and the template trap in
[TEST_CASES.md#f4](TEST_CASES.md#f4--the-tf-idf-template-trap) is a direct consequence.

---

## D5 · Reciprocal Rank Fusion instead of score normalisation

**Chosen:** fuse on rank.
**Rejected:** min-max / z-score blending.

**Why:** the three sources produce incomparable scales, and RRF is scale-free, needs no
calibration, and survives distribution drift on every rebuild.

**Cost:** discards score magnitude — recovered by passing raw per-source scores to the ranker.

---

## D6 · Watch-time-weighted odds instead of click probability

**Chosen:** weight positives by watch seconds; rank by odds.
**Rejected:** plain click prediction (still available as `ranker.objective: click`).

**Why:** click optimisation produces clickbait. This is the single most consequential objective
change in YouTube's own history, and the weighted-LR trick gets a watch-time model out of a
click model for free.

**Cost:** over-rewards long-form content — `log_duration_min` is a top-3 feature. YouTube's own
answer (survey-based "valued watch time") needs a signal we do not have.

---

## D7 · Pointwise ranking instead of pairwise/listwise

**Chosen:** pointwise binary classifier.
**Rejected:** LambdaMART / RankNet.

**Why:** the watch-time weighting only works pointwise — it gives the score a *calibrated*
meaning (expected watch seconds). Stage 3 blends that score with freshness and diversity terms,
and you cannot sensibly blend an uncalibrated pairwise margin with anything.

**Cost:** pairwise would likely win on NDCG. Top-3 item in [FUTURE_WORK.md](FUTURE_WORK.md).

---

## D8 · One global temporal cutoff, respected by every model

**Chosen:** a single cutoff in [`split.py`](../src/recsys/split.py), asserted at load time.
**Rejected:** random split; per-model splits; training CF on everything.

**Why:** this was learned the hard way. Training ALS on the full log while testing the ranker on
a holdout produced AUC 0.957 — pure leakage. See
[EVALUATION.md Finding 1](EVALUATION.md#finding-1--the-leakage-that-made-the-first-model-look-excellent).

**Cost:** the deployed models see only 80% of the data. Deliberate: the numbers in the docs
describe exactly the artifact that is running. `--full` refits on everything for a real
production deploy, and records that fact so the metrics are flagged as invalid.

---

## D9 · Cross-fitted CF features for ranker training

**Chosen:** 4-fold cross-fitting by user.
**Rejected:** using the production CF models directly (simpler, and wrong).

**Why:** even with a correct temporal split, ranker training rows are **in-sample** for ALS
(`als_score` AUC 0.856 in-sample vs 0.584 held out). The ranker over-trusted it and scored
*worse than its own best single feature*.

**Cost:** 4× the CF training time (~60 s). Trivially worth it.

---

## D10 · Mixed negative sampling

**Chosen:** in-feed negatives (hard) **plus** random catalog negatives (2 per positive).
**Rejected:** in-feed only; random only.

**Why:** in-feed negatives alone leave the model out-of-distribution when scoring the full
catalog at serving time. Adding random negatives moved AUC 0.643 → 0.697.

---

## D11 · Category-aware MMR similarity

**Chosen:** `sim = 0.75·cosine + 0.25·[same category]`.
**Rejected:** pure text cosine.

**Why:** measured. Pure cosine returned 7 of 8 slots as Food from a 3-Food/2-Gaming history;
category-aware returned 5/3 and raised intra-list diversity 0.737 → 0.855. Category is the
coarse signal a viewer actually perceives as "more of the same".

---

## D12 · Channel cap as a hard constraint

**Chosen:** masking during MMR, and re-checked during exploration.
**Rejected:** a soft penalty term.

**Why:** "at most 2 per creator" is a product rule, not a preference. Soft penalties leak —
and in fact the first implementation leaked anyway, because exploration slots were selected
after MMR without consulting the budget. Caught by a test.

---

## D13 · Stateless serving, history in the browser

**Chosen:** post the history with every request.
**Rejected:** server-side sessions or a user database.

**Why:** horizontally scalable, no persistence layer, no PII stored, and an evaluator can hard
refresh without losing anything. Fold-in makes it cheap — a brand-new user's latent vector costs
one 96×96 solve.

**Cost:** no cross-device or long-term personalisation. Right for a demo, wrong for a product.

---

## D14 · Vanilla JS for the UI

**Chosen:** ~400 lines of plain JS, no build step.
**Rejected:** React/Next.js.

**Why:** no bundler, no `node_modules` in the image, no framework version to rot, and the
Dockerfile stays a single Python stage. The UI's job is to make recommendations inspectable,
not to demonstrate a frontend stack.

---

## D15 · Ship the failure cases in the UI

**Chosen:** live sliders that let an evaluator set λ = 0 and break the page on purpose.
**Rejected:** clamping the ranges to "safe" values.

**Why:** a system whose failure modes are reachable is more honest than one that hides them.
The brief explicitly says understanding limitations is a strength.
