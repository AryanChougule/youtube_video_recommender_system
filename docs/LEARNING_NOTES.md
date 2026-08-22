# Recommender systems: the course

The theory behind every part of this repository, in the order it was built. Read alongside the
source — each module names the files it produced.

Structure: **Lecture** (concept) → **Lab** (what we built) → **Checkpoint** (what you should be
able to answer unaided).

---

# Module 0 · What a recommender actually is

### The one-sentence definition

> A recommender system is a **ranking function** `f(user, item, context) → score`, applied to a
> catalog and sorted.

Everything else — matrix factorisation, transformers, two-tower networks — is a different way of
*parameterising and learning `f`*. Hold onto this; it stops the field feeling like a bag of tricks.

### The four paradigms

**1 · Content-based** — recommend items similar to what you liked, where similarity comes from
**item attributes**.
`score(u,i) = sim(profile(u), features(i))`
✅ works for brand-new items; explainable. ❌ **filter bubble** — structurally incapable of
surprising you.

**2 · Collaborative filtering** — recommend items liked by people similar to you, using **only**
the interaction matrix. Never looks at content.
✅ discovers latent, non-obvious taste. ❌ **cold start**; popularity-biased.

**3 · Knowledge / rule-based** — hand-written business logic ("boost fresh uploads", "max 2 per
channel").
✅ instant, controllable, encodes strategy. ❌ doesn't learn.

**4 · Hybrid** — combine them. **Every production system is a hybrid**, because each paradigm's
weakness is another's strength:

| | Cold item | Cold user | Serendipity | Explainable |
|---|---|---|---|---|
| Content | ✅ | ❌ | ❌ | ✅ |
| CF | ❌ | ❌ | ✅ | ⚠️ |
| Rules | ✅ | ✅ | ❌ | ✅ |
| **Hybrid** | ✅ | ✅ | ✅ | ✅ |

### Three ideas that separate amateurs from professionals

**① Implicit ≠ explicit feedback.** Ratings are explicit; watches are implicit. The critical
difference: **implicit feedback has no negatives.** Not watching something means "hate it" *or*
"never saw it". Treating unwatched as dislike is the single most common beginner error.

**② You train on data your own system generated.** Users only click what you showed them. Your
logs are a biased sample of your own past policy. This is the **feedback loop**, and it is why
offline metrics lie.

**③ Optimising clicks is a trap.** YouTube optimised CTR and got clickbait; they switched to
watch time in 2012. **Your loss function is your product strategy.**

**Checkpoint 0** — Why can a pure content-based system never surprise a user? Why is "didn't
click" not a negative label?

---

# Module 1 · Data, and the problem nobody mentions

**Lab:** [`data/topics.py`](../src/recsys/data/topics.py),
[`synthetic.py`](../src/recsys/data/synthetic.py),
[`simulator.py`](../src/recsys/data/simulator.py)

### The uncomfortable truth

Video metadata is plentiful. **User watch histories do not exist publicly and never will** —
Netflix released anonymised ratings in 2006 and researchers de-anonymised them within weeks via
IMDb cross-referencing.

Three honest options: relabel MovieLens (a movie recommender in costume), go content-only
(delete the interesting half), or **use real metadata with explicitly simulated interactions**.

### Why a simulator is legitimate — when done right

A simulator is not "fake data". It is **a hypothesis about user behaviour, written in code**.
Its advantage: the latent structure is *known*, so you can ask a question real logs cannot
answer — *did the model recover genuine preference structure, or just memorise popularity?*

Its danger is **circularity** — a low-rank model trivially fits a low-rank generator. Defend
against it by (a) making the generator harder than any model you fit to it, (b) preserving
presentation bias, (c) validating the same algorithms on real data.

### The design that makes it interesting: latent micro-topics

A naive simulator says *"user likes Gaming → show Gaming"*. Useless — a category lookup solves
it and CF has nothing to discover.

Instead: 40 latent micro-topics; categories are **overlapping** distributions over them. A
"Budget Gaming PC Build" genuinely sits between Gaming and Tech. Content filtering can't see the
bridge (different words); category rules can't (different labels); **CF can** — and measurably
does.

**Checkpoint 1** — Why does a simulator with known ground truth permit a *stronger* evaluation
than real logs? What is the circularity risk?

---

# Module 2 · Content-based filtering

**Lab:** [`features/text.py`](../src/recsys/features/text.py),
[`recall/content.py`](../src/recsys/recall/content.py)

### TF-IDF

$$\text{tf}(t,d) = 1 + \log\text{count}(t,d) \qquad \text{idf}(t) = \log\frac{N}{\text{df}(t)} + 1$$

Sublinear TF: a word appearing 20× isn't 20× more important. **IDF is what makes rare words
matter** — a term in 5 of 6000 docs gets idf ≈ 8.1; a term in every doc gets ≈ 1.

### Cosine, not Euclidean

Two sourdough videos with 40-word and 400-word descriptions have very different vector
*magnitudes* but the same *direction*.

$$\cos(a,b) = \frac{a \cdot b}{\|a\|\|b\|}$$

**Engineering trick:** L2-normalise once at build time and cosine becomes a plain dot product.
This is why every vector database stores normalised vectors.

### The sparsity problem, and LSA

TF-IDF fails on **vocabulary mismatch**: "GPU benchmark" and "graphics card performance test"
share zero terms → similarity 0. Truncated SVD fixes it:

$$X \approx U_k\Sigma_k V_k^\top, \qquad \text{item vector} = U_k\Sigma_k$$

Terms co-occurring with the same neighbours collapse onto nearby latent dimensions. Applied to
text this is **Latent Semantic Analysis** (1990, still a strong baseline).

> **The general lesson:** dimensionality reduction is not only about speed. It **forces
> generalisation** — squeezing data through a bottleneck makes the model discover that different
> words mean the same thing.

### The architectural rule

> **Encode offline. Serve with a dot product.**

The heavy model runs once at build time and writes a matrix. Serving needs numpy and nothing
else. This is why our image is 450 MB not 2.5 GB.

**Checkpoint 2** — Why does L2-normalising at build time let you replace cosine with a dot
product? What specific failure does SVD fix?

---

# Module 3 · Collaborative filtering

**Lab:** [`recall/cf.py`](../src/recsys/recall/cf.py)

### Item–item CF ("users who watched this also watched")

Count co-occurrence in sessions: $c_{ij}$. **Raw counts are useless** — if $j$ is viral,
$c_{ij}$ is large for *every* $i$. Normalise:

| Normaliser | Formula |
|---|---|
| Cosine | $c_{ij}/\sqrt{c_ic_j}$ |
| Jaccard | $c_{ij}/(c_i+c_j-c_{ij})$ |
| PMI | $\log\frac{p_{ij}}{p_ip_j}$ |
| **Damped** | $c_{ij}/(c_i^\alpha c_j^{1-\alpha})$ |

Computational trick: $C = S^\top S$ gives every count in one sparse matmul. No loops.

### Matrix factorisation — the big one

Learn $x_u, y_i \in \mathbb{R}^f$ with $r_{ui} \approx x_u^\top y_i$. Compressing a 3799×6000
matrix into 3799×96 + 6000×96 **is** the learning — with 96 dimensions the model is forced to
discover that a dimension means something like "long-form technical content".

### Implicit feedback: the reframing that matters

You cannot minimise error over observed entries only — ignore the zeros and the model collapses
to "everything is loved". Hu, Koren & Volinsky (2008):

$$p_{ui} = \mathbb{1}[r_{ui}>0] \qquad c_{ui} = 1 + \alpha r_{ui}$$

$$\min_{X,Y}\sum_{u,i} c_{ui}(p_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|^2+\|Y\|^2)$$

- $p$ — **preference**: did they engage at all?
- $c$ — **confidence**: how sure are we? A zero gets confidence 1 (weak); a 90%-watched video
  gets $1+30(0.9)=28$.

The sum runs over **all** pairs. That single reframing is the most important idea in
implicit-feedback recommendation.

### Why *Alternating* Least Squares

Non-convex jointly; **fix one side and it's convex least squares** with a closed form:

$$x_u = (Y^\top C^u Y + \lambda I)^{-1} Y^\top C^u p(u)$$

No learning rate, guaranteed monotone decrease. The trick that makes it practical:

$$Y^\top C^u Y = \underbrace{Y^\top Y}_{\text{once}} + Y^\top(C^u - I)Y$$

$C^u - I$ is zero wherever the user didn't interact, so the second term touches only their ~30
items. $O(n_\text{items}f^2) \to O(n_u f^2)$ — a 200× saving.

**Fold-in** reuses the same closed form with frozen item factors: a brand-new user gets a vector
in one 96×96 solve. This is how cold *users* are solved.

**Checkpoint 3** — Why can't you train on observed entries only with implicit feedback? What
does the $Y^\top Y$ precomputation actually save?

---

# Module 4 · Candidate generation & fusion

**Lab:** [`recall/blend.py`](../src/recsys/recall/blend.py),
[`heuristic.py`](../src/recsys/recall/heuristic.py)

Stage 1's job is **recall, not precision**. Being wrong here is unrecoverable — the ranker can
only reorder what it's given.

### The fusion problem

Content cosine ∈ [0,1]; damped co-visitation ∈ [0,~0.5]; ALS dot products unbounded. Adding them
is adding metres to kilograms.

**Reciprocal Rank Fusion** throws away scores and keeps ranks:

$$\text{RRF}(d) = \sum_s \frac{w_s}{K + \text{rank}_s(d)}, \quad K = 60$$

Scale-free, no calibration, survives distribution drift. $K$ flattens the head of the curve so
rank 1 vs 2 is a nudge, not a cliff. Same technique hybrid search uses to fuse BM25 with vector
retrieval.

**Cost:** discards magnitude — recovered by feeding raw per-source scores to the ranker. *Fusion
decides what gets considered; the ranker decides what wins.*

**Checkpoint 4** — Why is a Stage 1 mistake unrecoverable? What does $K$ buy you?

---

# Module 5 · Learning to rank

**Lab:** [`rank/features.py`](../src/recsys/rank/features.py),
[`dataset.py`](../src/recsys/rank/dataset.py), [`ranker.py`](../src/recsys/rank/ranker.py),
[`crossfit.py`](../src/recsys/rank/crossfit.py)

### The objective is the product decision

Train a **weighted** classifier: positives weighted by watch seconds, negatives by 1. Then

$$\frac{P}{1-P} = \frac{\sum_i T_i}{N-k} \approx \mathbb{E}[T]$$

**The odds of a weighted click-classifier estimate expected watch time.** You get a watch-time
model out of a click model, for free. Rank by odds, not probability.

### Negatives

- **In-feed** negatives (things they saw and skipped) are *hard* negatives.
- **Random catalog** negatives cover the regime that dominates at serving, where you score
  everything.

Only in-feed → out-of-distribution at serving. Only random → you learn "plausible vs arbitrary",
a much easier and less useful task. **Use both.**

### Position bias

Our logs: 52% CTR at rank 0, 0.1% at rank 7. That is not a statement about the videos.
**Inverse Propensity Scoring**: weight clicks by $1/P(\text{examined at }r)$. A click at rank 6
is much stronger evidence than one at rank 0.

### The two failure modes that kill real systems

**Train/serve skew** — training and serving computing "the same" feature differently. Defence:
one `FeatureBuilder` used by both.

**Causality** — for a row at time $t$, the history must contain only clicks before $t$. Replay
timelines forward; append to history only *after* featurising.

### The stacking leak (learned the hard way here)

Stage 2 uses Stage 1's outputs as features. Even with a correct temporal split, training rows
are **in-sample** for ALS:

```
als_score AUC on training rows : 0.856
als_score AUC held out         : 0.584
```

The ranker over-trusted it and scored *worse than its own best single feature*. Fix:
**cross-fitting** — K folds by user, refitting CF with each fold's users held out.

> **The rule:** any upstream model's output used as a downstream feature must be out-of-fold.
> This is not recommender-specific; it is why stacking ensembles use out-of-fold predictions.

**Checkpoint 5** — Why do weighted-logistic odds approximate expected watch time? Why are logged
impressions better negatives than random items?

---

# Module 6 · Policy — diversity, novelty, serendipity

**Lab:** [`policy/rerank.py`](../src/recsys/policy/rerank.py)

Rank purely by predicted engagement and you get ten near-identical videos. Each is individually
optimal; the *page* is terrible. **Page quality is a property of the set**, which pointwise
scoring cannot see.

| Property | Definition | Why |
|---|---|---|
| Diversity | items differ from each other | avoids ten sourdough videos |
| Novelty | $-\log_2 p(i)$ | avoids an all-blockbuster page |
| **Serendipity** | relevant **and** not what trending would show | the actual point of the product |

### Maximal Marginal Relevance

$$\text{MMR} = \arg\max_{d \notin S}\big[\lambda\,\text{rel}(d) - (1-\lambda)\max_{d_j\in S}\text{sim}(d,d_j)\big]$$

The second term is the insight: redundancy is priced **contextually**, against what's already on
the page.

### Exploration is not optional

Your model recommends what it's confident about → users click those → that becomes your training
data → the model gets more confident. The long tail goes permanently dark. **You cannot learn a
video is good if you never show it.**

Options: ε-greedy (crude, unbiased — the only clean data you'll ever get), **Thompson sampling**
(explores in proportion to *uncertainty*, far more efficient), UCB.

**Checkpoint 6** — Why can't a pointwise ranker produce a diverse page no matter how accurate?
What breaks if you never explore?

---

# Module 7 · Evaluation — how not to fool yourself

**Lab:** [`evaluate.py`](../src/recsys/evaluate.py),
[`counterfactual.py`](../src/recsys/counterfactual.py), [`metrics.py`](../src/recsys/metrics.py)

### Protocol beats metric

Random split ❌ (predicts the past from the future) → per-user leave-one-out ⚠️ → **global
temporal split** ✅. And **every model in the pipeline must respect the same cutoff**.

### The metrics, and what each hides

| Metric | Question | Blind spot |
|---|---|---|
| Precision@K | of K shown, how many watched? | ignores order |
| Recall@K | of what they watched, how many shown? | punishes heavy users |
| NDCG@K | right things, right order, graded | needs graded relevance |
| MAP@K | precision at each hit | binary only |
| MRR | how deep to the first hit? | ignores hits 2..K |

Beyond accuracy: **coverage**, **Gini**, **novelty**, **intra-list diversity**, **serendipity**.
Reporting accuracy alone is how you ship a filter bubble and call it a win — our popularity
baseline reaches 0.5% coverage with Gini 0.997 while "winning" on NDCG.

### The most important lesson in this whole course

The standard offline protocol **measures the logging policy and retrieval breadth, not ranking
quality**. The decisive evidence: adding the learned ranker moves two metric families in
opposite directions.

```
                        full-catalog NDCG@10    Protocol A top-1
Stage 1 recall only            0.0125               0.1840
+ learned ranker               0.0107  ↓            0.1930  ↑
```

Two metrics disagreeing about the same change means one is wrong for the purpose. And per page,
the popularity scorer lands *below random* (0.1323 vs 0.1415) while scoring near the top on the
full-catalog metric — opposite verdicts on the identical scorer. Users can only click what they
were shown.

**Two protocols that survive this:**
- **A · Re-rank logged impressions** — only reorder what was actually shown; every label
  observed. Best offline predictor of online lift.
- **B · Sampled candidate sets** — 1 positive vs N negatives. Isolates ranking from retrieval,
  but flatters weak models (Krichene & Rendle 2020), so never quote it alone.

**Neither replaces an online A/B test. Nothing offline does.**

**Checkpoint 7** — Why does a popularity baseline beat a good recommender on logged data? How
would you prove your offline metric is trustworthy?

---

# Module 8 · Serving

**Lab:** [`engine.py`](../src/recsys/engine.py), [`artifacts.py`](../src/recsys/artifacts.py),
[`api/app.py`](../src/recsys/api/app.py)

- **Load once at startup**, never per request.
- **Stateless** — history posted with each request; scales horizontally, stores no PII.
- **Assert artifact integrity at load.** A row-order mismatch doesn't crash — it silently
  recommends the wrong videos forever. Fail loudly at startup instead.
- **Budget your latency deliberately.** Ours: recall 3 ms, ranking 12 ms, policy 5 ms. Ranking
  dominates *because that's where precision is bought*.

### A performance lesson that isn't about recommenders

ALS took 649 s. A single 96×96 `np.linalg.solve` was costing **3.65 ms** — ~0.16 GFLOPS.
Cause: OpenBLAS spawning a thread team for a matrix far too small to amortise it. Pinned to one
thread: **51 µs**. Total: **18 s**.

> **Profile before optimising.** The bug was not in my code, and no amount of algorithmic
> cleverness would have found it.

---

# Module 9 · Product thinking

- **Explanations are a feature.** Provenance recorded during fusion, not reconstructed
  afterwards. An explanation that misdescribes its own evidence is worse than none — we shipped
  a bug where watch-page items claimed "viewers with a taste profile like yours" when no profile
  existed.
- **Expose the knobs.** Letting an evaluator set λ = 0 and break the page is more honest than
  hiding the failure surface.
- **State your limitations.** They are the part of the work that proves you understand it.

---

## If you build one from scratch, in this order

1. **Data + honest provenance.** Decide what's real and what isn't, and write it down first.
2. **A popularity baseline.** Everything must beat it. Many things won't.
3. **Content vectors.** Cheap, no cold start, immediately useful.
4. **Co-visitation.** ~30 lines, and often the strongest single signal.
5. **ALS.** Where the magic is. Implement it once yourself.
6. **Fuse with RRF.** Don't normalise scores.
7. **Evaluate — and check your protocol is valid** before believing any number.
8. **Then** add a ranker. Cross-fit its upstream features.
9. **Then** policy: diversity, freshness, exploration.
10. **Then** UI and explanations.

Steps 7 and 8 are where most projects go wrong, and where the interesting engineering is.

## Papers worth reading, in order

1. Hu, Koren & Volinsky (2008) — *Collaborative Filtering for Implicit Feedback Datasets*.
   The confidence-weighting idea. Read this first.
2. Davidson et al. (2010) — *The YouTube Video Recommendation System*. Co-visitation at scale.
3. Covington, Adams & Sargin (2016) — *Deep Neural Networks for YouTube Recommendations*.
   Two-stage architecture; the watch-time weighting trick.
4. Joachims et al. (2017) — *Unbiased Learning-to-Rank with Biased Feedback*. IPS.
5. Zhao et al. (2019) — *Recommending What Video to Watch Next*. Multi-task ranking, MMoE.
6. Krichene & Rendle (2020) — *On Sampled Metrics for Item Recommendation*. Why your evaluation
   is probably wrong.
7. Carbonell & Goldstein (1998) — *The Use of MMR…*. Diversity, and still the standard.
