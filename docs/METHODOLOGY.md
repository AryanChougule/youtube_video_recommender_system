# Recommendation methodology

Every algorithm, the maths behind it, and what it was chosen over.

---

## Stage 1 · Candidate generation

### 1.1 Content-based — TF-IDF → Truncated SVD → cosine

Video text (title, tags, channel, category, description) → sparse TF-IDF → 256-dim dense
vectors → L2-normalised so cosine similarity is a plain dot product.

$$\text{tf}(t,d) = 1 + \log \text{count}(t,d), \qquad \text{idf}(t) = \log\frac{N}{\text{df}(t)} + 1$$

Sublinear TF because a term appearing 20× is not 20× more important. IDF is what makes rare,
discriminative terms (`sourdough`) outweigh ubiquitous ones (`video`).

Raw TF-IDF has a fatal weakness: **vocabulary mismatch.** "GPU benchmark" and "graphics card
performance test" share zero terms, so cosine = 0, despite being about the same thing.
Truncated SVD fixes it — $X \approx U_k\Sigma_k V_k^{\top}$, keeping $k=256$ directions
(explained variance 48.6%). Terms that co-occur with the same neighbours collapse onto nearby
latent dimensions. Applied to text this is **Latent Semantic Analysis**.

**Which text fields to index was measured, not assumed.** The intuitive choice — repeat the
title, since creators optimise titles hardest — *loses*. Scored by "what fraction of a video's
top-10 text neighbours share its true latent topic mixture":

| variant | topical precision@10 |
|---|---|
| title ×2 + tags | 0.8875 |
| **title ×1 + tags** | **0.9040** ← chosen |
| title ×1 + tags ×3 | 0.9028 |
| tags only | 0.8863 |
| title only | 0.6518 |
| random pair baseline | 0.0943 |

Titles follow strong *format* templates ("… Ranked From Worst to Best"), so up-weighting them
retrieves videos with the same title **shape** rather than the same subject. Reproduce with
`python scripts/07_ablate_text_fields.py`.

**Rejected:** sentence-transformers as the default. Better at paraphrase, worse at rare
technical terms, and it adds ~2 GB plus a torch dependency at query time for a small gain on
6k short keyword-dense documents. Available via `features.text_backend: sentence_transformers`.

### 1.2 Item–item co-visitation

How often two videos appear in the same session. Computed as one sparse matmul:

$$C = S^{\top}S$$

where $S$ is the session×item binary matrix. No loops.

**Raw counts are useless**, and understanding why is the whole lesson: if item $j$ is viral,
$c_{ij}$ is large for *every* $i$ — not because they are related but because $j$ is everywhere.
Raw co-occurrence recommends the same 20 videos to everyone. So we damp:

$$\text{score}(i,j) = \frac{c_{ij}}{c_i^{\alpha}\,c_j^{1-\alpha}}$$

$\alpha = 0.5$ recovers cosine exactly; higher punishes popular items harder. Making it a
config knob rather than a hidden constant is deliberate — it is the popularity/relevance dial.

**Rejected:** Jaccard (harsher but no tunable), PMI (surfaces niche pairs but is noisy at our
density — 12,026 surviving pairs after `min_cooccurrence=2`).

### 1.3 Implicit-feedback ALS — Hu, Koren & Volinsky (2008)

Learn $x_u, y_i \in \mathbb{R}^{96}$ with $r_{ui} \approx x_u^{\top}y_i$.

The core difficulty of implicit feedback: **there are no negatives.** An unwatched video means
"disliked" *or* "never saw it". Train only on observed entries and the model collapses to
"everything is loved". HKV's fix is two variables:

$$p_{ui} = \mathbb{1}[r_{ui} > 0], \qquad c_{ui} = 1 + \alpha r_{ui}$$

$$\min_{X,Y} \sum_{u,i} c_{ui}\left(p_{ui} - x_u^{\top}y_i\right)^2 + \lambda\left(\|X\|^2 + \|Y\|^2\right)$$

The sum runs over **all** pairs. A zero gets confidence 1 (weak evidence); a video watched 90%
through gets $1 + 30(0.9) = 28$. `r_ui` is **watch fraction, not a binary click** — a video
clicked and abandoned after three seconds is evidence *against* recommending it.

Fixing one side makes the other a convex least-squares problem with a closed form:

$$x_u = \left(Y^{\top}C^{u}Y + \lambda I\right)^{-1}Y^{\top}C^{u}p(u)$$

Naively $O(n_{\text{items}}f^2)$ per user — hopeless. The trick that makes ALS practical:

$$Y^{\top}C^{u}Y = \underbrace{Y^{\top}Y}_{\text{computed once}} + Y^{\top}(C^{u}-I)Y$$

and $C^u - I$ is zero everywhere the user did not interact, so the second term touches only
that user's ~30 items. Cost drops to $O(n_u f^2)$ — a ~200× saving here.

**Fold-in** reuses the identical closed form with frozen item factors, giving a brand-new user
a latent vector in one 96×96 solve. This is what makes the demo UI work without retraining.

**Rejected:** BPR / SGD (needs learning-rate tuning, no closed form); the `implicit` library
(the maths is the point, and it is a C-extension dependency for ~40 lines of NumPy).

### 1.4 Fusion — Reciprocal Rank Fusion

Content cosine ∈ [0,1]; damped co-visitation ∈ [0,~0.5] long-tailed; ALS dot products are
unbounded and popularity-correlated. Adding them is adding metres to kilograms.

$$\text{RRF}(d) = \sum_{s}\frac{w_s}{K + \text{rank}_s(d)}, \qquad K = 60$$

Scale-free, needs no per-source calibration, survives distribution drift on every rebuild.
$K$ flattens the head of the curve so rank 1 vs rank 2 is a nudge rather than a cliff, and no
over-confident source can steamroll the fusion.

**Cost:** RRF discards score *magnitude*. We recover it in Stage 2 — the ranker receives every
raw per-source score as a feature. Fusion decides what gets *considered*; the ranker decides
what *wins*.

**Rejected:** min-max / z-score normalisation. Simple but fragile — one outlier compresses
everything else, and each source's distribution shifts on every rebuild.

---

## Stage 2 · Learning to rank

### 2.1 The objective: watch time, not clicks

Optimising CTR produces clickbait; YouTube learned this publicly and switched to watch time in
2012. We implement the trick from the 2016 paper directly:

- positives get `sample_weight = watch_seconds / median_watch_seconds`
- negatives get `sample_weight = 1`

Then the learned odds are

$$\frac{P}{1-P} = \frac{\sum_i T_i}{N-k} \approx \mathbb{E}[T]$$

**The odds of a weighted click-classifier estimate expected watch time.** We rank by odds, not
probability. Ranking is invariant to a global scale on positive weights, so normalising by the
median (for numerical comfort) changes nothing.

`ranker.objective: click` reproduces the naive version for comparison.

### 2.2 Negatives: mixed, on purpose

- **In-feed negatives** — videos the user genuinely saw and skipped. Hard negatives: the old
  policy already judged them plausible.
- **Random catalog negatives** (2 per positive) — cover the regime that dominates at serving
  time. The engine scores all 6,000 items, most of which look nothing like a logged impression.

Trained only on in-feed negatives, the model is out-of-distribution when scoring the full
catalog. Adding random negatives raised AUC and weighted AUC together in the development run that motivated the change; the shipped model measures **AUC 0.6627 / weighted AUC 0.7553** ([`ranker_report.json`](../artifacts/ranker_report.json))
(Yi et al., *Mixed Negative Sampling*, 2019).

Using *only* random negatives — the common shortcut — would be worse than either: it trains the
model to separate "plausible" from "arbitrary", a far easier and far less useful task.

### 2.3 Position bias — Inverse Propensity Scoring

Our logs show 52% CTR at rank 0 and 0.1% at rank 7. **That is not a statement about the
videos.** Train naively and the model learns "things at the top get clicked", which is circular.

$$w_{\text{IPS}}(r) = \frac{1}{\max(\gamma^{\,r},\ \epsilon)}, \qquad \gamma = 0.82$$

Applied to observed clicks only, per the standard Joachims et al. estimator. A click at rank 6
is much stronger evidence than one at rank 0, and IPS says so. In production the propensities
must be *estimated* (via result randomisation); here we know them exactly.

Position itself is deliberately **not** a feature — it does not exist at serving time.

### 2.4 Causality and cross-fitting

Two properties are enforced structurally, both because violating them silently invalidated
earlier results:

**Causality** — the training set is built by replaying each user's timeline forward. A feed's
own clicks enter the history only *after* that feed has been featurised.

**Cross-fitting** — ALS and co-visitation feed the ranker, so in-sample rows made `als_score`
look near-oracular (AUC 0.856 in-sample vs 0.584 held out). The ranker leaned on it almost
exclusively and ended up *worse than its own best single feature*. Fixed with 4-fold
cross-fitting **by user**: for each fold, CF is refitted with that fold's users held out
entirely. See [`crossfit.py`](../src/recsys/rank/crossfit.py).

### 2.5 The model

`HistGradientBoostingClassifier` — ships with scikit-learn, handles unscaled heterogeneous
features natively, supports the sample weights the whole objective depends on. 19 features
across four groups: user↔item match, candidate-only, and one *relative* user feature.

Absolute user features (history length, distinct categories) were **removed after measurement**:
they are constant within a feed, so they cannot help rank one candidate above another, yet they
drift ~0.9σ between train and test as histories grow, so their learned splits do not transfer.

**Pointwise, not pairwise.** LambdaMART would likely win on NDCG, but the watch-time weighting
only works pointwise — it gives the score a *calibrated* interpretation (expected watch
seconds), and Stage 3 blends that score with freshness and diversity terms. You cannot sensibly
blend an uncalibrated pairwise margin with anything. Pairwise is the honest next step
([FUTURE_WORK.md](FUTURE_WORK.md)).

---

## Stage 3 · Policy

### 3.1 Maximal Marginal Relevance

$$\text{MMR} = \arg\max_{d \in R\setminus S}\Big[\lambda\cdot\text{rel}(d) - (1-\lambda)\max_{d_j\in S}\text{sim}(d,d_j)\Big]$$

$\lambda = 1$ → pure relevance (a repetitive page); $\lambda = 0$ → pure diversity. Default
0.72. The second term is the insight: redundancy is priced **contextually**, against what is
already on the page.

Similarity blends text cosine with a category-match indicator:

$$\text{sim}(a,b) = 0.75\cos(a,b) + 0.25\cdot\mathbb{1}[\text{cat}_a = \text{cat}_b]$$

Text cosine alone under-penalises stacking. Measured on a 3-Food/2-Gaming history, pure-cosine
MMR returned **7 of 8 slots as Food**; category-aware MMR returned **5 Food / 3 Gaming** and
raised intra-list diversity 0.737 → 0.855.

Relevance is **rank-normalised** before MMR, not min-max scaled: ranker output is an odds ratio
with a long right tail, and one candidate at odds 40 would compress everything else into a
sliver, making the diversity term meaningless.

### 3.2 Channel cap

A hard constraint, applied by masking rather than penalising — "at most 2 per creator" should
mean exactly that. Soft penalties leak. If every remaining candidate is capped out, the cap
relaxes rather than returning a short page.

### 3.3 Freshness

$$\text{multiplier} = 1 + w\cdot 0.5^{\,\text{age}/h}, \qquad w = 0.08,\ h = 21\text{ days}$$

Deliberately multiplicative and small: freshness should break ties between comparable videos,
never rescue an irrelevant one. It lives in policy rather than in the loss because a learned
model can only reproduce whatever recency bias was already in the log.

### 3.4 Exploration

A recommender trains on data it generated itself. Show only what you are confident about,
users click only that, and it becomes your next training set — the long tail goes permanently
dark. **You cannot learn that a video is good if you never show it.**

We reserve 2 of 24 slots for items sampled proportional to novelty $-\log_2 p(i)$, subject to a
relevance floor (top half of candidates only) so an exploration slot is never *actively bad*.
Picks are drawn sequentially so they respect the channel cap — selecting them as a batch
originally let a third video from one creator onto a max-2 page.

This is a structured ε-greedy. **Thompson sampling** — sampling from the posterior over each
item's value, so exploration scales with *uncertainty* — is the statistically superior answer
and is the top item in [FUTURE_WORK.md](FUTURE_WORK.md).
