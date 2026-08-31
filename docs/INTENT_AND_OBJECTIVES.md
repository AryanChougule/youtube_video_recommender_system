# Session intent and multi-objective ranking

Two hypotheses, both taken seriously, both tested properly. One is a clean
negative with a satisfying explanation; one is a partial win with a hard ceiling
that tells you exactly what to build next.

Reproduce with:

```bash
python scripts/10_evaluate_intent.py --alpha-sweep
python scripts/11_evaluate_objectives.py
```

---

# Part 1 · Session intent

## The hypothesis

A long-term profile answers *"what does this user generally like?"*. Often the
useful question is *"what do they want right now?"* — someone whose profile says
AI / cooking / gaming may today be working through one thing, learning RAG
before an interview. Filling their feed with assorted AI videos is a failure
even though every item is individually defensible.

Formally: split the history into a **long-term profile** and a **session
vector** (last 5 watches), and blend them by how *coherent* the session is.

$$\text{query} = \text{normalise}\big(\alpha \cdot \text{session} + (1-\alpha)\cdot\text{profile}\big)$$

$\alpha$ rises with session coherence (mean pairwise cosine of the recent items)
and with novelty (how unlike the profile the session is).

## Making it testable first

The simulator originally had **stable personas within a session**, so there was
no intent to detect and any measured "gain" would have been noise. So the
phenomenon was added to the generator before the detector was built: with
probability `intent_rate` (0.45) a session focuses on a single micro-topic, and
25% of those are **outside the user's persona entirely**. Ground truth is
written to `gt_session_intent.parquet` and never reaches any model.

## Result 1 — the detector works about as well as it possibly can

Coherence separates focused from browsing sessions at **AUC 0.6165**. That looks
weak until you compute the ceiling the same way this project computes every
ceiling — run the identical detector on the **true latent topic mixtures**:

| representation | AUC | p25 | p50 | p90 |
|---|---|---|---|---|
| text vectors (what we serve) | 0.6165 | 0.069 | 0.111 | 0.231 |
| ALS latent (behavioural) | 0.5766 | 0.082 | 0.131 | 0.215 |
| **ORACLE: true topic mixtures** | **0.6251** | 0.148 | 0.263 | 0.573 |

The servable detector reaches **98.6% of the oracle**. The bottleneck is not the
item representation — **five clicks simply do not contain much information about
intent**. A learned session encoder would be optimising against a ceiling that
is already nearly reached.

> This measurement also caught a calibration bug. The first thresholds
> (`LO=0.18, HI=0.55`) were plausible-looking numbers chosen by intuition — and
> they sit *above the 90th percentile* of the real distribution, so $\alpha$ was
> ≈0 for nearly every session and the detector did nothing at all. Calibrating
> to the observed p25/p90 fixed it.

## Result 2 — the mechanism works on exactly the cohort it should

Split by ground-truth intent (Protocol A, re-ranking logged impressions):

| cohort | n | profile-only | blended | lift |
|---|---|---|---|---|
| focused **+ off-persona** | 1,077 | 0.1495 | 0.1606 | **+7.5%** |
| focused | 3,675 | 0.1693 | 0.1733 | +2.4% |
| browsing | 4,325 | 0.1963 | 0.1896 | **−3.4%** |

Exactly the predicted pattern: it helps most where intent is real and different
from the profile, and it *hurts* when the user is just browsing.

## Result 3 — but it does not help overall, and here is why

Browsing sessions are the majority, and the detector cannot reliably tell them
apart, so the losses cancel the gains:

| protocol | best achievable | |
|---|---|---|
| A — re-rank logged impressions | **−0.1%** | best over all gates |
| B — 1 positive vs 100 negatives | **+0.1%** | best over all gates |
| α sweep 0.0 → 1.0 | best α = **0.0** | |

Every gating rule was tried: coherence thresholds from 0.15 to 0.30, novelty
thresholds, conjunctions of both, and a continuous blend. None nets positive.

## The explanation, which is the valuable part

The profile is a **recency-weighted** mean with a half-life of 8 positions. So
it is already dominated by recent watches:

$$\cos(\text{recency-decayed profile},\ \text{session vector}) = 0.803$$

**Exponential recency decay is already a soft session model.** The decisive test
is to remove it — on a uniform-mean profile with no decay, session blending
suddenly works:

| profile basis | α = 0 | α = 0.5 | lift |
|---|---|---|---|
| recency-decayed (our baseline) | 0.4522 | 0.4395 | **−2.8%** |
| uniform mean (no decay) | 0.4433 | 0.4547 | **+2.6%** |

Session-intent blending helps exactly when your profile **lacks** recency
weighting. Ours has it. The mechanism was already implemented under a different
name.

A half-life sweep confirms there is no tuning win hiding here either — the curve
is flat from 4 to 20 (HR@10 0.4369 → 0.4370, peaking at 0.4402 for half-life 6
versus 0.4395 at the shipped 8, a difference well inside noise).

> **The transferable lesson:** before adding a mechanism, check whether a
> simpler one already in the system covers it. A plausible product story is not
> evidence, and "we added session modelling" would have been an easy and
> completely unearned claim.

## What ships

- **Intent detection: ON**, used for **explanation**. The UI names the current
  focus ("street food, hawker stall, local eats"), which makes the feed legible
  in a way the raw profile cannot.
- **Intent blending: OFF** (`policy.intent_alpha_scale: 0.0`), exposed as a
  slider in the Recommendation Lab so an evaluator can turn it on and watch it
  fail to help. Demonstrating the negative result is more useful than hiding it.

---

# Part 2 · Multi-objective ranking

## The hypothesis

Optimising watch time beats optimising clicks, but watch time is still a
*proxy*. Predict several outcomes and combine them with explicit weights:

$$\text{value} = \sum_k w_k \cdot P_k(x)$$

## Making it testable first

Same discipline: if watch time and satisfaction agreed, multi-task ranking would
be complexity for its own sake. So the simulator gained a latent **clickbait**
factor per video — it wins the click, and sharply reduces satisfaction.

Measured on the generated log:

```
top-decile clickbait : watch 0.425, satisfied 42%
bottom-half clickbait: watch 0.438, satisfied 73%
corr(clickbait, watch_fraction) = +0.01
corr(clickbait, satisfied)      = -0.21
```

Note that watch time is nearly **blind** to clickbait. The +0.01 correlation is
not the "holds you longer" effect I first assumed — clickbait also pulls in
lower-affinity viewers who watch *less*, and the two effects cancel. That was
written into the docstring as a claim before it was checked, and the measurement
contradicted it.

## Result — six heads, all learning something real

| head | AUC | positive rate |
|---|---|---|
| click | 0.6627 | 14.14% |
| long_watch | 0.8124 | 6.63% |
| completion | 0.9154 | 0.36% |
| liked | 0.7436 | 1.50% |
| satisfied | 0.7376 | 10.41% |
| dismissed | 0.9154 | 0.44% |

Comparing objectives on identical feeds (Protocol A):

| objective | engage@1 | satisf@1 | clickbait@1 | complete@1 |
|---|---|---|---|---|
| A. CTR-optimised | 0.1972 | 0.1463 | 0.2059 | 0.0055 |
| B. Watch-time optimised | 0.1933 | 0.1413 | 0.2091 | 0.0035 |
| C. Satisfaction-only | 0.1972 | 0.1457 | **0.2023** | 0.0097 |
| D. Multi-objective (shipped) | 0.1967 | 0.1452 | 0.2048 | **0.0100** |

**What worked:** `completion@1` nearly doubles from CTR-optimised to
multi-objective (0.0055 → 0.0100, **+82%**). The objective the features *can*
see is served.

**What did not:** clickbait exposure barely moves (−0.5%), and satisfaction@1
does not improve.

## The ceiling, and why it is the most useful finding here

Can *any* model see clickbait from the features the ranker has?

| target | GBDT R² from all item features |
|---|---|
| `latent_quality` | **0.6355** |
| `latent_clickbait` | **−0.1122** |

R² below zero means worse than predicting the mean. **Quality is partially
visible through engagement rate; clickbait is invisible.** No ranker, however
sophisticated, can optimise for something absent from its inputs.

> **A multi-objective ranker can only trade off objectives it can observe.**
> Adding a satisfaction head without satisfaction-predictive features is
> theatre.

This is exactly why YouTube collects **user surveys** rather than inferring
satisfaction from engagement metadata. The finding points at the right next
step, and it is a *feature* problem, not a *model* problem: title-vs-content
mismatch signals, early-abandon rates, thumbnail-vs-content agreement.

## What ships

Multi-objective ranking is **on**, because it earns its place on two grounds
that do not depend on the clickbait result:

1. **A real measured win** on completion@1 (+82%).
2. **Controllability.** The heads are fixed at training time but the objective
   is chosen *per request*, so the Recommendation Lab can turn the system from
   engagement-maximising to satisfaction-maximising with no retraining. Switching
   preset changes 6 of the top 6 items.

Default weights (`ranker.objective_weights`) are an **engineering choice**, not
YouTube's — they have never published theirs.

---

## Honest summary

| claim | verdict |
|---|---|
| Session intent is a real phenomenon worth modelling | ✅ measured, +7.5% on the right cohort |
| Explicit session blending improves ranking | ❌ no — recency decay already does it |
| Session intent aids *explainability* | ✅ shipped for exactly this |
| Multi-objective ranking gives controllability | ✅ shipped |
| Multi-objective ranking improves completion | ✅ +82% |
| Multi-objective ranking reduces clickbait | ❌ no — clickbait is invisible to the features |

Two of six hypotheses failed. Both failures are more informative than the
successes, and both were only visible because the simulator provides ground
truth that real logs never would.
