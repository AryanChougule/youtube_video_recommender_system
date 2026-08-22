# Dataset

## The constraint that shapes everything

A recommender needs two things: a **catalog** and **interactions**. For YouTube, only one of
them is obtainable.

| | Available publicly? |
|---|---|
| Video metadata | ✅ YouTube Data API v3, Kaggle trending dumps |
| User watch histories | ❌ **Do not exist, and will not** |

Watch history is among the most re-identifying data a platform holds. Netflix released an
anonymised ratings dataset in 2006; researchers de-anonymised it within weeks by
cross-referencing public IMDb reviews, and the sequel competition was cancelled. No platform
has released watch-level data since.

That leaves three honest options:

| Option | Cost |
|---|---|
| **A.** Use MovieLens and call the movies "videos" | A movie recommender in a costume — no channels, no watch-time, no Shorts, no freshness dynamics |
| **B.** Content-only, no collaborative filtering | Deletes the most interesting half of the field: no serendipity, no "users who watched this…" |
| **C.** Real metadata + explicitly simulated interactions | Must defend the simulator, and cannot claim the metrics predict real-world performance |

**This project takes C**, and states the cost loudly rather than hiding it.

---

## Catalog sources

All three normalise to the same schema
([`src/recsys/data/schema.py`](../src/recsys/data/schema.py)), so downstream code never learns
where the data came from. `catalog.source: auto` tries them in order.

### 1. YouTube Data API v3 — real, fresh, recommended

```bash
YOUTUBE_API_KEY=your_key python scripts/build_all.py --source youtube_api
```

Real titles, channels, tags, view/like/comment counts, durations, thumbnails.
Quota is the constraint: `search.list` costs 100 units and returns ≤50 ids, `videos.list`
costs 1 unit for 50 items, and the free tier is 10,000 units/day. A 6,000-video catalog needs
~120 searches ≈ 12,000 units. Results are cached to `data/raw/youtube_api_cache.jsonl`, so a
re-run never re-spends quota on ids it already holds — the build can span days.

Search queries are generated from our own micro-topic taxonomy, keeping the real catalog
topically aligned with the synthetic one.

### 2. Kaggle "Trending YouTube Video Statistics" — real, static, no key

Drop any `*videos.csv` into `data/raw/`. Two quirks handled explicitly:

- It is a **daily trending snapshot**, so a video appears on many rows with growing view
  counts. We keep the row with the highest view count per `video_id`.
- It has **no duration** (the export omits `contentDetails`). Duration matters here — it drives
  watch-time and the duration-fit feature — so it is imputed from a per-category median rather
  than silently zero-filled. This is an assumption; it is listed in [ASSUMPTIONS.md](ASSUMPTIONS.md).
- It has no `channel_id`, so a stable surrogate is hashed from the channel name.

### 3. Synthetic generator — default, zero setup

Runs with no credentials, which is why it is the default: a fresh clone works immediately.
It is not a random-noise generator; it is a statistical model built to reproduce the properties
that make recommendation *hard*.

| Property | Why it must be there |
|---|---|
| Heavy-tailed views (log-normal, 5 orders of magnitude) | popularity baselines become genuinely competitive; popularity bias is a real problem to defeat |
| Latent topic mixtures over 40 micro-topics | gives CF non-obvious structure to discover |
| Overlapping category→topic maps | creates **cross-category bridges** only behaviour can reveal |
| Channel clustering with topic drift | channel affinity becomes a separable signal |
| Hidden per-video quality, partly visible via engagement rate | the ranker has something real to learn but cannot learn it perfectly |
| Sub-linear view accumulation with age | decouples "most viewed" from "trending now" |

Measured output (seed 42, 6,000 videos, 418 channels):

```
views      p50 69,256   p90 730,511   p99 5,086,542   max 73,396,968
           top 1% of videos hold 31.7% of all views
duration   median 14.9 min, 7.7% Shorts (≤75s)
like rate  median 2.8%
```

### The micro-topic design — the decision that makes this interesting

A naive simulator says *"user likes Gaming → show Gaming"*. That is useless: a category lookup
would solve it and CF would have nothing to discover.

Instead every video has a **mixture over 40 latent micro-topics**, and the 13 categories are
*overlapping* distributions over them:

```
Category "Gaming"  ─┬─► micro-topic  #0 (pc_hardware)  ◄─┬─ Category "Tech Reviews"
                    ├─► micro-topic  #1 (speedrunning)   │
                    └─► micro-topic  #2 (esports)        └─► micro-topic #9 (gadget_review)
```

15 of the 40 micro-topics appear in more than one category. A "Budget Gaming PC Build" genuinely
sits between Gaming and Tech. **Content-based filtering cannot see this** (different words),
**category rules cannot see it** (different labels), **collaborative filtering can** — and
[ARCHITECTURE.md](ARCHITECTURE.md#why-each-recall-source-exists) shows it does.

---

## The interaction simulator

[`src/recsys/data/simulator.py`](../src/recsys/data/simulator.py) — read it; it is the most
important design decision in the project.

### Behavioural model

```
session intent → feed (logging policy) → examine (cascade)
               → click → watch → like / satisfied / dismissed
```

**Users.** Each has a *sparse* persona (2–4 micro-topics — real people watch a few things, and
a dense persona would make everyone look average and destroy the CF signal), plus a
mainstream-ness ∈ [0,1] (how much raw popularity drives their clicks), a preferred video length
(log-normal, median ~11 min), and an activity level.

**Logging policy.** A deliberately weak incumbent recommender: part popularity
($P \propto \text{views}^{0.55}$), part persona-relevant, part "related to what you just
watched" (autoplay), plus 12% pure random exploration. **This is what creates presentation
bias** — the training data is a biased sample of a previous policy's choices, which is the
central pathology of all real recommender data.

**Cascade click model** (Craswell et al., 2008). The user scans top-down, clicks item $i$ with
probability $p_i$, otherwise continues to the next slot with probability $\gamma = 0.82$.
Position bias is therefore an *emergent* property of scanning, not a hand-applied multiplier.

$$\text{logit}(p_i) = -1.95 + 3.4\,(12a_i) + 0.55\,m_u\,(2z^{\text{views}}_i) + 0.45\,z^{\text{quality}}_i - 0.40\left|\log\tfrac{d_i}{d^{*}_u}\right|$$

where $a_i$ is latent topic affinity, $m_u$ mainstream-ness, $d^{*}_u$ the user's preferred
duration.

**Watch fraction.** Beta-distributed around an affinity- and quality-driven mean, penalised by
video length. This is what makes optimising *watch time* rather than clicks possible.

**Session intent and clickbait.** Two later additions, each made so a hypothesis could be
TESTED rather than asserted — session intent so an intent detector has something real to detect,
and a latent clickbait factor so satisfaction genuinely diverges from watch time. Both, with
their results (one negative, one partial), are in
[INTENT_AND_OBJECTIVES.md](INTENT_AND_OBJECTIVES.md).

**Feed order is randomised** before display. Without this, position is confounded with
candidate *source* (popularity picks are assembled first, so autoplay picks would always land
in the rarely-examined tail). This makes our logs *less* biased than a real production log —
noted as a limitation.

### Output (seed 42)

```
3,831 users · 1,142,312 impressions · 124,097 clicks
CTR 10.9% · 32.4 clicks/user · matrix density 0.54%
27,043 sessions, 45% of them with a focused intent
73.5% of clicks were satisfying (the multi-objective label)
```

Density 0.54% is sparser than MovieLens-100k (6.3%) — deliberately, because YouTube-scale data
*is* sparser.

### Why this is defensible, and where it is not

**Defensible:** the latent structure is known, so we can ask whether a model recovered genuine
preference structure or merely memorised popularity — a question real logs cannot answer. It
also let us build the **oracle control** that proved the standard offline protocol invalid
(see [EVALUATION.md](EVALUATION.md)).

**The risk is circularity** — a low-rank model will trivially fit a low-rank generator. Three
defences:

1. **The generator is harder than any model fitted to it.** Sigmoid link, cascade process,
   position bias, popularity bias, duration-dependent completion, per-user mainstream-ness,
   taste drift. None of that is representable by matrix factorisation, so if ALS still helps,
   the signal is real.
2. **Presentation bias is preserved** — we only ever observe what the logging policy chose.
3. **The same algorithms are validated on MovieLens-100k** — 100,000 ratings from 943 *real
   people*, collected by GroupLens in 1998. `python scripts/06_validate_on_movielens.py`
   runs the identical `ImplicitALS` and `CoVisitation` classes from
   [`recall/cf.py`](../src/recsys/recall/cf.py) against it.

   Leave-one-out by timestamp, ratings ≥4 as implicit positives, 897 evaluation users:

   | | full-catalog NDCG@10 | sampled (1 vs 99) HR@10 | sampled NDCG@10 |
   |---|---|---|---|
   | random | 0.0041 | 0.1049 | 0.0484 |
   | popularity | 0.0422 | 0.4240 | 0.2393 |
   | co-visitation (ours) | 0.0653 | 0.5385 | 0.3101 |
   | **ALS (ours)** | **0.0795** | **0.5375** | **0.3318** |

   Both protocols are reported because they are **not comparable to each other**. Published
   ML-100k results almost always use the sampled-negative protocol (He et al., WWW 2017), where
   strong models reach HR@10 ≈ 0.60–0.70; our from-scratch ALS at 0.5375 / NDCG 0.3318 sits in
   the expected band for plain implicit MF — below the best neural models, comfortably above
   popularity. Quoting the full-catalog number against those papers would be a category error,
   which is the same trap [EVALUATION.md](EVALUATION.md) is about.

   **The conclusion that matters:** on real human data the algorithms beat popularity by
   1.9× (full-catalog NDCG@10). The implementation is sound; only the YouTube *data* is simulated.

**Where it is NOT defensible:** these numbers do not predict real-world YouTube performance,
and nothing offline does. Real viewers have moods, social context, sessions on three devices,
and taste that shifts for reasons no simulator encodes.

## Reproducibility

Everything derives from `config.yaml` + `project.seed` — **plus the build's reference instant**,
which is easy to overlook. The catalog assigns publish dates relative to "now" and the simulator
places sessions in a 90-day window ending at "now", so a rebuild on a different day yields
different (statistically equivalent) data.

[`src/recsys/clock.py`](../src/recsys/clock.py) makes that instant explicit:

```yaml
project:
  reference_date: "2026-01-15"   # pin it -> byte-identical builds
  # reference_date: null         # default -> use today (right for a live demo)
```

With it pinned, two runs produce byte-identical catalogs and interaction logs — asserted by
`test_build_is_byte_identical_with_a_pinned_reference_date`. Left unpinned, the artifacts shipped
here correspond to their build date, and freshness features stay meaningful for the demo.
