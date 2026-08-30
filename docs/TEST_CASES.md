# Test cases

Every case below is **executable**: `python scripts/09_test_cases.py` re-runs all of them and
prints live system output. The intent and objective experiments behind **S9/S10/F8/F9** are
reproduced by `scripts/10_evaluate_intent.py` and `scripts/11_evaluate_objectives.py`. Writing them as code rather than prose means they cannot silently
drift from what the system actually does — and the failure cases stay honest, because they are
re-measured on every run.

Output shown is from a build with seed 42 (6,000 videos, 3,831 simulated users).
`*` marks a slot filled by the exploration policy.

---

# Successful scenarios

## S1 · Cold start — no history at all

**Expected:** trending/popular mix, high category diversity, no personalisation claimed.

```
 1. [Gaming              ] Stop Getting Tournament Wrong                 Trending in Gaming
 2. [Autos & Vehicles    ] Real Range Test, Explained Simply             Trending in Autos & Vehicles
 3. [Education           ] The Full Neuroscience Walkthrough             Trending in Education
 4. [Science & Technology] I Tried Inference Cost for 3 Days             Trending in Science & Technology
 5.*[Science & Technology] The Only Training Dataset Guide You Will Need Something different
 6.*[Tech Reviews        ] I Tried Long-term Review for 10 Days          Something different
 7. [Howto & Style       ] Joinery vs One Bag: Which One Actually Wins?  Trending in Howto & Style
 8. [Entertainment       ] A Beginner Guide to Colour Grade              Trending in Entertainment

mix = 7 categories across 8 slots · diversity 0.954 · 8 distinct channels · 5.3 ms
```

✅ No history, no crash, no fake personalisation. The explanations correctly say *"Trending in X"*
rather than pretending to know the user.

## S2 · Single-interest viewer (5 Gaming videos)

**Expected:** strong, correct personalisation towards the stated interest.

```
 1. [Gaming] I Tried World Record for 30 Days             Because you watched "Stop Getting Route Optimisation Wrong"
 2. [Gaming] Crafting Recipe vs Case Airflow              Viewers with a taste profile like yours watch this
 3. [Gaming] My Complete Route Optimisation Setup (2026)  Because you watched "Stop Getting Route Optimisation Wrong"
 5. [Gaming] Endgame Build Tier List (2024)               People who watched "How Sequence Break…" also watched this

Gaming share = 100%   (cold-start share was 8%)   ·   10 distinct channels
```

✅ Personalisation is unambiguous: 8% → 100% Gaming. Explanations name the specific video that
caused each recommendation.
⚠️ **100% is too much** — see **F7**, where this same result is analysed as a failure.

## S3 · Multi-interest viewer (3 Food + 3 Gaming)

**Expected:** both interests represented, roughly in proportion to the history.

```
 1. [Food  ] 10 Laminated Dough Mistakes You Are Making   More from "NoahAppetite"
 2. [Gaming] Building Leaderboard Completely From Scratch Because you watched "Stop Getting Tournament Wrong"
 3. [Food  ] The Truth About Brown Butter                 Because you watched "Hawker Stall: Ranked From Worst to Best"
 4. [Gaming] A Beginner Guide to GPU                      People who watched "I Tried Draft Phase for 12 Days…" also watched this
 7.*[Food  ] Local Eats, Explained Simply                 Something different

mix = {Food: 4, Gaming: 8}  →  Food 33% / Gaming 67%  ·  diversity 0.839
```

✅ Both interests survive — but Gaming takes 2:1, not the 50/50 of the history. Recency
weighting is the cause: the Gaming videos were watched more recently and the profile decays with
a half-life of 8 positions. Intended behaviour, though a 2:1 tilt from a 1:1 history is stronger
than ideal.

Note slot 4: **"A Beginner Guide to GPU" surfaced by co-visitation** — the `pc_hardware`
micro-topic bridging the two interests, which is exactly what the latent design is for.

## S4 · Semantic search — *"sourdough bread baking at home"*

```
 1. [Food] The Truth About Brown Butter                          Matches your search in Food
 2. [Food] Proofing: Everything You Actually Need to Know        Matches your search in Food
 3. [Food] How Open Crumb Completely Changed My Weeknight Dinner Matches your search in Food
 6. [Food] My Complete Pastry Setup (2026)                       Matches your search in Food
```

✅ None of the top hits contain the word "sourdough". LSA maps *laminated dough*, *proofing*
and *open crumb* into the same latent region — exactly the vocabulary-mismatch problem SVD
exists to solve.

## S5 · Watch page — "more like this"

**Seed:** *A Beginner Guide to Fundraising* (Finance)

```
 1. [Finance] My Complete Bootstrapped Setup (2025)   Watched by the same viewers as "A Beginner Guide to Fundraising"
 2. [Finance] The Full Earnings Call Walkthrough      Similar to "A Beginner Guide to Fundraising"
 3. [Finance] 20 Valuation Mistakes You Are Making    Similar to "A Beginner Guide to Fundraising"

mix = {Finance: 6}   ·   seed excluded   ·   ~12 ms
```

✅ Coherent rail, seed excluded, explanations reference the seed rather than a user profile.

> **This case caught a real bug.** Originally the watch page returned *Autos / Sports / Food*
> above Finance, explained as *"viewers with a taste profile like yours"* — but in watch-page
> mode there **is** no user profile. The ranker was receiving an empty history, so every match
> feature was zero and it fell back to generic item quality, discarding Stage 1's ordering. Fixed
> by treating the seed as a one-item history. Regression tests:
> `test_watch_page_stays_on_topic`, `test_explanations_do_not_claim_a_profile_that_does_not_exist`.

## S6 · Channel affinity + the hard cap

Watched 3 videos from *NoahAppetite*:

```
 2 of 12 slots from that channel   (hard cap = 2)   ·   8 distinct channels
```

✅ Channel affinity raises the creator's other videos into contention, and the cap stops it
becoming a single-creator page. Two policies pulling in opposite directions, both working.

## S7 · Diversity control is real, not decorative

```
λ = 1.0   diversity 0.729   categories  1   mix {Food: 16}
λ = 0.3   diversity 0.933   categories 10   mix {Food: 7, Gaming: 1, Education: 1, Sci&Tech: 1,
                                                 Sports: 1, Health: 1, Ent: 1, Music: 1,
                                                 Tech: 1, Finance: 1}
```

✅ One slider moves the page from a monoculture to 10 categories. The UI exposes this live.

## S8 · Cross-category bridge — the payoff of latent micro-topics

Seeded with 5 **Gaming** videos that are all about **PC hardware**. `pc_hardware` is a
micro-topic shared between Gaming, Science & Technology and Tech Reviews.

**(a) At the recall layer — the bridge is real:**

```
ALS neighbours of "RTX 5080 Tier List (2025)":
    [Gaming ] How Clutch Play Completely Changed My Hardcore Mode
    [Sports ] Patch Notes: Everything You Actually Need to Know      ← crossed
    [Gaming ] We Tested Thermal Paste So You Do Not Have To

ALS      top-20 cross-category rate: 60%
content  top-20 cross-category rate: 25%
```

**(b) After the full pipeline — it is suppressed:**

```
0 / 16 slots left Gaming.
```

⚠️ **Mixed result, reported honestly.** The latent bridge is genuinely discoverable — ALS
crosses categories 60% of the time for this seed, and the recall layer surfaces it. But the ranker plus
the exploration relevance floor filter those candidates out before they reach the page. The
mechanism is identical to **F7**. Fixing one fixes both.

## S9 · Session intent is detected and named

**Expected:** a topically tight session is recognised and labelled for the UI.

```
5 Food videos:
  detected : True
  label    : "weeknight dinner, open crumb, laminated dough"
  coherence: 0.471   novelty: 0.001   alpha: 0.80
  blend actually applied to ranking: 0.0   ← detection only, see F8

5 unrelated videos, for contrast:
  detected : False
  coherence: 0.031
```

✅ Coherent sessions get named; scattered ones don't fire. The label is what the
sidebar shows as **"Current focus"**, which makes the feed legible in a way the
raw profile vector cannot.

⚠️ Note `blend applied = 0.0`. Detection ships; **blending does not** — see
**F8** for the measurement that decided this.

## S10 · The objective is switchable at request time

**Expected:** same history, different objective → materially different feed, with
no retraining.

```
history: 5 Science & Technology videos

balanced (shipped)   Attention vs Tailwind CSS… | Why Voicing Is Much Harder… | How to Master Embeddings…
CTR-only             20 Large Language Model…   | Why Voicing Is Much Harder… | 20 CPU Cooler Mistakes…
satisfaction-only    REST API, Explained Simply | 20 CPU Cooler Mistakes…     | Stop Getting CPU Cooler Wrong

overlap between CTR-only and satisfaction-only top-10:  3/10
```

And the per-item breakdown that the **"Why this video?"** panel renders:

| objective | P | weight | contribution |
|---|---|---|---|
| satisfied | 26.2% | +0.40 | **+0.1048** |
| click | 49.4% | +0.10 | +0.0494 |
| long_watch | 1.7% | +0.25 | +0.0042 |
| dismissed | 1.8% | −0.20 | −0.0036 |
| | | **value score** | **+0.1568** |

✅ Six calibrated heads over one shared feature matrix; the objective is a
per-request weighted sum, so the Recommendation Lab changes what the system is
*for* without touching the models. 7 of 10 slots change between CTR-only and
satisfaction-only.

---

# Failure scenarios

These matter more than the successes. A system whose limits you cannot state is a system you do
not understand.

## F1 · Cold **items** — the structural hole in collaborative filtering

```
   17 / 6000 videos have ZERO clicks
 2327 / 6000 videos have NO co-visitation neighbours   (39% of the catalog)

probe: "Infinity: Everything You Actually Need to Know"
    co-visitation neighbours : 0
    ALS factor norm          : 0.0000   (catalog median 0.1529)
    content recall still works:
      - Probability Puzzle in 2025: Is It Still Worth It?
      - My Complete Infinity Setup (2025)
```

❌ **Two of the three recall sources are completely blind to this video.** Its ALS vector is
exactly zero — ALS has literally nothing to say about an item nobody watched.

✅ **Mitigation:** content-based recall works from metadata alone, and trending gives new
uploads a temporary boost. This is precisely why the hybrid includes a content leg — a pure-CF
system could never surface a new upload, which on a platform receiving 500 hours of video per
minute would be fatal.

**Severity:** inherent to CF, mitigated but not solved. See [FUTURE_WORK.md](FUTURE_WORK.md) for
the proper fix (content-conditioned item towers).

## F2 · Single-video history — one watch is not a taste

Watched: *Cheap Flights: Ranked From Worst to Best*

```
 62% of slots are Travel · nearly every explanation cites the SAME anchor video
```

❌ One watch cannot distinguish *"interested in budget travel"* from *"clicked a listicle"*.
The system commits hard to a single data point, and the page collapses to one topic and one
format.

**Severity:** moderate, and self-correcting after 2–3 watches (see S3). A production system
would blend in a demographic or geographic prior for this regime.

## F3 · Contradictory history — 6 unrelated categories

```
mix = {Education: 4, Autos: 3, Sports: 2, Food: 1, Finance: 1, Music: 1}
diversity 0.922
```

❌ With six unrelated interests the averaged profile vector points nowhere in particular. Each
recommendation is defensible in isolation, but the page has no thesis — it reads like six
different users' feeds interleaved.

This is the **profile-averaging weakness** documented in
[`recall/content.py`](../src/recsys/recall/content.py): a user who likes cooking *and* Formula 1
gets a centroid that matches neither. We partially mitigate it with a recency anchor and a
max-over-history feature; the real fix is clustering the history into multiple profiles and
retrieving for each.

**Severity:** moderate. Affects genuinely eclectic viewers — a real population.

## F4 · The TF-IDF template trap

Search: *"beginner guide to investing in index funds"*

```
 1. [Gaming              ] A Beginner Guide to GPU                  ← wrong topic
 2. [Science & Technology] A Beginner Guide to React                ← wrong topic
 3. [Education           ] A Beginner Guide to Probability Puzzle   ← wrong topic
 6. [Finance             ] Dividend Yield on a Budget: 15 Ideas     ← right topic, rank 6
```

❌ **A genuine, unmitigated content-based failure.** The phrase *"A Beginner Guide to"* is
common enough to dominate the TF-IDF match, so the index retrieves videos with the same title
**format** rather than the same **subject**. The one on-topic result is ranked 6th.

This is the same effect measured in the text-field ablation
([METHODOLOGY.md](METHODOLOGY.md#11-content-based--tf-idf--truncated-svd--cosine)), where
title-only indexing scored 0.6518 topical precision against 0.9040 for title+tags.

**Fixes not implemented:** (a) neural sentence embeddings, which handle this far better —
already supported via `features.text_backend`; (b) mining and down-weighting template n-grams;
(c) a learned query-understanding layer.

**Severity:** high for search, low for the feed (which relies mostly on CF).

## F5 · Extreme policy setting — λ = 0.0

```
mix = 10 categories across 10 slots · Food share collapses to 10% (history was 100% Food)
```

❌ Pure diversity produces a page with no coherent relevance. **The knobs can be set badly**,
and the UI lets an evaluator do exactly that on purpose — a system whose failure modes are
reachable is more honest than one that hides them behind clamped ranges.

## F6 · Popularity beats the hybrid on the naive offline metric

```
full-catalog NDCG@10:   Stage-1 recall 0.0125  >  popularity 0.0121  >  FULL pipeline 0.0107
Protocol A top-1:       FULL pipeline  0.1930  >  content 0.1840  >  popularity 0.1323
```

❌ On the standard offline protocol, adding the learned ranker makes things *worse*, and a
trivial popularity list beats content-based retrieval.

✅ **But the metric is at fault, not the model.** The same ranker that loses on full-catalog
NDCG wins decisively on both counterfactually-valid protocols, and the popularity scorer that
looks strong there lands *below random* per page (0.1323 vs 0.1415). Full analysis in
[EVALUATION.md](EVALUATION.md#finding-2--the-standard-offline-protocol-does-not-measure-the-recommender).

## F7 · Filter bubble — a single-interest history yields a monoculture

**The system's most important weakness.**

```
category mix from a 5-video Gaming history: {Gaming: 16}
distinct categories: 1 of 13
```

❌ A viewer who watches five Gaming videos sees a page that is 100% Gaming, for as long as they
keep watching Gaming. That is the feedback loop the whole exploration mechanism was supposed to
break, and here it does not.

**Diagnosed mechanism** (not a guess — measured in the script):

1. Recall **does** contain other categories: trending alone contributes 13 distinct categories
   to the candidate set.
2. Those candidates score low under a ranker whose strongest features are `content_sim_profile`
   and `category_affinity`.
3. The exploration slots apply a **relevance floor** (top half of candidates only) — which is
   exactly what filters the cross-category candidates out.
4. MMR can only diversify among candidates that survive ranking. **It cannot invent variety that
   Stage 2 removed.**

**Fix (designed, not implemented):** reserve slots for the highest-scoring candidate *outside
the dominant category*, rather than relying on a global relevance floor. The floor was added to
stop exploration showing junk; a per-category floor achieves both. This is the top item in
[FUTURE_WORK.md](FUTURE_WORK.md).

**Severity:** high. It is the difference between a recommender and an echo chamber, and it is
the single thing I would fix first with more time.

## F8 · Session-intent blending does not improve ranking

**The product story was right; the mechanism was already implemented.**

```
Protocol A (re-rank logged impressions)   best gate:  -0.1%
Protocol B (1 positive vs 100 negatives)  best gate:  +0.1%
alpha sweep 0.0 → 1.0                     best alpha:  0.0
```

❌ Every gating rule (coherence thresholds 0.15–0.30, novelty thresholds,
conjunctions, continuous blends) nets **negative or zero**.

It *does* behave exactly as predicted per cohort — **+7.5%** on focused,
off-persona sessions and **−3.4%** on browsing sessions. Browsing sessions are
the majority and the detector (AUC 0.6165) cannot reliably separate them, so
the losses cancel the gains.

**Diagnosis:** the profile is already recency-weighted with a half-life of 8
positions, so `cos(profile, session vector) = 0.803`. Remove the decay and
blending suddenly works (**+2.6%** on a uniform-mean profile). **Exponential
recency decay was already a soft session model.**

Turning the slider from 0.0 to 1.0 visibly moves the feed — it just doesn't move
it in a *better* direction. Full analysis in
[INTENT_AND_OBJECTIVES.md](INTENT_AND_OBJECTIVES.md).

**Severity:** none to the product (shipped off). High value as a lesson: before
adding a mechanism, check whether a simpler one already covers it.

## F9 · Multi-objective ranking cannot reduce clickbait exposure

Every head fits well:

```
click 0.6627 · long_watch 0.8124 · completion 0.9154
liked 0.7436 · satisfied 0.7376 · dismissed 0.9154        (AUC)
```

Yet clickbait exposure barely moves:

```
CTR-optimised    clickbait@1 = 0.2057
multi-objective  clickbait@1 = 0.2044   (-0.6%)
```

❌ **Why — can any model see clickbait from the served features?**

| target | GBDT R² from all item features |
|---|---|
| `latent_quality` | **+0.6355** (visible) |
| `latent_clickbait` | **−0.1122** (invisible) |

R² below zero is worse than predicting the mean. **No ranker can optimise an
objective absent from its inputs.** This is exactly why YouTube runs user
*surveys* rather than inferring satisfaction from engagement metadata — the next
step here is a **feature** problem, not a model problem.

✅ **What multi-objective ranking does buy, measured:** completion@1 rises
0.0055 → 0.0100 (**+82%**), plus per-request controllability (**S10**).

**Severity:** moderate — a genuine ceiling, correctly diagnosed, with a clear
next action (title-vs-content mismatch signals, early-abandon rates).

---

## F10 · Out-of-vocabulary search retrieves nothing

**Input:** search `"machine learning"` against the synthetic catalog.

**Observed (before the fix):** a full page of ranked-looking results, top hit
*"The Truth About VO2 Max"* — a fitness video, for a machine-learning query.

❌ **Why.** Not noise, and not a bad ranking. The TF-IDF vocabulary is built
from the catalog, and this catalog has no machine-learning content, so:

```
query      'machine learning'
tokens     ['machine', 'learning', 'machine learning']
in-vocab   []                                    <- none of them
||vector|| 0.000000                              <- exactly zero
```

Every cosine similarity is then `0.0`, the scores are all tied, and `argpartition`
returns whichever items the tie-break happened to leave first. The output has the
*shape* of a ranked list and contains no information whatsoever.

Compare an in-vocabulary query:

```
query      'brown butter'
in-vocab   ['brown', 'butter', 'brown butter']
||vector|| 0.891757                              -> correct Food results
```

This is a **known, structural property of lexical retrieval**, not a bug in the
ranker. TF-IDF can only match terms it has seen. Dense embeddings from a
sentence transformer would degrade gracefully instead of collapsing to zero —
that trade-off is [D3](DESIGN_DECISIONS.md), and the upgrade path is in
[FUTURE_WORK.md](FUTURE_WORK.md).

✅ **Fixed behaviour.** The zero-vector case is now detected rather than ranked:

| | before | after |
|---|---|---|
| `ContentRecall.search` | returns 50 tied items | returns **0 items** |
| feed | unrelated videos, presented as results | falls back to trending |
| heading | `Results for "machine learning"` | `No matches for "machine learning" — showing trending instead` |
| API | indistinguishable from success | `diagnostics.query_matched: false` |
| explanation | *"Matches your search in Howto & Style"* | *"Trending in Education"* |

**Is it expected?** The retrieval failure, yes — a lexical index cannot match
absent vocabulary. Presenting it as a successful search was not; an unlabelled
fallback looks identical to a working search that returned poor results, and
that is the more damaging of the two, because it makes the ranker look broken
when the retriever simply found nothing.

**Partial matches still work.** `"quantum blockchain nft"` matches only
`quantum`, keeps `query_matched: true`, and ranks on that one term — correct,
since the query genuinely did match something.

**How it could be improved:** (1) a dense retriever as a fallback leg so
semantic matches survive vocabulary gaps; (2) query expansion over the tag
graph; (3) spelling correction, which would catch the most common real cause of
a zero-vector query. Regression test:
`test_out_of_vocabulary_search_reports_no_match`.

**Severity:** low after the fix — the system now reports the limit instead of
hiding it.

---

## Automated coverage

61 tests, `python -m pytest tests/ -q`. The interesting ones are regressions for bugs that
actually occurred:

| Test | Guards against |
|---|---|
| `test_within_group_top1_breaks_ties_randomly` | a tie-breaking bug that made a useless feature score 80% |
| `test_channel_cap_is_enforced_including_exploration_slots` | exploration bypassing the hard channel cap |
| `test_watch_page_stays_on_topic` | the empty-history ranker bug in S5 |
| `test_explanations_do_not_claim_a_profile_that_does_not_exist` | explanations misdescribing their own evidence |
| `test_ground_truth_never_leaks_into_the_response` | `latent_quality` reaching the API payload |
| `test_cf_was_trained_under_the_declared_temporal_cutoff` | the leakage in EVALUATION.md Finding 1 |
| `test_intent_blend_falls_back_to_the_profile_when_incoherent` | a scattered session hijacking the query |
| `test_intent_never_fully_overrides_the_profile` | five clicks outweighing a whole history |
| `test_clickbait_is_generated_and_depresses_the_like_rate` | the wedge that makes multi-objective meaningful |
| `test_all_artifacts_agree_on_catalog_size` | silent row-order drift between artifacts |
| `test_latency_is_within_budget` | p95 regression above 250 ms |
| `test_out_of_vocabulary_search_reports_no_match` | F10 — arbitrary tie-broken results shown as search hits |
| `test_serving_runs_without_heavy_dependencies` | pandas/sklearn/SciPy creeping back into the deployed path |
| `test_numpy_export_matches_sklearn` | the exported models drifting from the evaluated ones |
| `test_vercel_mount_prefix_is_stripped` | a rewrite/strip mismatch 404-ing every route in production |
| `test_missing_ground_truth_raises_instead_of_defaulting` | hidden variables silently zeroing, reporting clickbait@1 = 0.0000 |
| `test_recall_reports_whether_a_query_matched` | `_recall`'s contract, unpacked directly by three scripts |
