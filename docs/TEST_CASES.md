# Test cases

Every case below is **executable**: `python scripts/09_test_cases.py` re-runs all of them and
prints live system output. Writing them as code rather than prose means they cannot silently
drift from what the system actually does — and the failure cases stay honest, because they are
re-measured on every run.

Output shown is from a build with seed 42 (6,000 videos, 3,803 simulated users).
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
full-catalog NDCG@10:   popularity 0.0178  >  ORACLE 0.0169  >  FULL pipeline 0.0120
```

❌ On the standard offline protocol, a trivial popularity list wins.

✅ **But the metric is at fault, not the model** — the oracle built from the simulator's own
generative parameters *also* loses. Users can only click what the old policy showed them.
Under the counterfactually-valid protocol the learned ranker leads every genuine model
(top-1 0.2090 vs 0.1338 for popularity). Full analysis in
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

---

## Automated coverage

40 tests, `python -m pytest tests/ -q`. The interesting ones are regressions for bugs that
actually occurred:

| Test | Guards against |
|---|---|
| `test_within_group_top1_breaks_ties_randomly` | a tie-breaking bug that made a useless feature score 80% |
| `test_channel_cap_is_enforced_including_exploration_slots` | exploration bypassing the hard channel cap |
| `test_watch_page_stays_on_topic` | the empty-history ranker bug in S5 |
| `test_explanations_do_not_claim_a_profile_that_does_not_exist` | explanations misdescribing their own evidence |
| `test_ground_truth_never_leaks_into_the_response` | `latent_quality` reaching the API payload |
| `test_cf_was_trained_under_the_declared_temporal_cutoff` | the leakage in EVALUATION.md Finding 1 |
| `test_all_artifacts_agree_on_catalog_size` | silent row-order drift between artifacts |
| `test_latency_is_within_budget` | p95 regression above 250 ms |
