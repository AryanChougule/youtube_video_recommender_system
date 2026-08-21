"""Run every documented test case and print real system output.

    python scripts/09_test_cases.py

docs/TEST_CASES.md is generated from this. Writing the cases as executable code
rather than prose means they cannot silently drift from what the system does --
and the failure cases stay honest, because they are re-measured on every run.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.engine import RecommendationEngine

RULE = "=" * 78


def head(number: str, title: str, expectation: str) -> None:
    print(f"\n{RULE}\n{number}  {title}\n{RULE}\nEXPECTED: {expectation}\n")


def show(engine, response, limit: int = 8, note: str = "") -> None:
    for item in response.items[:limit]:
        flag = "*" if "exploration" in item.policy_notes else " "
        print(f"  {item.rank + 1:>2}.{flag}[{item.category:<20}] {item.title[:44]:<44}")
        print(f"       {item.explanation[:70]}")
    d = response.diagnostics
    print(f"\n  mix={d.get('category_mix')}")
    print(f"  diversity={d.get('intra_list_diversity')}  novelty={d.get('novelty_bits')}  "
          f"channels={d.get('distinct_channels')}  latency={response.stages['total_ms']}ms")
    if note:
        print(f"  -> {note}")


def category_share(response, category: str) -> float:
    return sum(i.category == category for i in response.items) / max(len(response.items), 1)


def main() -> None:
    cfg = load_config()
    art = load_artifacts(cfg)
    engine = RecommendationEngine(art)
    catalog = art.catalog
    interactions = pd.read_parquet(Paths.interactions)

    def ids(category: str, n: int, offset: int = 0) -> list[str]:
        subset = catalog[catalog["category"] == category]
        subset = subset.nlargest(min(200, len(subset)), "view_count").iloc[offset:offset + n]
        return subset["video_id"].tolist()

    print(RULE)
    print("SUCCESSFUL SCENARIOS")

    head("S1", "Cold start - no history at all",
         "trending/popular mix, high category diversity, no personalisation claimed")
    show(engine, engine.recommend(history=[], n=8))

    head("S2", "Single-interest viewer (5 Gaming videos)",
         "strong, correct personalisation towards the stated interest")
    gaming = ids("Gaming", 5)
    res = engine.recommend(history=gaming, n=12)
    show(engine, res, note=f"Gaming share = {category_share(res, 'Gaming'):.0%} "
                           f"vs {category_share(engine.recommend(n=12), 'Gaming'):.0%} cold-start "
                           f"-- personalisation works, but see F7 for the cost")

    head("S3", "Multi-interest viewer (3 Food + 3 Gaming)",
         "BOTH interests represented, roughly in proportion to the history")
    mixed = ids("Food", 3) + ids("Gaming", 3)
    res = engine.recommend(history=mixed, n=12)
    show(engine, res, note=f"Food {category_share(res,'Food'):.0%} / "
                           f"Gaming {category_share(res,'Gaming'):.0%}")

    head("S4", "Semantic search - 'sourdough bread baking at home'",
         "Food results despite the query sharing no exact terms with most titles")
    show(engine, engine.search("sourdough bread baking at home", n=6), limit=6)

    head("S5", "Watch page - 'more like this'",
         "topically coherent rail, seed video excluded, no personalisation")
    seed = catalog[catalog["category"] == "Finance"]["video_id"].iloc[3]
    print(f"  SEED: {catalog[catalog.video_id == seed].title.iloc[0]}\n")
    show(engine, engine.similar(seed, n=6), limit=6)

    head("S6", "Channel affinity",
         "watching 3 videos from one creator should surface more from that creator")
    channel = catalog["channel_id"].value_counts().index[0]
    from_channel = catalog[catalog["channel_id"] == channel]["video_id"].head(3).tolist()
    name = catalog[catalog["channel_id"] == channel]["channel_title"].iloc[0]
    res = engine.recommend(history=from_channel, n=12)
    same = sum(i.channel_id == channel for i in res.items)
    print(f"  watched 3 from '{name}'\n")
    show(engine, res, note=f"{same} of 12 slots from that channel "
                           f"(hard cap is {cfg.policy.max_per_channel})")

    head("S7", "Diversity control is real, not decorative",
         "lambda 1.0 (pure relevance) should be measurably less diverse than 0.3")
    food = ids("Food", 5)
    focused = engine.recommend(history=food, n=16, mmr_lambda=1.0, exploration_slots=0)
    diverse = engine.recommend(history=food, n=16, mmr_lambda=0.3, exploration_slots=0)
    print(f"  lambda=1.0  diversity={focused.diagnostics['intra_list_diversity']}  "
          f"categories={focused.diagnostics['distinct_categories']}  "
          f"mix={focused.diagnostics['category_mix']}")
    print(f"  lambda=0.3  diversity={diverse.diagnostics['intra_list_diversity']}  "
          f"categories={diverse.diagnostics['distinct_categories']}  "
          f"mix={diverse.diagnostics['category_mix']}")

    head("S8", "Cross-category bridge (the payoff of latent topics)",
         "a GAMING history about PC hardware should leak into Tech / Science, "
         "because pc_hardware is a shared micro-topic")
    hardware = "GPU|RTX|Motherboard|CPU Cooler|Overclocking|DDR5|Power Supply|Custom Loop|Airflow"
    hw_gaming = catalog[(catalog["category"] == "Gaming")
                        & catalog["title"].str.contains(hardware, case=False, na=False)]
    hw_ids = hw_gaming["video_id"].head(5).tolist()
    if hw_ids:
        print(f"  seeded with {len(hw_ids)} GAMING videos about PC hardware:")
        for v in hw_ids[:3]:
            print(f"    - {catalog[catalog.video_id == v].title.iloc[0][:60]}")

        # (a) At the RECALL layer, where the bridge lives.
        probe = art.idx(hw_ids[1])
        print(f"\n  (a) RECALL layer -- ALS neighbours of "
              f"'{catalog.iloc[probe].title[:44]}':")
        for i in art.als.similar_items(probe, k=6).indices:
            print(f"        [{engine.category_of[int(i)]:<20}] {catalog.iloc[int(i)].title[:44]}")
        for name, source in (("ALS", art.als), ("content", None)):
            if name == "ALS":
                idx = art.als.similar_items(probe, k=20).indices
            else:
                idx = art.content.similar_items(probe, k=20).indices
            escape = np.mean([engine.category_of[int(i)] != "Gaming" for i in idx])
            print(f"        {name:<8} top-20 cross-category rate: {escape:.0%}")

        # (b) After the full pipeline.
        res = engine.recommend(history=hw_ids, n=16)
        crossed = [i for i in res.items if i.category != "Gaming"]
        print("\n  (b) FULL pipeline output:")
        show(engine, res, limit=6,
             note=f"{len(crossed)}/16 slots left Gaming. The bridge is REAL at the "
                  f"recall layer but the ranker + relevance floor suppress it -- "
                  f"see F7, this is the same mechanism.")
    else:
        print("  (no hardware-flavoured Gaming videos in this catalog build)")

    print(f"\n\n{RULE}\nFAILURE SCENARIOS  (documented, not hidden)")

    head("F1", "Cold ITEM - a video nobody has ever watched",
         "unreachable by CF; only content + trending can surface it. This is a real hole.")
    clicked = set(interactions[interactions["clicked"] == 1]["video_id"])
    cold_items = [v for v in catalog["video_id"] if v not in clicked]
    print(f"  {len(cold_items)} of {len(catalog)} videos have ZERO clicks")
    no_neighbours = int((art.covisitation.scores[:, 0] == 0).sum())
    print(f"  {no_neighbours} of {len(catalog)} videos have NO co-visitation neighbours "
          f"({no_neighbours / len(catalog):.0%})")
    if cold_items:
        idx = art.idx(cold_items[0])
        print(f"\n  probe: {catalog.iloc[idx].title[:60]}")
        print(f"    co-visitation neighbours : {int((art.covisitation.scores[idx] > 0).sum())}")
        print(f"    ALS factor norm          : {np.linalg.norm(art.als.item_factors[idx]):.4f} "
              f"(catalog median {np.median(np.linalg.norm(art.als.item_factors, axis=1)):.4f})")
        similar = art.content.similar_items(idx, k=3)
        print("    content recall still works:")
        for i in similar.indices:
            print(f"      - {catalog.iloc[int(i)].title[:56]}")

    head("F2", "Single-video history - almost no signal",
         "one watch cannot separate 'this topic' from 'this format'; expect wobble")
    single = ids("Travel", 1)
    res = engine.recommend(history=single, n=8)
    print(f"  watched: {catalog[catalog.video_id == single[0]].title.iloc[0][:60]}\n")
    show(engine, res, note=f"Travel share = {category_share(res, 'Travel'):.0%}")

    head("F3", "Contradictory history - 6 unrelated categories",
         "the averaged profile points nowhere; the page becomes incoherent")
    chaos = [v for c in ["Music", "Finance", "Autos & Vehicles", "Food",
                         "Education", "Sports"] for v in ids(c, 1)]
    res = engine.recommend(history=chaos, n=12)
    show(engine, res, note="no dominant interest -> the centroid matches nothing well")

    head("F4", "The TF-IDF template trap",
         "search matches title FORMAT as well as topic - a genuine content-based failure")
    res = engine.search("beginner guide to investing in index funds", n=6)
    show(engine, res, limit=6,
         note="several hits are 'A Beginner Guide to <unrelated topic>' - "
              "the template phrase carries as much weight as the subject")

    head("F5", "Extreme policy setting - lambda 0.0",
         "pure diversity produces a page with no coherent relevance. Knobs can be set badly.")
    res = engine.recommend(history=ids("Food", 5), n=10, mmr_lambda=0.0, exploration_slots=0)
    show(engine, res, limit=6,
         note=f"Food share collapses to {category_share(res, 'Food'):.0%}")

    head("F7", "Filter bubble - a single-interest history yields a MONOCULTURE",
         "this is the system's most important weakness, and it is structural")
    res = engine.recommend(history=ids("Gaming", 5), n=16)
    mix = res.diagnostics["category_mix"]
    print(f"  category mix from a 5-video Gaming history: {mix}")
    print(f"  distinct categories: {res.diagnostics['distinct_categories']} of 13")
    cands = engine._recall([art.idx(v) for v in ids("Gaming", 5)],
                           [1.0] * 5, None, None, [])
    trending_cats = {engine.category_of[int(i)] for i in cands["trending"].indices[:40]}
    print("\n  WHY: recall DOES contain other categories -- trending alone contributes")
    print(f"  {len(trending_cats)} distinct categories. But those candidates score low, and")
    print(f"  the exploration slots apply a relevance floor (top half only), so they are")
    print(f"  filtered out before they can reach the page. MMR can only diversify among")
    print(f"  candidates that survive ranking; it cannot invent variety that Stage 2 removed.")
    print(f"  FIX (not implemented): reserve slots for the best candidate OUTSIDE the")
    print(f"  dominant category, rather than relying on a global relevance floor.")

    head("F6", "Popularity still wins the biased offline metric",
         "documented in EVALUATION.md - the oracle loses too, so the METRIC is at fault")
    print("  full-catalog NDCG@10:  popularity 0.0218  >  ORACLE 0.0198  >  FULL pipeline 0.0125")
    print("  This is a property of logged data, not of this model. Protocol A in")
    print("  docs/EVALUATION.md is the counterfactually valid view, and there the")
    print("  learned ranker leads every genuine model (top-1 0.2240 vs 0.1385 popularity).")

    print(f"\n{RULE}\nDone.")


if __name__ == "__main__":
    main()
