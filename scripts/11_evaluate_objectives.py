"""Does multi-objective ranking beat optimising a single metric?

    python scripts/11_evaluate_objectives.py

The experiment the whole multi-task machinery exists to justify. Four ranking
objectives are compared on IDENTICAL feeds, using Protocol A (re-ranking logged
impressions), because the full-catalog metric is invalid on logged data --
see docs/EVALUATION.md.

    A. CTR-optimised        weight only P(click)
    B. Watch-time optimised the single-objective ranker (odds ~ E[watch time])
    C. Satisfaction-only    weight only P(satisfied)
    D. Multi-objective      the shipped blend

Four outcomes are measured for each, because "which is better" depends
entirely on what you are measuring -- which is the point:

    engagement@1     did the user click the item we put first
    satisfaction@1   was that click a SATISFIED one
    clickbait@1      mean latent clickbait of the item we put first (lower
                     is better; the model never sees this)
    completion@1     did they watch >= 90%

`clickbait@1` is the honest test. It is a hidden generative variable, so no
model can optimise it directly, and any difference between objectives is a
real consequence of what they were told to maximise.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json

import joblib
import numpy as np
import pandas as pd

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.counterfactual import replay_test_feeds
from recsys.split import temporal_split

RNG = np.random.default_rng(7)

OBJECTIVES = {
    "A. CTR-optimised":        {"click": 1.0},
    "C. Satisfaction-only":    {"satisfied": 1.0},
    "D. Multi-objective":      None,       # the configured blend
}


def main() -> None:
    cfg = load_config()
    art = load_artifacts(cfg, with_ranker=True)
    multi = joblib.load(Paths.multitask_ranker)
    interactions = pd.read_parquet(Paths.interactions)
    catalog = art.catalog
    split = temporal_split(interactions, test_size=cfg.ranker.test_size)

    clickbait = (catalog["latent_clickbait"].to_numpy(dtype=np.float64)
                 if "latent_clickbait" in catalog.columns else np.zeros(len(catalog)))

    print("=" * 78)
    print("DOES MULTI-OBJECTIVE RANKING BEAT A SINGLE METRIC?")
    print("=" * 78)

    feeds = replay_test_feeds(interactions, split, art.video_index,
                              min_history=3, max_feeds=6000, seed=cfg.project.seed)
    print(f"\n{len(feeds):,} test-period feeds (Protocol A: re-rank what was shown)")

    # per-impression outcome lookup, so we can score whatever ends up at rank 1
    key = interactions["user_id"].astype(str) + "|" + interactions["video_id"].astype(str)
    outcomes = (interactions.assign(_k=key)
                .drop_duplicates("_k").set_index("_k")[["clicked", "satisfied",
                                                        "watch_fraction"]])
    sess_user = (interactions[["session_id", "user_id"]].drop_duplicates("session_id")
                 .set_index("session_id")["user_id"])

    # precompute features once per feed
    prepared = []
    for feed in feeds:
        ctx = art.features.build_context(feed.history, feed.weights)
        prepared.append((feed, art.features.build(ctx, feed.items)))

    def evaluate(name, scorer):
        eng, sat, bait, comp = [], [], [], []
        for feed, X in prepared:
            scores = np.asarray(scorer(X), dtype=float)
            order = np.lexsort((RNG.random(len(scores)), -scores))
            top = int(order[0])
            item = int(feed.items[top])
            eng.append(float(feed.labels[top] == 1))
            bait.append(float(clickbait[item]))
            uid = sess_user.get(feed.session_id, "")
            row = outcomes.loc[f"{uid}|{art.video_ids[item]}"] \
                if f"{uid}|{art.video_ids[item]}" in outcomes.index else None
            if row is not None and int(row["clicked"]) == 1:
                sat.append(float(row["satisfied"]))
                comp.append(float(row["watch_fraction"] >= 0.9))
            else:
                sat.append(0.0)
                comp.append(0.0)
        return {"engagement@1": float(np.mean(eng)),
                "satisfaction@1": float(np.mean(sat)),
                "clickbait@1": float(np.mean(bait)),
                "completion@1": float(np.mean(comp))}

    results = {}
    print(f"\n{'objective':<26}{'engage@1':>11}{'satisf@1':>11}"
          f"{'clickbait@1':>13}{'complete@1':>12}")
    print("-" * 74)

    results["B. Watch-time optimised"] = evaluate(
        "B", lambda X: art.ranker.score(X))
    for label in ("A. CTR-optimised", "C. Satisfaction-only", "D. Multi-objective"):
        w = OBJECTIVES[label]
        results[label] = evaluate(label, lambda X, w=w: multi.score(X, weights=w)
                                  if w else multi.score(X))

    for label in ("A. CTR-optimised", "B. Watch-time optimised",
                  "C. Satisfaction-only", "D. Multi-objective"):
        m = results[label]
        print(f"{label:<26}{m['engagement@1']:>11.4f}{m['satisfaction@1']:>11.4f}"
              f"{m['clickbait@1']:>13.4f}{m['completion@1']:>12.4f}")

    a, d = results["A. CTR-optimised"], results["D. Multi-objective"]
    print("\nCTR-optimised vs multi-objective:")
    print(f"  clickbait exposure  {a['clickbait@1']:.4f} -> {d['clickbait@1']:.4f}  "
          f"({d['clickbait@1'] / max(a['clickbait@1'], 1e-9) - 1:+.1%})")
    print(f"  satisfaction@1      {a['satisfaction@1']:.4f} -> {d['satisfaction@1']:.4f}  "
          f"({d['satisfaction@1'] / max(a['satisfaction@1'], 1e-9) - 1:+.1%})")
    print("\nclickbait is a HIDDEN generative variable -- no model optimises it")
    print("directly, so any difference is a real consequence of the objective.")

    out = Paths.artifacts / "objective_evaluation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(Paths.root)}")


if __name__ == "__main__":
    main()
