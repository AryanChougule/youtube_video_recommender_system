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
from recsys.groundtruth import load_latent
from recsys.config import Paths, load_config
from recsys.counterfactual import replay_test_feeds
from recsys.metrics import dcg
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

    # NOT catalog["latent_clickbait"] -- the serving catalog excludes hidden
    # generative variables by construction. This used to fall back to zeros when
    # the column was absent, which turned the entire clickbait comparison into
    # "0.0000 for every strategy" without failing. load_latent raises instead.
    clickbait = load_latent("latent_clickbait", len(catalog))

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
        """Ranking metrics AND outcome metrics for one objective.

        Ranking metrics (top-1 / NDCG / MRR) say whether the clicked item was
        ordered correctly. Outcome metrics say what KIND of item reached the
        top. Both are needed: an objective can rank clicks well while promoting
        exactly the videos that leave people unhappy, which is the whole reason
        this comparison exists.
        """
        eng, sat, bait, comp = [], [], [], []
        ndcg, mrr, sat_ndcg = [], [], []
        for feed, X in prepared:
            scores = np.asarray(scorer(X), dtype=float)
            order = np.lexsort((RNG.random(len(scores)), -scores))
            top = int(order[0])
            item = int(feed.items[top])

            # --- ranking metrics, graded by watch fraction ---
            eng.append(float(feed.labels[top] == 1))
            gains = feed.gains * feed.labels
            ideal = dcg(sorted(gains, reverse=True))
            ndcg.append(dcg(gains[order]) / ideal if ideal > 0 else 0.0)
            mrr.append(1.0 / (int(np.flatnonzero(feed.labels[order] == 1)[0]) + 1))

            # --- same ranking, re-graded by SATISFACTION instead of watch time ---
            sat_gains = np.zeros(len(feed.items), dtype=float)
            for pos, it in enumerate(feed.items):
                if feed.labels[pos] != 1:
                    continue
                k_ = f"{sess_user.get(feed.session_id, '')}|{art.video_ids[int(it)]}"
                if k_ in outcomes.index:
                    sat_gains[pos] = float(outcomes.loc[k_]["satisfied"])
            ideal_s = dcg(sorted(sat_gains, reverse=True))
            sat_ndcg.append(dcg(sat_gains[order]) / ideal_s if ideal_s > 0 else 0.0)

            # --- outcome metrics for whatever landed at rank 1 ---
            bait.append(float(clickbait[item]))
            k = f"{sess_user.get(feed.session_id, '')}|{art.video_ids[item]}"
            row = outcomes.loc[k] if k in outcomes.index else None
            if row is not None and int(row["clicked"]) == 1:
                sat.append(float(row["satisfied"]))
                comp.append(float(row["watch_fraction"] >= 0.9))
            else:
                sat.append(0.0)
                comp.append(0.0)
        return {"top1": float(np.mean(eng)),
                "ndcg": float(np.mean(ndcg)),
                "mrr": float(np.mean(mrr)),
                "ndcg_satisfaction": float(np.mean(sat_ndcg)),
                "engagement@1": float(np.mean(eng)),
                "satisfaction@1": float(np.mean(sat)),
                "clickbait@1": float(np.mean(bait)),
                "completion@1": float(np.mean(comp))}

    results = {}
    results["B. Watch-time optimised"] = evaluate(
        "B", lambda X: art.ranker.score(X))
    for label in ("A. CTR-optimised", "C. Satisfaction-only", "D. Multi-objective"):
        w = OBJECTIVES[label]
        results[label] = evaluate(label, lambda X, w=w: multi.score(X, weights=w)
                                  if w else multi.score(X))

    order_ = ("A. CTR-optimised", "B. Watch-time optimised",
              "C. Satisfaction-only", "D. Multi-objective")

    print("\n[1] RANKING METRICS (Protocol A, graded by watch fraction)")
    print(f"    {'objective':<26}{'top-1':>10}{'NDCG':>10}{'MRR':>10}{'NDCG(satisf)':>14}")
    print("    " + "-" * 70)
    for label in order_:
        m = results[label]
        print(f"    {label:<26}{m['top1']:>10.4f}{m['ndcg']:>10.4f}"
              f"{m['mrr']:>10.4f}{m['ndcg_satisfaction']:>14.4f}")
    print("\n    NDCG(satisf) re-grades the SAME ranking by whether the click was")
    print("    satisfying rather than by watch fraction -- the objective a")
    print("    watch-time ranker cannot see.")

    print("\n[2] OUTCOME METRICS (what kind of item reached rank 1)")
    print(f"    {'objective':<26}{'engage@1':>11}{'satisf@1':>11}"
          f"{'clickbait@1':>13}{'complete@1':>12}")
    print("    " + "-" * 74)
    for label in order_:
        m = results[label]
        print(f"    {label:<26}{m['engagement@1']:>11.4f}{m['satisfaction@1']:>11.4f}"
              f"{m['clickbait@1']:>13.4f}{m['completion@1']:>12.4f}")

    print("\n[3] EACH HEAD ALONE AS A RANKER")
    print(f"    {'head':<26}{'top-1':>10}{'NDCG':>10}{'MRR':>10}{'AUC(fit)':>11}")
    print("    " + "-" * 67)
    per_head = {}
    for task in multi.models:
        m = evaluate(task, lambda X, t=task: multi.predict_tasks(X)[t])
        per_head[task] = m
        auc = multi.metrics.per_task_auc.get(task, float("nan"))
        print(f"    {task:<26}{m['top1']:>10.4f}{m['ndcg']:>10.4f}"
              f"{m['mrr']:>10.4f}{auc:>11.4f}")
    results["_per_head"] = per_head
    # Persist the fitted AUCs next to the ranking metrics. Without this the
    # AUC-vs-ranking comparison in docs/EVALUATION.md -- the point of this
    # whole script -- would cite numbers that live only in a gitignored
    # joblib, so a reviewer could read the claim but not check it.
    results["_per_head_auc"] = {t: round(float(a), 4)
                                for t, a in multi.metrics.per_task_auc.items()}

    print("\n    AUC measures separation on a fixed label; the ranking columns")
    print("    measure whether the right video reached the top of a real page.")
    print("    A head can score high AUC on a rare label and still rank poorly,")
    print("    which is exactly why both are reported.")

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
