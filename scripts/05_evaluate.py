"""Stage 5 of the build: offline evaluation + pipeline ablation.

    python scripts/05_evaluate.py [--users N] [--quick]

Runs every baseline and every stage of the hybrid against the same held-out
data and writes artifacts/evaluation.json.

The ablation is the point. One NDCG number proves nothing; showing what each
stage ADDS is what justifies the architecture -- and what would expose a stage
that earns nothing and should be deleted.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np
import pandas as pd

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.engine import RecommendationEngine
from recsys.evaluate import (
    EvalUser, build_eval_users, evaluate_strategy, make_baselines,
)
from recsys.counterfactual import (
    evaluate_reranking, evaluate_sampled, make_scorers, replay_test_feeds,
)
from recsys.split import temporal_split

HEADLINE = ["ndcg@10", "precision@10", "recall@20", "map@10", "mrr",
            "hit_rate@10", "coverage", "novelty_bits", "intra_list_diversity"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="300 users, fewer strategies")
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg, with_ranker=True, verbose=True)
    engine = RecommendationEngine(art)
    interactions = pd.read_parquet(Paths.interactions)

    started = time.time()
    print("=" * 78)
    print("STAGE 5/5  EVALUATION")
    print("=" * 78)

    split = temporal_split(interactions, test_size=cfg.ranker.test_size)
    max_users = args.users or (300 if args.quick else cfg.evaluation.max_eval_users)
    users = build_eval_users(
        interactions, split, art.video_index,
        min_history=3, min_holdout=1, max_users=max_users, seed=cfg.project.seed,
    )
    print(f"\nprotocol : global temporal split at {split.cutoff:%Y-%m-%d %H:%M}")
    print(f"users    : {len(users):,} with >=3 pre-cutoff clicks and >=1 held-out click")
    print(f"holdout  : median {np.median([len(u.relevant) for u in users]):.0f} items/user")
    print(f"history  : median {np.median([len(u.history) for u in users]):.0f} items/user")

    stats = art.item_stats
    pop_baseline = list(np.argsort(-stats["log_views"].to_numpy())[:50])

    k_values = cfg.evaluation.k_values
    results = []

    def run(name, fn, notes=""):
        res = evaluate_strategy(name, fn, users, art, k_values, pop_baseline, notes)
        results.append(res)
        print(f"  {name:<28} ndcg@10={res.metrics['ndcg@10']:.4f}  "
              f"recall@20={res.metrics['recall@20']:.4f}  "
              f"cov={res.metrics['coverage']:.3f}  "
              f"p50={res.latency_ms['p50']:.1f}ms")
        return res

    print("\n[1] Baselines")
    for name, fn in make_baselines(art, seed=cfg.project.seed).items():
        if args.quick and name in ("trending",):
            continue
        run(name, fn, notes="baseline")

    print("\n[2] Hybrid pipeline ablation")

    def hybrid(**kwargs):
        def strategy(user: EvalUser, k: int):
            response = engine.recommend(
                history=[art.video_ids[i] for i in user.history],
                watch_weights=user.weights, n=k, explain=False, **kwargs,
            )
            return [art.idx(item.video_id) for item in response.items]
        return strategy

    # Stage 1 only: fused recall, no learned ranker, no policy.
    def recall_only(user: EvalUser, k: int):
        from recsys.recall.blend import reciprocal_rank_fusion
        results_map, _ = engine._recall(user.history, user.weights, None, None, user.history)
        fused = reciprocal_rank_fusion(
            results_map, weights=cfg.recall.weights, rrf_k=cfg.recall.rrf_k,
            max_candidates=cfg.recall.max_candidates)
        return [int(i) for i in fused.indices[:k]]

    run("hybrid_recall_only", recall_only, "Stage 1 only (RRF fusion)")
    run("hybrid_+ranker", hybrid(mmr_lambda=1.0, exploration_slots=0, max_per_channel=0),
        "Stage 1 + 2, policy disabled")
    run("hybrid_+ranker+diversity", hybrid(exploration_slots=0),
        "Stage 1 + 2 + MMR/channel cap")
    run("FULL (default config)", hybrid(), "Stage 1 + 2 + 3, as deployed")

    if not args.quick:
        print("\n[3] Policy sensitivity (MMR lambda)")
        for lam in (0.5, 0.72, 0.9, 1.0):
            run(f"mmr_lambda={lam}", hybrid(mmr_lambda=lam, exploration_slots=0),
                "policy sweep")

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("SUMMARY  (higher is better except gini)")
    print("=" * 78)
    header = f"{'strategy':<28}" + "".join(f"{m.replace('@','@'):>13}" for m in HEADLINE[:5])
    print(header)
    print("-" * len(header))
    for res in results:
        row = f"{res.name:<28}"
        for metric in HEADLINE[:5]:
            row += f"{res.metrics.get(metric, 0):>13.4f}"
        print(row)

    print(f"\n{'strategy':<28}{'coverage':>11}{'novelty':>10}{'diversity':>11}"
          f"{'gini':>8}{'p50 ms':>9}")
    print("-" * 77)
    for res in results:
        print(f"{res.name:<28}{res.metrics.get('coverage', 0):>11.4f}"
              f"{res.metrics.get('novelty_bits', 0):>10.2f}"
              f"{res.metrics.get('intra_list_diversity', 0):>11.4f}"
              f"{res.metrics.get('gini_exposure', 0):>8.3f}"
              f"{res.latency_ms['p50']:>9.1f}")

    # ---------------- counterfactually-valid protocols ----------------
    print("\n[4] Protocol A: re-ranking LOGGED impressions (counterfactually valid)")
    print("    Only re-orders items the user was actually shown, so every label")
    print("    is observed -- no counterfactual guessing. '[ref] shown position'")
    print("    is NOT a competing model: position CAUSES clicks under a cascade")
    print("    click model, so it measures the size of position bias, not quality.")
    feeds = replay_test_feeds(interactions, split, art.video_index,
                              max_feeds=6000, seed=cfg.project.seed)
    scorers = make_scorers(art)
    rerank = evaluate_reranking(scorers, feeds, seed=cfg.project.seed)
    print(f"\n    {'scorer':<26}{'top-1':>9}{'NDCG':>9}{'MRR':>9}   ({len(feeds):,} feeds of 8)")
    print("    " + "-" * 53)
    for name, m in sorted(rerank.items(), key=lambda kv: -kv[1]["top1"]):
        print(f"    {name:<26}{m['top1']:>9.4f}{m['ndcg']:>9.4f}{m['mrr']:>9.4f}")

    print("\n[5] Protocol B: 1 positive vs 100 sampled negatives")
    sampled = evaluate_sampled(scorers, users, art, n_negatives=100, k=10,
                               seed=cfg.project.seed)
    print(f"\n    {'scorer':<26}{'HR@10':>9}{'NDCG@10':>10}{'mean rank':>11}")
    print("    " + "-" * 56)
    for name, m in sorted(sampled.items(), key=lambda kv: -kv[1]["hit_rate@10"]):
        print(f"    {name:<26}{m['hit_rate@10']:>9.4f}{m['ndcg@10']:>10.4f}"
              f"{m['mean_rank']:>11.2f}")

    report = {
        "reranking_logged_impressions": rerank,
        "sampled_candidates": sampled,
        "protocol": {
            "split": split.to_dict(),
            "n_eval_users": len(users),
            "k_values": k_values,
            "graded_relevance": "watch_fraction",
            "catalog_size": art.n_items,
            "data_source": art.data_meta.get("source"),
            "caveat": (
                "Full-catalog metrics measure the LOGGING POLICY as much as the "
                "recommender. Run scripts/13_oracle_control.py for the control: "
                "an oracle built from the true generative parameters scores "
                "0.0165 against popularity's 0.0121, so the metric is not pure "
                "noise -- but the ranking that wins on it is not the ranking "
                "that wins on observed labels, and adding the learned ranker "
                "moves the two families in opposite directions. Sections 4 and "
                "5 are the counterfactually-valid views. "
                "See src/recsys/counterfactual.py."
            ),
        },
        "results": [
            {"name": r.name, "notes": r.notes, "metrics": r.metrics,
             "latency_ms": r.latency_ms}
            for r in results
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Paths.eval_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {Paths.eval_report.relative_to(Paths.root)}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
