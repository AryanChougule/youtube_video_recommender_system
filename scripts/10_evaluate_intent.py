"""Does modelling session intent actually improve recommendations?

    python scripts/10_evaluate_intent.py [--alpha-sweep]

This is the experiment behind the project's central product claim. The claim is
falsifiable, and this script is what would falsify it.

    H0: blending a session vector into the query changes nothing useful.
    H1: it improves ranking, and improves it MOST on sessions where the user
        has a focused intent that differs from their long-term profile.

Protocol A is used throughout (re-ranking logged impressions), because the
full-catalog metric is invalid on logged data -- see docs/EVALUATION.md. Only
items the user was actually shown are re-ordered, so every label is observed.

Three scorers are compared on identical feeds:

    profile only   the conventional recommender: recency-weighted mean of the
                   whole watch history
    session only   the opposite extreme: only the last N watches
    BLENDED        coherence-weighted mix of the two (recsys.intent)

Results are then split by the simulator's GROUND-TRUTH session intent, which
the models never see. That split is the whole point: an aggregate win could
come from anywhere, but a win concentrated exactly on focused, off-persona
sessions is evidence the mechanism works for the stated reason.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from recsys.artifacts import load_artifacts
from recsys.config import Paths, load_config
from recsys.counterfactual import replay_test_feeds
from recsys.intent import (
    COHERENCE_HI, COHERENCE_LO, DEFAULT_WINDOW, detect_intent, session_coherence,
)
from recsys.metrics import dcg
from recsys.split import temporal_split

RNG = np.random.default_rng(7)


def score_feeds(feeds, scorer):
    """Top-1 / NDCG / MRR over feeds, with random tie-breaking."""
    top1, ndcgs, mrr = [], [], []
    for feed in feeds:
        scores = np.asarray(scorer(feed), dtype=float)
        order = np.lexsort((RNG.random(len(scores)), -scores))
        top1.append(float(feed.labels[order[0]] == 1))
        gains = feed.gains * feed.labels
        ideal = dcg(sorted(gains, reverse=True))
        ndcgs.append(dcg(gains[order]) / ideal if ideal > 0 else 0.0)
        mrr.append(1.0 / (int(np.flatnonzero(feed.labels[order] == 1)[0]) + 1))
    return {"top1": float(np.mean(top1)), "ndcg": float(np.mean(ndcgs)),
            "mrr": float(np.mean(mrr)), "n": len(feeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-sweep", action="store_true")
    parser.add_argument("--feeds", type=int, default=8000)
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg, with_ranker=False)
    vectors = art.text_index.vectors
    interactions = pd.read_parquet(Paths.interactions)
    split = temporal_split(interactions, test_size=cfg.ranker.test_size)

    print("=" * 78)
    print("DOES SESSION INTENT IMPROVE RECOMMENDATIONS?")
    print("=" * 78)

    feeds = replay_test_feeds(interactions, split, art.video_index,
                              min_history=DEFAULT_WINDOW, max_feeds=args.feeds,
                              seed=cfg.project.seed)
    print(f"\n{len(feeds):,} test-period feeds, each with >= {DEFAULT_WINDOW} prior watches")

    # ---- precompute per-feed vectors once ------------------------------
    prepared = []
    for feed in feeds:
        hist = np.asarray(feed.history, dtype=int)
        w = np.asarray(feed.weights, dtype=np.float32)
        back = np.arange(len(hist) - 1, -1, -1, dtype=np.float32)
        combined = w * np.power(0.5, back / 8.0)
        profile = (combined[:, None] * vectors[hist]).sum(axis=0)
        norm = np.linalg.norm(profile)
        profile = (profile / norm).astype(np.float32) if norm > 1e-9 else profile

        intent = detect_intent(vectors, hist, profile, window=DEFAULT_WINDOW)
        prepared.append((feed, profile, intent))

    def make(kind, max_alpha=None):
        def scorer(feed):
            i = id(feed)
            f, profile, intent = lookup[i]
            if kind == "profile":
                q = profile
            elif kind == "session":
                q = intent.session_vector
            else:
                a = intent.alpha if max_alpha is None else (
                    intent.alpha / max(intent.alpha, 1e-9)
                    * min(intent.alpha * max_alpha / 0.80, max_alpha)
                    if intent.alpha > 0 else 0.0)
                v = a * intent.session_vector + (1 - a) * profile
                n = np.linalg.norm(v)
                q = (v / n).astype(np.float32) if n > 1e-9 else v
            return vectors[f.items] @ q
        return scorer

    lookup = {id(f): (f, p, i) for f, p, i in prepared}
    all_feeds = [f for f, _, _ in prepared]

    print("\n[1] Overall (Protocol A: re-ranking logged impressions)")
    print(f"    {'query vector':<24}{'top-1':>9}{'NDCG':>9}{'MRR':>9}")
    print("    " + "-" * 51)
    overall = {}
    for name, kind in (("profile only (baseline)", "profile"),
                       ("session only", "session"),
                       ("BLENDED (intent-aware)", "blend")):
        overall[name] = score_feeds(all_feeds, make(kind))
        m = overall[name]
        print(f"    {name:<24}{m['top1']:>9.4f}{m['ndcg']:>9.4f}{m['mrr']:>9.4f}")
    base = overall["profile only (baseline)"]["top1"]
    lift = overall["BLENDED (intent-aware)"]["top1"] / base - 1
    print(f"\n    blended vs profile-only: {lift:+.1%} on top-1")

    # ---- the split that tests the MECHANISM ----------------------------
    gt = pd.read_parquet(Paths.gt_session_intent).set_index("session_id")
    cohorts: dict[str, list] = {"focused + off-persona": [], "focused": [],
                                "browsing": []}
    user_topics = np.load(Paths.gt_user_topics)
    users = pd.read_parquet(Paths.users)
    u_index = {u: i for i, u in enumerate(users["user_id"])}
    # a feed carries no user_id, so recover it from the session table
    sess_user = (interactions[["session_id", "user_id"]].drop_duplicates("session_id")
                 .set_index("session_id")["user_id"])

    for feed, _p, _i in prepared:
        row = gt.loc[feed.session_id] if feed.session_id in gt.index else None
        if row is None:
            continue
        if not int(row["has_intent"]):
            cohorts["browsing"].append(feed)
            continue
        cohorts["focused"].append(feed)
        uid = sess_user.get(feed.session_id)
        topic = int(row["intent_topic"])
        if uid in u_index and topic >= 0:
            # off-persona = the focus topic is not one this user normally watches
            if user_topics[u_index[uid], topic] < 0.05:
                cohorts["focused + off-persona"].append(feed)

    print("\n[2] Split by GROUND-TRUTH session intent (models never see this)")
    print(f"    {'cohort':<26}{'n':>7}{'profile':>10}{'blended':>10}{'lift':>9}")
    print("    " + "-" * 62)
    cohort_results = {}
    for name, subset in cohorts.items():
        if len(subset) < 100:
            continue
        pr = score_feeds(subset, make("profile"))["top1"]
        bl = score_feeds(subset, make("blend"))["top1"]
        cohort_results[name] = {"n": len(subset), "profile": pr, "blended": bl,
                                "lift": bl / pr - 1 if pr else 0.0}
        print(f"    {name:<26}{len(subset):>7,}{pr:>10.4f}{bl:>10.4f}"
              f"{cohort_results[name]['lift']:>+9.1%}")

    # ---- detector quality ----------------------------------------------
    coh = np.array([i.coherence for _, _, i in prepared])
    has_intent = np.array([
        int(gt.loc[f.session_id, "has_intent"]) if f.session_id in gt.index else 0
        for f, _, _ in prepared])
    auc = roc_auc_score(has_intent, coh) if 0 < has_intent.mean() < 1 else float("nan")
    print(f"\n[3] Detector: coherence AUC vs ground-truth intent = {auc:.4f}")
    print(f"    calibration in use: LO={COHERENCE_LO} HI={COHERENCE_HI}")
    print(f"    observed coherence p25={np.quantile(coh, .25):.3f} "
          f"p50={np.quantile(coh, .5):.3f} p90={np.quantile(coh, .9):.3f}")
    alphas = np.array([i.alpha for _, _, i in prepared])
    print(f"    alpha actually applied: mean {alphas.mean():.3f}, "
          f"{(alphas > 0.15).mean():.0%} of sessions above the 'detected' threshold")

    if args.alpha_sweep:
        print("\n[4] MAX_ALPHA sweep")
        print(f"    {'max_alpha':<12}{'top-1':>9}{'NDCG':>9}")
        print("    " + "-" * 30)
        for ma in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            m = score_feeds(all_feeds, make("blend", max_alpha=ma))
            print(f"    {ma:<12.1f}{m['top1']:>9.4f}{m['ndcg']:>9.4f}")

    report = {"overall": overall, "by_cohort": cohort_results,
              "detector_auc": float(auc), "n_feeds": len(all_feeds)}
    out = Paths.artifacts / "intent_evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(Paths.root)}")


if __name__ == "__main__":
    main()
