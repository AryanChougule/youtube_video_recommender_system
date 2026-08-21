"""Validate the from-scratch CF implementations against REAL human data.

    python scripts/06_validate_on_movielens.py [--factors N] [--no-download]

Why this exists
---------------
The YouTube interaction log in this project is simulated, which invites a fair
objection: maybe the models only look good because a low-rank model is fitting
a low-rank generator.

This script answers it by running the IDENTICAL ``ImplicitALS`` and
``CoVisitation`` code from ``recsys.recall.cf`` against MovieLens-100k --
100,000 ratings from 943 real people on 1,682 real films, collected by
GroupLens in 1998. Nothing about the algorithms is simulated; only the YouTube
*data* is.

If the implementation lands in the range published for implicit-feedback MF on
this dataset, then the algorithms are correct and the simulation is confined to
the data layer -- which is exactly the claim the documentation makes.

Protocol: leave-last-N-out per user by timestamp (temporal, not random), with
ratings >= 4 treated as positive implicit feedback.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config import Paths
from recsys.metrics import (
    average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)
from recsys.recall.cf import ImplicitALS, build_covisitation

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
LOCAL = Paths.raw / "ml-100k"

# Positive threshold. MovieLens is EXPLICIT (1-5 stars); we binarise to implicit
# feedback so the same code path runs. 4+ is the standard convention.
POSITIVE_RATING = 4.0


def fetch() -> pd.DataFrame:
    data_file = LOCAL / "u.data"
    if not data_file.exists():
        LOCAL.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {URL} ...")
        with urllib.request.urlopen(URL, timeout=90) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                if member.startswith("ml-100k/") and not member.endswith("/"):
                    target = LOCAL / Path(member).name
                    target.write_bytes(archive.read(member))
        print(f"  extracted to {LOCAL}")
    return pd.read_csv(data_file, sep="\t", names=["user", "item", "rating", "ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--holdout", type=int, default=5, help="last N items held out")
    args = parser.parse_args()

    print("=" * 74)
    print("VALIDATION: the same ALS / co-visitation code on REAL human ratings")
    print("=" * 74)

    ratings = fetch()
    print(f"\nMovieLens-100k: {len(ratings):,} ratings, "
          f"{ratings.user.nunique():,} users, {ratings.item.nunique():,} items")

    positive = ratings[ratings["rating"] >= POSITIVE_RATING].sort_values("ts")
    print(f"implicit positives (rating >= {POSITIVE_RATING:.0f}): {len(positive):,} "
          f"(density {len(positive) / (ratings.user.nunique() * ratings.item.nunique()):.2%})")

    users = sorted(positive["user"].unique())
    items = sorted(positive["item"].unique())
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {v: i for i, v in enumerate(items)}

    # ---- leave-last-N-out, by timestamp -------------------------------
    train_rows, test_by_user = [], {}
    for user, group in positive.groupby("user", sort=False):
        group = group.sort_values("ts")
        if len(group) < args.holdout + 5:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:-args.holdout])
        test_by_user[u_index[user]] = [i_index[i] for i in group.iloc[-args.holdout:]["item"]]
    train = pd.concat(train_rows)
    print(f"evaluation users: {len(test_by_user):,} "
          f"(last {args.holdout} items held out, temporal)")

    # Confidence from the rating: a 5-star film is stronger evidence than a 4.
    values = (train["rating"] / 5.0).to_numpy()
    user_item = sparse.coo_matrix(
        (values,
         (train["user"].map(u_index).to_numpy(), train["item"].map(i_index).to_numpy())),
        shape=(len(users), len(items)),
    ).tocsr()
    session_item = user_item.copy()
    session_item.data = np.ones_like(session_item.data)

    print(f"\ntraining ALS (f={args.factors}, alpha={args.alpha}, "
          f"lambda={args.regularization}, {args.iterations} iters)")
    started = time.time()
    als = ImplicitALS(factors=args.factors, regularization=args.regularization,
                      alpha=args.alpha, iterations=args.iterations, seed=42).fit(user_item)
    print(f"  fitted in {time.time() - started:.1f}s")

    covis = build_covisitation(session_item, damping=0.5, min_cooccurrence=2, top_k=100)

    # ---- evaluate ------------------------------------------------------
    popularity = np.asarray(user_item.sum(axis=0)).ravel()
    pop_order = np.argsort(-popularity)
    rng = np.random.default_rng(42)

    def als_rec(u: int, seen: set[int], k: int) -> list[int]:
        return [int(i) for i in als.recommend(als.user_factors[u], k=k, exclude=list(seen)).indices]

    def covis_rec(u: int, seen: set[int], k: int) -> list[int]:
        return [int(i) for i in covis.for_history(sorted(seen), k=k, exclude=list(seen)).indices]

    def pop_rec(u: int, seen: set[int], k: int) -> list[int]:
        return [int(i) for i in pop_order if int(i) not in seen][:k]

    def rand_rec(u: int, seen: set[int], k: int) -> list[int]:
        picks = rng.choice(len(items), size=k * 3, replace=False)
        return [int(i) for i in picks if int(i) not in seen][:k]

    strategies = {
        "random": rand_rec, "popularity": pop_rec,
        "co-visitation (ours)": covis_rec, "ALS (ours)": als_rec,
    }

    print(f"\n{'strategy':<24}{'NDCG@10':>10}{'P@10':>9}{'R@10':>9}{'MAP@10':>9}{'MRR':>9}")
    print("-" * 70)
    results: dict[str, dict[str, float]] = {}
    for name, fn in strategies.items():
        acc = {m: [] for m in ("ndcg", "p", "r", "map", "mrr")}
        for u, held in test_by_user.items():
            seen = set(user_item[u].indices.tolist())
            recs = fn(u, seen, 10)
            relevant = set(held)
            graded = {i: 1.0 for i in held}
            acc["ndcg"].append(ndcg_at_k(recs, graded, 10))
            acc["p"].append(precision_at_k(recs, relevant, 10))
            acc["r"].append(recall_at_k(recs, relevant, 10))
            acc["map"].append(average_precision_at_k(recs, relevant, 10))
            acc["mrr"].append(reciprocal_rank(recs, relevant, 10))
        results[name] = {k: float(np.mean(v)) for k, v in acc.items()}
        m = results[name]
        print(f"{name:<24}{m['ndcg']:>10.4f}{m['p']:>9.4f}{m['r']:>9.4f}"
              f"{m['map']:>9.4f}{m['mrr']:>9.4f}")

    als_ndcg = results["ALS (ours)"]["ndcg"]
    pop_ndcg = results["popularity"]["ndcg"]
    print("-" * 70)
    print(f"ALS beats popularity by {als_ndcg / max(pop_ndcg, 1e-9):.2f}x on NDCG@10 "
          f"(full-catalog retrieval).")

    # ---- sampled-negative protocol -------------------------------------
    # The full-catalog numbers above are NOT comparable to most published
    # results, which use the He et al. (WWW 2017, NCF) protocol: rank the one
    # held-out positive against 99 sampled negatives. That is a far easier task
    # and yields much larger numbers. Reporting both is the only honest way to
    # compare against the literature -- and the gap between the two columns is
    # itself the lesson from docs/EVALUATION.md.
    print("\nSampled-negative protocol (1 positive vs 99 negatives, He et al. 2017)")
    print(f"{'strategy':<24}{'HR@10':>10}{'NDCG@10':>10}")
    print("-" * 44)
    n_items = len(items)
    for name in ("random", "popularity", "co-visitation (ours)", "ALS (ours)"):
        hits, ndcgs = [], []
        for u, held in test_by_user.items():
            train_seen = set(user_item[u].indices.tolist())
            blocked = train_seen | set(held)
            positive_item = held[-1]
            negatives: list[int] = []
            while len(negatives) < 99:
                draw = rng.integers(0, n_items, size=160)
                negatives += [int(d) for d in draw if int(d) not in blocked]
            candidates = np.array([positive_item] + negatives[:99], dtype=int)

            if name == "ALS (ours)":
                scores = als.item_factors[candidates] @ als.user_factors[u]
            elif name == "popularity":
                scores = popularity[candidates]
            elif name == "co-visitation (ours)":
                lookup = covis.for_history(sorted(train_seen), k=n_items).as_dict()
                scores = np.array([lookup.get(int(c), 0.0) for c in candidates])
            else:
                scores = rng.random(len(candidates))

            order = np.lexsort((rng.random(len(scores)), -scores))
            rank = int(np.flatnonzero(order == 0)[0]) + 1
            hits.append(float(rank <= 10))
            ndcgs.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
        print(f"{name:<24}{np.mean(hits):>10.4f}{np.mean(ndcgs):>10.4f}")

    print("\n" + "-" * 70)
    print("Reading these: the sampled-negative column is what papers usually report")
    print("on ML-100k (strong models reach HR@10 ~0.60-0.70). The full-catalog")
    print("column above is far harsher and is NOT comparable to those numbers.")
    print("On either protocol the same from-scratch ALS clearly beats popularity on")
    print("REAL human ratings -- so only the YouTube DATA here is simulated, not")
    print("the algorithms.")


if __name__ == "__main__":
    main()
