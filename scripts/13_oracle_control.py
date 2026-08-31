"""The oracle control: is the full-catalog metric measuring anything at all?

    python scripts/13_oracle_control.py

Why this exists
---------------
`scripts/05_evaluate.py` reports that adding the learned ranker *lowers*
full-catalog NDCG@10 (0.0125 -> 0.0107) while *raising* Protocol A top-1
(0.1840 -> 0.1930). Two metrics disagreeing about the same change means one of
them is wrong for this purpose, and the natural objection is: maybe the model
is just bad.

The oracle answers that objection. It scores candidates using the simulator's
own generative parameters -- the exact quantities that decided whether each
click happened:

    CLICK_BIAS
    + W_AFFINITY     * <user topic vector, item topic vector>
    + W_POPULARITY   * mainstream[user] * z(log views)
    + W_QUALITY      * z(log latent_quality)
    + W_DURATION_FIT * -|log(duration / preferred_duration)|

No real model can beat this, because it *is* the data-generating process. So
its score is an upper bound on what the metric can reward. If a metric ranks
the oracle below a popularity list, the metric is not measuring recommendation
quality. That is a statement about the metric, and it needs no model at all.

This runs separately from stage 5 because it needs the hidden generative
variables, which the serving artifacts deliberately exclude
(see `src/recsys/groundtruth.py`), and because it is cheap: pure arithmetic
over precomputed arrays, no feature building.

Honest note on what changed
---------------------------
On an earlier build -- before `latent_clickbait` was added to the generator --
the oracle scored *below* the popularity baseline outright, which was an even
sharper version of the same argument. The current generator gives the oracle a
win over popularity. Both builds point the same way: the ranking that wins on
full-catalog NDCG is not the ranking that wins on observed labels. The number
this script prints is the one that reproduces today.
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
from recsys.counterfactual import make_scorers
from recsys.evaluate import build_eval_users
from recsys.metrics import ndcg_at_k
from recsys.split import temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg, with_ranker=True)
    started = time.time()

    for path in (Paths.gt_user_topics, Paths.gt_item_topics, Paths.users):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. The oracle needs the simulator's ground truth; "
                "run `python scripts/01_build_data.py` first."
            )

    interactions = pd.read_parquet(Paths.interactions)
    users_df = pd.read_parquet(Paths.users)
    gt_user_topics = np.load(Paths.gt_user_topics)
    gt_item_topics = np.load(Paths.gt_item_topics)

    split = temporal_split(interactions, test_size=cfg.ranker.test_size)
    max_users = args.users or cfg.evaluation.max_eval_users
    users = build_eval_users(interactions, split, art.video_index,
                             min_history=3, min_holdout=1,
                             max_users=max_users, seed=cfg.project.seed)

    scorers = make_scorers(art, include_oracle=True,
                           gt_user_topics=gt_user_topics,
                           gt_item_topics=gt_item_topics, users_df=users_df)
    make_oracle = scorers.pop("_oracle_factory", None)
    if make_oracle is None:
        raise SystemExit(
            "make_scorers did not return an oracle factory -- ground truth is "
            "present but was not wired through."
        )

    print("=" * 78)
    print("ORACLE CONTROL -- full-catalog NDCG@%d" % args.k)
    print("=" * 78)
    print(f"\nusers   : {len(users):,}")
    print(f"catalog : {art.n_items:,} items")
    print("\nscoring every catalog item for every user with the generative")
    print("parameters that actually produced the clicks...\n")

    user_row = {str(u): i for i, u in enumerate(users_df["user_id"])}
    all_items = np.arange(art.n_items)

    scores_by_name: dict[str, list[float]] = {"ORACLE": [], "popularity": [],
                                              "content_only": []}
    skipped = 0
    for user in users:
        row = user_row.get(str(user.user_id))
        if row is None:
            skipped += 1
            continue
        for name in scores_by_name:
            fn = make_oracle(row) if name == "ORACLE" else scorers[
                {"popularity": "popularity", "content_only": "content_only"}[name]]
            values = np.asarray(fn(user.history, user.weights, all_items),
                                dtype=np.float64)
            # A recommender never re-shows the watch history.
            if len(user.history):
                values[np.asarray(user.history, dtype=int)] = -np.inf
            top = np.argpartition(-values, args.k)[:args.k]
            top = top[np.argsort(-values[top])]
            scores_by_name[name].append(
                ndcg_at_k([int(i) for i in top], user.relevance, args.k))

    print(f"    {'scorer':<26}{'NDCG@%d' % args.k:>10}")
    print("    " + "-" * 36)
    report = {}
    for name, values in sorted(scores_by_name.items(),
                               key=lambda kv: -float(np.mean(kv[1]))):
        mean = float(np.mean(values))
        report[name] = round(mean, 6)
        print(f"    {name:<26}{mean:>10.4f}")

    if skipped:
        print(f"\n    ({skipped} users skipped: not present in users.parquet)")

    oracle, pop = report["ORACLE"], report["popularity"]
    print("\nREADING THIS:")
    if oracle > pop:
        print(f"  The oracle ({oracle:.4f}) beats popularity ({pop:.4f}), so the")
        print("  full-catalog metric is NOT pure noise -- there is real headroom")
        print("  our models do not capture.")
    else:
        print(f"  The oracle ({oracle:.4f}) LOSES to popularity ({pop:.4f}).")
        print("  A metric that ranks the data-generating process below a")
        print("  popularity list is not measuring recommendation quality.")
    print("\n  The baselines here come from counterfactual.make_scorers, which")
    print("  builds the user profile slightly differently from the stage-5")
    print("  baselines, so content_only reads a little higher than it does in")
    print("  05_evaluate.py. The comparison WITHIN this table is the point.")
    print("\n  Either way, the ranking that wins here is not the ranking that")
    print("  wins on observed labels (Protocol A), which is why the")
    print("  counterfactual protocols are the ones quoted. See docs/EVALUATION.md.")

    out = Paths.artifacts / "oracle_control.json"
    out.write_text(json.dumps({
        "ndcg_at_k": args.k,
        "n_users": len(users) - skipped,
        "results": report,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(Paths.root)}")
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
