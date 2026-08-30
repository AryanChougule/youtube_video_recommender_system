"""Build everything from scratch, in order.

    python scripts/build_all.py                # full build (~7 min)
    python scripts/build_all.py --quick        # small build for smoke tests (~1 min)
    python scripts/build_all.py --skip-eval    # skip stage 5

The whole system is reproducible from ``config.yaml`` + ``project.seed``, so a
fresh clone plus this one command produces byte-identical artifacts.

Stage order is a hard dependency chain:

    01 data      catalog + simulated watch log
    02 features  text vectors (needs catalog)
    03 cf        co-visitation + ALS  (needs log; respects the temporal cutoff)
    04 ranker    learning-to-rank     (needs 02 + 03; cross-fits CF per fold)
    05 evaluate  offline metrics      (needs everything)
    12 export    NumPy serving bundle (needs 02 + 04; drops scikit-learn)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGES = [
    ("01_build_data.py", "catalog + simulated watch log"),
    ("02_build_features.py", "text vectors + item stats"),
    ("03_train_cf.py", "co-visitation + implicit ALS"),
    ("04_train_ranker.py", "learning-to-rank model"),
    ("05_evaluate.py", "offline evaluation + ablation"),
    ("12_export_serving.py", "NumPy-only serving bundle"),
]


def run(script: str, extra: list[str]) -> float:
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPTS / script), *extra],
        cwd=str(SCRIPTS.parent),
    )
    if result.returncode != 0:
        raise SystemExit(f"\n!! {script} failed with exit code {result.returncode}")
    return time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny build: 600 users, 1500 videos")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--source", default=None,
                        choices=["auto", "youtube_api", "kaggle", "synthetic"])
    args = parser.parse_args()

    timings: list[tuple[str, float]] = []
    started = time.time()

    for script, description in STAGES:
        if args.skip_eval and script.startswith("05"):
            continue
        extra: list[str] = []
        if script.startswith("01"):
            if args.quick:
                extra += ["--users", "600", "--videos", "1500"]
            if args.source:
                extra += ["--source", args.source]
        if script.startswith("04") and args.quick:
            extra += ["--folds", "2"]
        if script.startswith("05") and args.quick:
            extra += ["--quick"]

        print(f"\n\n>>> {script}  --  {description}\n")
        timings.append((script, run(script, extra)))

    print("\n" + "=" * 72)
    print("BUILD COMPLETE")
    print("=" * 72)
    for script, seconds in timings:
        print(f"  {script:<26} {seconds:>7.1f}s")
    print(f"  {'TOTAL':<26} {time.time() - started:>7.1f}s")
    print("\nStart the app with:\n"
          "  python -m uvicorn recsys.api.app:app --app-dir src --port 7860\n"
          "then open http://localhost:7860")


if __name__ == "__main__":
    main()
