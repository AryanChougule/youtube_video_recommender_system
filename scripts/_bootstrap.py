"""Make ``recsys`` importable when running scripts without installing.

``pip install -e .`` is the recommended setup, but the scripts must also work
straight out of a fresh clone, so every script imports this first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin BLAS to one thread BEFORE numpy is imported.
#
# Every stage of this pipeline is dominated by many SMALL linear-algebra ops
# (96x96 ALS solves, 256-dim dot products) rather than a few large ones. For
# those, multi-threaded OpenBLAS spends more time spawning and synchronising a
# thread team than doing arithmetic -- measured at 3652us vs 51us for a single
# 96x96 solve, a 70x penalty. The few genuinely large matmuls here are ~1.5M
# flops and finish in microseconds single-threaded either way.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
