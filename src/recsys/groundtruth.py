"""Access to the simulator's hidden generative variables, for evaluation only.

Two variables drive behaviour in the simulator and are never observable by any
model: ``latent_quality`` and ``latent_clickbait``. They are what make the
honest questions answerable --

* does the ranker recover genuine quality, or just popularity?
* does multi-objective ranking actually reduce clickbait exposure?

-- and they are exactly what must never reach a served response, because a model
that could see them would be evaluating itself against its own answer key.

Where they live, and why that matters
-------------------------------------
The build writes them to ``data/processed/catalog.parquet``. The **serving**
bundle (``recsys.catalog_view.CatalogView``) excludes them by construction, so
leaking them through the API is structurally impossible rather than merely
forbidden by a test.

That exclusion is correct, and it broke three evaluation call sites that had
been reading ground truth from ``artifacts.catalog`` back when the catalog was
one DataFrame shared by everything. Two started raising ``KeyError``; the third
had an ``if column in catalog else np.zeros(...)`` fallback and silently
reported **clickbait@1 = 0.0000 for every strategy** -- a number that looks
like a finding and is an absence.

Hence this module. Ground truth comes from the parquet, and asking for it when
it is not there is an **error**, never a default. A missing answer key must stop
an evaluation, not quietly pass it.
"""

from __future__ import annotations

import numpy as np

from .config import Paths


class GroundTruthUnavailable(RuntimeError):
    """Raised when a hidden generative variable was requested but is absent."""


def load_latent(name: str, expected_rows: int | None = None) -> np.ndarray:
    """Return one hidden generative variable, in catalog row order.

    Parameters
    ----------
    name:
        Column in ``data/processed/catalog.parquet``, e.g. ``latent_quality``.
    expected_rows:
        If given, assert the column length matches. Ground truth silently
        misaligned with the catalog would produce plausible, wrong numbers --
        the most expensive kind of bug in an evaluation harness.

    Raises
    ------
    GroundTruthUnavailable
        If the parquet or the column is missing. Deliberately not a warning and
        deliberately not a zero-filled fallback.
    """
    import pandas as pd            # build-time only; see recall/cf.py

    if not Paths.catalog.exists():
        raise GroundTruthUnavailable(
            f"{Paths.catalog} not found. Hidden generative variables live in the "
            "build output, not in the serving bundle (which excludes them on "
            "purpose). Run `python scripts/01_build_data.py` first."
        )
    frame = pd.read_parquet(Paths.catalog)
    if name not in frame.columns:
        raise GroundTruthUnavailable(
            f"column {name!r} is not in {Paths.catalog.name}. Available latent "
            f"columns: {[c for c in frame.columns if c.startswith('latent_')] or 'none'}."
        )
    values = frame[name].to_numpy(dtype=np.float64)
    if expected_rows is not None and len(values) != expected_rows:
        raise GroundTruthUnavailable(
            f"{name} has {len(values)} rows but the catalog has {expected_rows}. "
            "The build output and the serving artifacts are out of step -- rerun "
            "`python scripts/build_all.py`."
        )
    return values
