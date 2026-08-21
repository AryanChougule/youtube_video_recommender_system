"""The single source of "now" for the whole build.

Why this module exists
----------------------
The catalog generator assigns publish dates relative to now, and the simulator
places sessions in a 90-day window ending now. Calling ``utcnow()`` in each of
them made the build depend on wall-clock time, so two runs with the same seed
produced different data -- quietly falsifying the reproducibility claim in the
README.

Setting ``project.reference_date`` in config.yaml pins it, and the entire
pipeline becomes byte-identical across machines and across days. Leaving it
null uses today, which is what you want for a live demo where "trending" should
mean trending *now*.

Freshness features are anchored to this same instant, so a pinned build also
keeps age-based features stable -- otherwise a rebuild a month later would
silently shift every `age_days` value and change what the ranker learned.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def reference_now(reference_date: str | None = None) -> pd.Timestamp:
    """Resolve the build's reference instant.

    ``reference_date`` may be any string pandas can parse (``"2026-01-01"``).
    ``None`` means "use the real clock", normalised to midnight so at least a
    single day's builds agree with each other.
    """
    if reference_date:
        return pd.Timestamp(reference_date).normalize()
    return pd.Timestamp(datetime.utcnow()).normalize()
