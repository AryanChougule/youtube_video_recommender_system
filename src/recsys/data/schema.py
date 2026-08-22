"""Canonical schemas for the catalog and the interaction log.

Every data source (YouTube API / Kaggle CSV / simulator) is normalised into
these two frames.  Fixing the schema at the boundary means the rest of the
system never has to care where the data came from -- that is the whole point
of an adapter layer, and it is what lets us swap synthetic data for real
YouTube data without touching a single line of modelling code.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..clock import reference_now as clock_reference_now

TAG_SEP = "|"

#: One row per video.
CATALOG_SCHEMA: dict[str, str] = {
    "video_id": "string",
    "title": "string",
    "channel_id": "string",
    "channel_title": "string",
    "category": "string",
    "tags": "string",            # TAG_SEP-joined; see :func:`split_tags`
    "description": "string",
    "published_at": "datetime64[ns]",
    "duration_seconds": "int64",
    "view_count": "int64",
    "like_count": "int64",
    "comment_count": "int64",
    "thumbnail_url": "string",
    "source": "string",          # youtube_api | kaggle | synthetic
}

#: One row per impression.  Note we log *impressions*, not just clicks --
#: without the un-clicked rows you cannot correct for position bias later,
#: and you have no negatives to train a ranker on.
INTERACTION_SCHEMA: dict[str, str] = {
    "user_id": "string",
    "video_id": "string",
    "session_id": "string",
    "ts": "datetime64[ns]",
    "rank_shown": "int64",       # position in the shown list (0-based)
    "clicked": "int64",          # 1 if the user opened the video
    "watch_fraction": "float64", # fraction of the video actually watched
    "watch_seconds": "float64",
    "liked": "int64",
    # Multi-objective labels. `satisfied` is deliberately NOT a function of
    # watch time alone -- clickbait produces long watches and low satisfaction,
    # which is the whole reason a multi-task ranker is worth building.
    "satisfied": "int64",
    "dismissed": "int64",
}


def split_tags(value: object) -> List[str]:
    """Turn a stored tag string back into a list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [t.strip() for t in text.split(TAG_SEP) if t.strip()]


def join_tags(tags: List[str]) -> str:
    return TAG_SEP.join(t.strip() for t in tags if t and t.strip())


def coerce(df: pd.DataFrame, schema: dict[str, str], name: str) -> pd.DataFrame:
    """Force ``df`` into ``schema``: add missing columns, cast, drop extras."""
    out = pd.DataFrame(index=df.index)
    for col, dtype in schema.items():
        if col in df.columns:
            series = df[col]
        else:
            series = pd.Series([None] * len(df), index=df.index)
        if dtype.startswith("datetime"):
            out[col] = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
        elif dtype == "int64":
            out[col] = pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
        elif dtype == "float64":
            out[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")
        else:
            out[col] = series.astype("string").fillna("")
    missing = [c for c in schema if c not in df.columns]
    if missing:
        print(f"  [schema] {name}: filled {len(missing)} missing column(s): {missing}")
    return out


def coerce_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce(df, CATALOG_SCHEMA, "catalog")
    out = out[out["video_id"].str.len() > 0]
    out = out.drop_duplicates(subset="video_id", keep="first").reset_index(drop=True)
    # A video with no title is unusable downstream (no text features, nothing
    # to render in the UI), so drop rather than impute.
    out = out[out["title"].str.strip().str.len() > 0].reset_index(drop=True)
    return out


def coerce_interactions(df: pd.DataFrame) -> pd.DataFrame:
    return coerce(df, INTERACTION_SCHEMA, "interactions").reset_index(drop=True)


def catalog_reference_now(catalog: pd.DataFrame,
                          reference_date: str | None = None) -> pd.Timestamp:
    """The 'current time' used for freshness/recency features.

    Two behaviours, both necessary:

    * For a live catalog this is simply now -- or ``reference_date`` when the
      build clock is pinned, so ``age_days`` stays reproducible.
    * For a *static* dataset (e.g. the 2017-18 Kaggle dump) the real clock would
      make every video ~8 years old and collapse freshness to a constant, so we
      anchor to the newest video in the catalog instead.

    Named distinctly from :func:`recsys.clock.reference_now`, which resolves the
    build clock itself; this one derives a catalog-aware anchor from it.
    """
    newest = pd.to_datetime(catalog["published_at"]).max()
    now = clock_reference_now(reference_date)
    if pd.isna(newest):
        return now
    return max(newest + pd.Timedelta(days=1), now) if newest > now - pd.Timedelta(days=365) else newest + pd.Timedelta(days=1)


def derive_catalog_features(catalog: pd.DataFrame, now: pd.Timestamp | None = None,
                            reference_date: str | None = None) -> pd.DataFrame:
    """Add the cheap, always-available item features used by recall + ranking."""
    df = catalog.copy()
    now = now or catalog_reference_now(df, reference_date)

    df["age_days"] = (now - pd.to_datetime(df["published_at"])).dt.total_seconds() / 86400.0
    df["age_days"] = df["age_days"].clip(lower=0.0).fillna(365.0)

    views = df["view_count"].clip(lower=0)
    df["log_views"] = np.log1p(views)
    # Engagement *rate* is far more informative than raw counts: it separates a
    # video people loved from a video that merely got a lot of impressions.
    df["like_rate"] = (df["like_count"] / views.replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["comment_rate"] = (df["comment_count"] / views.replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["engagement_rate"] = (df["like_rate"] + 5.0 * df["comment_rate"]).clip(0, 1)

    # Views-per-day is the closest cheap proxy for "is this trending *now*"
    df["views_per_day"] = views / df["age_days"].clip(lower=1.0)
    df["log_views_per_day"] = np.log1p(df["views_per_day"])

    df["duration_minutes"] = df["duration_seconds"] / 60.0
    df["is_short"] = (df["duration_seconds"] <= 75).astype("int64")
    df["title_length"] = df["title"].str.len().astype("int64")
    df["n_tags"] = df["tags"].map(lambda s: len(split_tags(s))).astype("int64")
    return df
