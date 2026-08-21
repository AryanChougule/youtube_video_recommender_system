"""Adapter: the Kaggle "Trending YouTube Video Statistics" dumps.

Drop any of the ``*videos.csv`` files (USvideos.csv, GBvideos.csv, ...) into
``data/raw/`` and the pipeline picks them up automatically.

Two quirks of that dataset matter:

* **It is a trending snapshot per day**, so the same video appears on many
  rows with growing view counts.  We keep the row with the highest view count
  per ``video_id`` -- the final observed state.
* **There is no video duration**, because the trending export omits
  ``contentDetails``.  Duration matters to us (it drives watch-time and the
  duration-fit feature), so we impute it from the category median rather than
  silently zero-filling.  This is an assumption, and it is flagged as one in
  ``docs/ASSUMPTIONS.md``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import coerce_catalog, join_tags

# Column aliases across the country variants of the dataset.
COLUMN_ALIASES = {
    "video_id": ["video_id"],
    "title": ["title"],
    "channel_title": ["channel_title", "channelTitle"],
    "category_id": ["category_id", "categoryId"],
    "tags": ["tags"],
    "description": ["description"],
    "published_at": ["publish_time", "publishedAt", "publish_date"],
    "view_count": ["views", "view_count"],
    "like_count": ["likes", "like_count"],
    "comment_count": ["comment_count", "comments"],
    "thumbnail_url": ["thumbnail_link", "thumbnail_url"],
}

# YouTube's own numeric category ids (from the *_category_id.json side files).
CATEGORY_ID_NAMES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 19: "Travel", 20: "Gaming", 22: "Howto & Style",
    23: "Entertainment", 24: "Entertainment", 25: "News & Politics",
    26: "Howto & Style", 27: "Education", 28: "Science & Technology",
    29: "Nonprofits & Activism", 30: "Movies", 43: "Shows",
}

# Median duration in seconds per category, used only when the source omits it.
CATEGORY_DURATION_FALLBACK = {
    "Music": 240, "Gaming": 900, "Entertainment": 600, "Howto & Style": 540,
    "Education": 780, "Science & Technology": 660, "Sports": 480,
    "Autos & Vehicles": 720, "Travel": 660, "News & Politics": 300,
    "Film & Animation": 420, "Pets & Animals": 180,
}
DEFAULT_DURATION = 600


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def _load_category_map(raw_dir: Path) -> dict[int, str]:
    """Merge any ``*_category_id.json`` side files shipped with the dataset."""
    mapping = dict(CATEGORY_ID_NAMES)
    for path in raw_dir.glob("*category_id.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("items", []):
                mapping[int(item["id"])] = item["snippet"]["title"]
        except (ValueError, KeyError, OSError):
            continue    # side file is optional; the built-in map is enough
    return mapping


def load_catalog(
    csv_glob: str = "data/raw/*videos.csv",
    target_size: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"no CSVs matched {csv_glob!r}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        # These exports are latin-1 in several country variants.
        for encoding in ("utf-8", "latin-1"):
            try:
                frames.append(pd.read_csv(path, encoding=encoding))
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"  [kaggle] could not decode {path}, skipping")
    if not frames:
        raise RuntimeError("no readable CSVs found")

    raw = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  [kaggle] {len(paths)} file(s), {len(raw):,} raw rows")

    out = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        series = _pick(raw, aliases)
        if series is not None:
            out[target] = series

    if "video_id" not in out.columns:
        raise RuntimeError("CSV has no video_id column; is this the right dataset?")

    category_map = _load_category_map(Path("data/raw"))
    if "category_id" in out.columns:
        out["category"] = (
            pd.to_numeric(out["category_id"], errors="coerce")
            .map(lambda x: category_map.get(int(x), "Entertainment") if pd.notna(x) else "Entertainment")
        )
    else:
        out["category"] = "Entertainment"

    # Trending dumps repeat a video across days -> keep its peak observed state.
    out["view_count"] = pd.to_numeric(out.get("view_count", 0), errors="coerce").fillna(0)
    out = out.sort_values("view_count").drop_duplicates("video_id", keep="last")

    # The export has no channel_id; derive a stable surrogate from the name so
    # channel-affinity recall still works.
    titles = out.get("channel_title", pd.Series([""] * len(out), index=out.index)).fillna("")
    out["channel_id"] = "UC" + pd.util.hash_pandas_object(titles, index=False).astype("uint64").astype(str).str[:20]

    # Tags in this dataset are pipe-separated and quoted: `"tag1"|"tag2"`.
    if "tags" in out.columns:
        out["tags"] = out["tags"].fillna("").map(
            lambda s: join_tags([t.strip().strip('"') for t in str(s).split("|")
                                 if t.strip().strip('"') not in ("", "[none]")])
        )

    durations = out["category"].map(CATEGORY_DURATION_FALLBACK).fillna(DEFAULT_DURATION)
    out["duration_seconds"] = durations.astype("int64")
    out["source"] = "kaggle"

    catalog = coerce_catalog(out)
    if target_size:
        # Keep the most-viewed slice: a trending dump's long tail is mostly
        # near-duplicate re-uploads and adds little structure.
        catalog = catalog.nlargest(target_size, "view_count").reset_index(drop=True)
    if verbose:
        print(f"  [kaggle] {len(catalog):,} unique videos "
              f"(duration imputed from category median -- see docs/ASSUMPTIONS.md)")
    return catalog
