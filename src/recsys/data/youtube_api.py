"""Adapter: real video metadata from the YouTube Data API v3.

Set ``YOUTUBE_API_KEY`` and the pipeline switches from synthetic to real data
with no other change -- that is the payoff of normalising every source into
``CATALOG_SCHEMA``.

Quota notes (the free tier is 10,000 units/day):
  * ``search.list``  costs 100 units and returns up to 50 ids  -> expensive
  * ``videos.list``  costs   1 unit  and returns up to 50 items -> cheap

So we spend quota on search and batch the detail lookups.  A 6,000-video
catalog needs ~120 searches = 12,000 units, i.e. slightly more than one day of
free quota; ``--max-queries`` lets you build it incrementally across days, and
results are cached to ``data/raw/youtube_api_cache.jsonl`` so a re-run never
re-spends quota on ids it already has.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .schema import coerce_catalog, join_tags
from .topics import CATEGORIES, TOPIC_NAMES, TOPIC_VOCAB, category_topic_weights

API_ROOT = "https://www.googleapis.com/youtube/v3"
_ISO_DURATION = re.compile(
    r"P(?:(?P<days>\d+)D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?"
)


def parse_iso_duration(value: str) -> int:
    """ISO-8601 duration (``PT12M34S``) -> seconds."""
    if not value:
        return 0
    match = _ISO_DURATION.fullmatch(value.strip())
    if not match:
        return 0
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return parts["days"] * 86400 + parts["h"] * 3600 + parts["m"] * 60 + parts["s"]


def build_query_list(queries_per_category: int) -> list[tuple[str, str]]:
    """Search queries derived from our own taxonomy, as (query, category).

    Reusing the micro-topic vocabulary keeps the real catalog topically aligned
    with the synthetic one, so the two are interchangeable downstream.
    """
    out: list[tuple[str, str]] = []
    for category in CATEGORIES:
        weights = sorted(
            category_topic_weights(category).items(), key=lambda kv: -kv[1]
        )
        for slug, _ in weights[:queries_per_category]:
            terms = TOPIC_VOCAB[slug]
            out.append((f"{TOPIC_NAMES[slug]} {terms[0]}", category))
            if len(terms) > 3:
                out.append((f"{terms[2]} {terms[3]}", category))
    return out


def _get(session: requests.Session, endpoint: str, params: dict) -> dict:
    resp = session.get(f"{API_ROOT}/{endpoint}", params=params, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            "YouTube API returned 403 -- quota exhausted or key not authorised "
            f"for the Data API v3. Body: {resp.text[:300]}"
        )
    resp.raise_for_status()
    return resp.json()


def search_video_ids(
    session: requests.Session, api_key: str, query: str, max_results: int
) -> list[str]:
    data = _get(session, "search", {
        "key": api_key, "part": "id", "q": query, "type": "video",
        "maxResults": min(max_results, 50), "relevanceLanguage": "en",
        "safeSearch": "moderate",
    })
    return [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]


def fetch_video_details(
    session: requests.Session, api_key: str, video_ids: Iterable[str]
) -> list[dict]:
    ids = list(video_ids)
    out: list[dict] = []
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        data = _get(session, "videos", {
            "key": api_key, "part": "snippet,contentDetails,statistics",
            "id": ",".join(chunk),
        })
        out.extend(data.get("items", []))
    return out


def _to_row(item: dict, category_hint: str) -> dict:
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})
    thumbs = snippet.get("thumbnails", {})
    thumb = (thumbs.get("medium") or thumbs.get("high") or thumbs.get("default") or {})
    return {
        "video_id": item.get("id", ""),
        "title": snippet.get("title", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "category": category_hint,
        "tags": join_tags(snippet.get("tags", []) or []),
        "description": (snippet.get("description", "") or "")[:1200],
        "published_at": snippet.get("publishedAt"),
        "duration_seconds": parse_iso_duration(content.get("duration", "")),
        "view_count": int(stats.get("viewCount", 0) or 0),
        "like_count": int(stats.get("likeCount", 0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),
        "thumbnail_url": thumb.get("url", ""),
        "source": "youtube_api",
    }


def fetch_catalog(
    api_key: str,
    target_size: int = 6000,
    queries_per_category: int = 6,
    max_results_per_query: int = 50,
    cache_path: Path | None = None,
    sleep: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch a catalog, resuming from cache so quota is never re-spent."""
    session = requests.Session()
    cache_path = cache_path or Path("data/raw/youtube_api_cache.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["video_id"]] = row
        if verbose:
            print(f"  [youtube_api] resumed {len(rows):,} cached videos")

    queries = build_query_list(queries_per_category)
    with cache_path.open("a", encoding="utf-8") as cache_fh:
        for i, (query, category) in enumerate(queries, 1):
            if len(rows) >= target_size:
                break
            try:
                ids = search_video_ids(session, api_key, query, max_results_per_query)
                new_ids = [v for v in ids if v not in rows]
                if not new_ids:
                    continue
                for item in fetch_video_details(session, api_key, new_ids):
                    row = _to_row(item, category)
                    if row["video_id"] and row["title"]:
                        rows[row["video_id"]] = row
                        cache_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache_fh.flush()
            except RuntimeError as exc:            # quota -> keep what we have
                print(f"  [youtube_api] stopping early: {exc}")
                break
            if verbose and i % 10 == 0:
                print(f"  [youtube_api] {i}/{len(queries)} queries, {len(rows):,} videos")
            time.sleep(sleep)

    if not rows:
        raise RuntimeError("YouTube API returned no videos")
    return coerce_catalog(pd.DataFrame(list(rows.values())[:target_size]))


def api_key_from_env(var_name: str = "YOUTUBE_API_KEY") -> str | None:
    key = os.environ.get(var_name, "").strip()
    return key or None
