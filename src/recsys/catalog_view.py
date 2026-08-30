"""A pandas-free view of the catalog, used by the serving path.

Why this exists
---------------
The build pipeline is happy to depend on pandas -- it does joins, groupbys and
parquet IO, which is exactly what pandas is for. **Serving does none of that.**
It needs column access, row lookup by index, and two filters.

Carrying pandas into the deployed bundle costs 55MB, which is the difference
between fitting inside a 250MB serverless limit and not:

    with pandas     279 MB   (over)
    without         231 MB   (19 MB headroom)

So the build writes a plain NumPy + JSON bundle and serving reads that. The same
saving applies to the Docker image, so this is not purely a Vercel concession.

The columns listed here are exactly the ones the engine and API touch. Anything
else is deliberately absent -- notably ``latent_quality`` and
``latent_clickbait``, which are hidden generative variables. Leaving them out of
the serving bundle makes leaking them structurally impossible rather than merely
forbidden by a test.

``Column`` deliberately mimics the handful of pandas Series methods the serving
code already used, so the recall and feature classes did not need rewriting for
a deployment concern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

STRING_COLUMNS = ("video_id", "title", "channel_id", "channel_title",
                  "category", "tags", "description", "thumbnail_url",
                  "published_at")
NUMERIC_COLUMNS = ("view_count", "like_count", "comment_count", "duration_seconds")
STATS_COLUMNS = ("age_days", "log_views", "like_rate", "comment_rate",
                 "engagement_rate", "views_per_day", "log_views_per_day",
                 "duration_minutes", "is_short", "title_length", "n_tags")


class Column:
    """The small slice of the pandas Series API the serving path actually used."""

    __slots__ = ("values",)

    def __init__(self, values: np.ndarray):
        self.values = values

    def to_numpy(self, dtype=None) -> np.ndarray:
        return self.values.astype(dtype) if dtype is not None else self.values

    def fillna(self, _value) -> "Column":
        # The bundle is written from already-coerced frames: strings are "" and
        # numerics are 0, never NaN. Kept so call sites read the same.
        return self

    def map(self, fn) -> "Column":
        return Column(np.array([fn(v) for v in self.values], dtype=object))

    def astype(self, dtype) -> "Column":
        return Column(self.values.astype(dtype))

    def tolist(self) -> list:
        return self.values.tolist()

    def head(self, n: int) -> "Column":
        return Column(self.values[:n])

    def dropna(self) -> "Column":
        return self

    @property
    def iloc(self) -> np.ndarray:
        return self.values

    def unique(self) -> np.ndarray:
        return np.array(sorted(set(self.values.tolist())), dtype=object)

    def nunique(self) -> int:
        return len(set(self.values.tolist()))

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __eq__(self, other):                      # type: ignore[override]
        return self.values == other


class Subset:
    """A row selection over a CatalogView, with the accessors call sites use."""

    __slots__ = ("_view", "_idx")

    def __init__(self, view: "CatalogView", idx: np.ndarray):
        self._view = view
        self._idx = np.asarray(idx, dtype=int)

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def index(self) -> np.ndarray:
        return self._idx

    def __getitem__(self, name: str) -> Column:
        return Column(self._view[name].values[self._idx])

    def head(self, n: int) -> "Subset":
        return Subset(self._view, self._idx[:n])

    def nlargest(self, n: int, column: str) -> "Subset":
        values = self._view[column].values[self._idx]
        n = min(n, len(self._idx))
        if n <= 0:
            return Subset(self._view, self._idx[:0])
        top = np.argpartition(-values, n - 1)[:n]
        return Subset(self._view, self._idx[top[np.argsort(-values[top])]])

    def iloc_slice(self, start: int, stop: int) -> "Subset":
        return Subset(self._view, self._idx[start:stop])

    def rows(self) -> list[dict]:
        return [self._view.row(int(i)) for i in self._idx]

    @property
    def empty(self) -> bool:
        return len(self._idx) == 0


class CatalogView:
    """Column-oriented catalog exposing only what serving needs."""

    def __init__(self, strings: dict[str, list[str]], numerics: dict[str, np.ndarray],
                 stats: dict[str, np.ndarray]):
        self._strings = {k: np.asarray(v, dtype=object) for k, v in strings.items()}
        self._numerics = {k: np.asarray(v) for k, v in numerics.items()}
        self._stats = {k: np.asarray(v) for k, v in stats.items()}
        self.n = len(self._strings["video_id"])
        # channel -> item rows, precomputed because ChannelRecall needs the
        # grouping and a groupby is the one pandas call worth replacing outright.
        self.by_channel: dict[str, np.ndarray] = {}
        for i, ch in enumerate(self._strings["channel_id"]):
            self.by_channel.setdefault(ch, []).append(i)
        self.by_channel = {k: np.asarray(v, dtype=int) for k, v in self.by_channel.items()}

    # -- frame-like access ------------------------------------------------
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        # Boolean-mask selection, so `view[view["category"] == "Gaming"]` reads
        # the same as it did with a DataFrame.
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return Subset(self, np.flatnonzero(key))
        if key in self._strings:
            return Column(self._strings[key])
        if key in self._numerics:
            return Column(self._numerics[key])
        return Column(self._stats[key])

    def __contains__(self, name: str) -> bool:
        return name in self._strings or name in self._numerics or name in self._stats

    @property
    def columns(self) -> list[str]:
        return list(self._strings) + list(self._numerics) + list(self._stats)

    def row(self, i: int) -> dict:
        i = int(i)
        out = {k: v[i] for k, v in self._strings.items()}
        out.update({k: v[i] for k, v in self._numerics.items()})
        return out

    # -- the filters the API needs ---------------------------------------
    def indices_where(self, column: str, value) -> np.ndarray:
        return np.flatnonzero(self._strings[column] == value)

    def top_by(self, column: str, k: int, subset: np.ndarray | None = None) -> np.ndarray:
        values = self._numerics[column]
        pool = np.arange(self.n) if subset is None else np.asarray(subset, dtype=int)
        k = min(k, len(pool))
        if k <= 0:
            return np.zeros(0, dtype=int)
        top = pool[np.argpartition(-values[pool], k - 1)[:k]]
        return top[np.argsort(-values[top])]

    def value_counts_index(self, column: str) -> list[str]:
        """Distinct values ordered by frequency, most common first."""
        values = self._strings[column]
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    def largest_channel(self) -> str:
        return max(self.by_channel, key=lambda c: len(self.by_channel[c]))

    # -- persistence ------------------------------------------------------
    def save(self, npz_path: Path, json_path: Path) -> None:
        payload = dict(self._numerics)
        payload.update({f"stat__{k}": v for k, v in self._stats.items()})
        np.savez_compressed(npz_path, **payload)
        json_path.write_text(
            json.dumps({k: [str(x) for x in v] for k, v in self._strings.items()}),
            encoding="utf-8")

    @classmethod
    def load(cls, npz_path: Path, json_path: Path) -> "CatalogView":
        blob = np.load(npz_path, allow_pickle=False)
        strings = json.loads(json_path.read_text(encoding="utf-8"))
        numerics = {k: blob[k] for k in blob.files if not k.startswith("stat__")}
        stats = {k[len("stat__"):]: blob[k] for k in blob.files if k.startswith("stat__")}
        return cls(strings, numerics, stats)

    @classmethod
    def from_frames(cls, catalog, item_stats) -> "CatalogView":
        """Build from pandas frames. Build time only."""
        strings = {c: catalog[c].astype(str).tolist()
                   for c in STRING_COLUMNS if c in catalog.columns}
        strings["published_at"] = [str(v)[:10] for v in catalog["published_at"]]
        numerics = {c: catalog[c].to_numpy(dtype=np.int64)
                    for c in NUMERIC_COLUMNS if c in catalog.columns}
        stats = {c: item_stats[c].to_numpy(dtype=np.float64)
                 for c in STATS_COLUMNS if c in item_stats.columns}
        return cls(strings, numerics, stats)
