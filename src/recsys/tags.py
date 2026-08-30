"""Tag encoding, isolated from pandas.

Tags are stored as a single delimiter-joined string so the catalog stays a flat
table across parquet, JSON and NumPy. Both the build path and the serving path
need to split them, but only the build path needs pandas -- so these three
symbols live here rather than in ``recsys.data.schema``, which imports pandas at
module level. Keeping them separate is what lets the serving bundle exclude
pandas entirely (55MB).
"""

from __future__ import annotations

from typing import List

TAG_SEP = "|"


def split_tags(value: object) -> List[str]:
    """Turn a stored tag string back into a list."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [t.strip() for t in text.split(TAG_SEP) if t.strip()]


def join_tags(tags: List[str]) -> str:
    return TAG_SEP.join(t.strip() for t in tags if t and t.strip())
