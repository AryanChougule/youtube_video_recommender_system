"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI app named ``app`` in files under
``api/``. Everything below that is the same FastAPI application the container
and the local dev server run -- there is no Vercel-specific code path, so the
deployed behaviour cannot drift from what is tested.
"""

import sys
from pathlib import Path

# The package lives under src/ (src-layout), which is not on the path when
# Vercel imports this file directly.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.api.app import app  # noqa: E402

__all__ = ["app"]
