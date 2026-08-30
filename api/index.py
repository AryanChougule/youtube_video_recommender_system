"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI app named ``app`` in files under
``api/``. Everything below is the same FastAPI application the container and
the local dev server run -- there is no Vercel-specific code path, so the
deployed behaviour cannot drift from what the tests exercise.

The one genuinely platform-specific problem is path recovery.

Why the shim exists
-------------------
One ASGI app owning the whole domain needs a catch-all rewrite. The obvious
spelling of that is::

    { "source": "/(.*)", "destination": "/api/index" }

and it does not work. The function is invoked, but every request arrives with
``scope["path"] == "/api/index"`` regardless of what the client asked for, so
FastAPI answers ``404 {"detail":"Not Found"}`` to everything -- including its
own ``/docs`` and ``/openapi.json``, which is the tell that the routing table
was never the problem.

The original path is not recoverable from the request either. Dumping the full
ASGI scope and every header from the deployed function showed the path present
in neither: ``x-forwarded-host`` survives the rewrite, ``x-forwarded-path`` and
friends do not exist, and only the query string comes through intact.

So the path has to be carried in the destination instead::

    { "source": "/(.*)", "destination": "/api/index/$1" }

which invokes the same function with ``/api/index/<original path>``. This shim
strips that prefix back off before Starlette's router sees it. That is the
whole job: five lines, and a deployment that routes.
"""

import sys
from pathlib import Path

# The package lives under src/ (src-layout), which is not on the path when
# Vercel imports this file directly.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.api.app import app as _app  # noqa: E402

#: The rewrite destination prefix from vercel.json. Kept in one place because
#: the two must agree, and a silent disagreement 404s the entire site.
MOUNT_PREFIX = "/api/index"


def strip_mount_prefix(path: str) -> str:
    """Undo the vercel.json rewrite: ``/api/index/foo`` -> ``/foo``.

    Idempotent for paths that do not carry the prefix, so running the same app
    locally or in Docker -- where no rewrite happens -- is unaffected.
    """
    if path == MOUNT_PREFIX:
        return "/"
    if path.startswith(MOUNT_PREFIX + "/"):
        return path[len(MOUNT_PREFIX):] or "/"
    return path


async def app(scope, receive, send):
    if scope["type"] in ("http", "websocket"):
        original = scope.get("path", "/")
        restored = strip_mount_prefix(original)
        if restored != original:
            scope = dict(scope)
            scope["path"] = restored
            # raw_path is authoritative for some middleware; keep it in step
            # rather than leaving the two disagreeing.
            if scope.get("raw_path"):
                scope["raw_path"] = restored.encode()
    await _app(scope, receive, send)


__all__ = ["app", "strip_mount_prefix", "MOUNT_PREFIX"]
