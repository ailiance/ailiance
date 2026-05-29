"""Serve the Gaia-X credential set under /.well-known on the gateway."""
from __future__ import annotations

import os
from pathlib import Path

from starlette.staticfiles import StaticFiles


def mount_well_known(app, directory: "str | Path | None" = None) -> bool:
    """Mount the Gaia-X well-known directory as static files on *app*.

    Returns True if mounted, False if the directory does not exist (so the
    gateway starts fine before artifacts are rendered). Directory defaults to
    $GAIA_X_WELL_KNOWN_DIR or 'var/well-known'.
    """
    path = Path(directory or os.environ.get("GAIA_X_WELL_KNOWN_DIR", "var/well-known"))
    if not path.is_dir():
        return False
    app.mount("/.well-known", StaticFiles(directory=str(path)), name="well-known")
    return True
