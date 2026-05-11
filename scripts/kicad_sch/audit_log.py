"""NDJSON append-only audit-log writer for EU AI Act Annex IV §7.

Each line of the log is a JSON object with at least an `event_type`
field plus arbitrary structured fields. The log is sha256-signed at
the end of a run via `sha256_manifest`; tampering can be detected
later via `verify`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 65_536


class AuditLogger:
    """Append-only NDJSON audit logger.

    Multiple instances can target the same path — each `log()` call
    opens the file in append mode, writes one JSON line, and closes.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        if "timestamp" not in fields:
            fields["timestamp"] = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {"event_type": event_type, **fields}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def sha256_manifest(log_path: Path) -> str:
    """Return the hex sha256 of the log file's bytes."""
    h = hashlib.sha256()
    with Path(log_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(log_path: Path, expected_sha: str) -> bool:
    """Return True iff the file's current sha256 matches `expected_sha`."""
    return sha256_manifest(log_path) == expected_sha
