"""NDJSON append-only audit-log writer for EU AI Act Annex IV §7.

Each line of the log is a JSON object with at least an `event_type`
field plus arbitrary structured fields. The log is sha256-signed at
the end of a run via `sha256_manifest`; tampering can be detected
later via `verify`.
"""

from __future__ import annotations

import fcntl
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
    opens the file in append mode, acquires an exclusive POSIX flock,
    writes one JSON line, flushes, fsyncs, then releases the lock on
    close.  This is safe for concurrent use from multiple processes.
    """

    def __init__(self, path: Path) -> None:
        """Initialise the logger and ensure the parent directory exists.

        The parent directory is created with ``mkdir(parents=True,
        exist_ok=True)``.  ``path`` is operator-trusted input; mkdir
        follows symlinks as usual.

        Args:
            path: Destination file for the NDJSON log.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        """Append one JSON record to the log file.

        An exclusive ``fcntl.LOCK_EX`` flock is held for the entire
        write+flush+fsync critical section, preventing interleaved
        writes from concurrent processes (POSIX O_APPEND alone only
        guarantees atomicity up to PIPE_BUF ≈ 4 KB).  The lock is
        released automatically when the file handle is closed.

        A UTC ISO-8601 ``timestamp`` is injected unless the caller
        supplies one explicitly.

        Args:
            event_type: Discriminator string stored as ``event_type``.
            **fields: Arbitrary structured fields merged into the record.
        """
        if "timestamp" not in fields:
            fields["timestamp"] = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {"event_type": event_type, **fields}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def sha256_manifest(log_path: Path) -> str:
    """Return the hex sha256 of the log file's bytes.

    Args:
        log_path: Path to the NDJSON log file.

    Returns:
        Lower-case hex digest string (64 characters).

    Raises:
        FileNotFoundError: If ``log_path`` does not exist.
    """
    h = hashlib.sha256()
    with Path(log_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(log_path: Path, expected_sha: str) -> bool:
    """Return True iff the file's current sha256 matches ``expected_sha``."""
    return sha256_manifest(log_path) == expected_sha
