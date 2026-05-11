"""Dataset manifest CSV writer for EU AI Act Annex IV §2.b lineage.

Each row records lineage for a single `.kicad_sch` training sample.
Columns match the spec §Datasets:

    source_type, source_url, commit_sha, license_spdx, dedup_hash,
    file_size_bytes, kicad_version_before, kicad_version_after
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

Split = Literal["D1", "D2", "D3"]

FIELDNAMES: tuple[str, ...] = (
    "source_type",
    "source_url",
    "commit_sha",
    "license_spdx",
    "dedup_hash",
    "file_size_bytes",
    "kicad_version_before",
    "kicad_version_after",
)


class DatasetManifest:
    """Accumulate manifest rows in memory, then write once via `write()`."""

    def __init__(self, path: Path, split: Split) -> None:
        if split not in ("D1", "D2", "D3"):
            raise ValueError(f"split must be one of D1/D2/D3, got {split!r}")
        self.path = Path(path)
        self.split = split
        self.rows: list[dict[str, object]] = []

    def add(
        self,
        *,
        source_type: str,
        source_url: str,
        commit_sha: str,
        license_spdx: str,
        dedup_hash: str,
        file_size_bytes: int,
        kicad_version_before: str,
        kicad_version_after: str,
    ) -> None:
        self.rows.append(
            {
                "source_type": source_type,
                "source_url": source_url,
                "commit_sha": commit_sha,
                "license_spdx": license_spdx,
                "dedup_hash": dedup_hash,
                "file_size_bytes": file_size_bytes,
                "kicad_version_before": kicad_version_before,
                "kicad_version_after": kicad_version_after,
            }
        )

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES))
            writer.writeheader()
            writer.writerows(self.rows)
