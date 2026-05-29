"""Dataset manifest CSV writer for EU AI Act Annex IV §2.b lineage.

Each row records lineage for a single `.kicad_sch` training sample.
Columns match the spec §Datasets (8 columns — split is NOT a column,
it is encoded in the output filename; see ``DatasetManifest.write()``):

    source_type, source_url, commit_sha, license_spdx, dedup_hash,
    file_size_bytes, kicad_version_before, kicad_version_after

Trust assumption: ``path`` is operator-trusted input. ``mkdir(parents=True)``
follows symlinks — same trust model as audit_log.
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
    """Accumulate manifest rows in memory, then write once via ``write()``.

    ``path`` is the caller-provided base path (operator-trusted input).
    The actual file written is split-encoded — see ``output_path``.
    """

    def __init__(self, path: Path, split: Split) -> None:
        if split not in ("D1", "D2", "D3"):
            raise ValueError(f"split must be one of D1/D2/D3, got {split!r}")
        self.path = Path(path)
        self.split = split
        self.rows: list[dict[str, object]] = []

    @property
    def output_path(self) -> Path:
        """Return the split-encoded output path.

        If ``self.path``'s stem already ends with ``_{split}``, return it
        as-is (idempotent — callers may pass the conventional name directly).
        Otherwise, insert the split before the suffix:
        ``manifest.csv`` + ``D1`` → ``manifest_D1.csv``.
        """
        stem = self.path.stem
        suffix = self.path.suffix
        if stem.endswith(f"_{self.split}"):
            return self.path
        return self.path.with_name(f"{stem}_{self.split}{suffix}")

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
        """Append a lineage row. Validates types and key fields at runtime.

        Raises:
            TypeError: if ``file_size_bytes`` is not a plain ``int`` (bool is
                explicitly rejected even though it is an int subclass), or if
                any string field is not a ``str``.
            ValueError: if ``file_size_bytes`` is negative, or if
                ``dedup_hash``, ``source_type`` or ``source_url`` is
                empty/blank.

        Note:
            ``commit_sha`` and ``license_spdx`` MAY be empty: D3 mix rows
            inherit lineage from their source rows and carry no own commit
            or license. Per-split license/source policy is enforced
            separately by ``lineage_validator``, not here.
        """
        # -- file_size_bytes --
        if isinstance(file_size_bytes, bool):
            raise TypeError(
                "file_size_bytes must be int, got bool"
            )
        if not isinstance(file_size_bytes, int):
            raise TypeError(
                f"file_size_bytes must be int, got {type(file_size_bytes).__name__!r}"
            )
        if file_size_bytes < 0:
            raise ValueError(
                f"file_size_bytes must be >= 0, got {file_size_bytes!r}"
            )

        # -- str type for every string field (catches type confusion) --
        _str_fields = {
            "source_type": source_type,
            "source_url": source_url,
            "commit_sha": commit_sha,
            "license_spdx": license_spdx,
            "dedup_hash": dedup_hash,
            "kicad_version_before": kicad_version_before,
            "kicad_version_after": kicad_version_after,
        }
        for field, value in _str_fields.items():
            if not isinstance(value, str):
                raise TypeError(
                    f"{field} must be a str, got {type(value).__name__!r}"
                )

        # -- non-empty required (commit_sha/license_spdx MAY be empty: D3
        #    inherited rows; policy enforced by lineage_validator) --
        _non_empty = {
            "dedup_hash": dedup_hash,
            "source_type": source_type,
            "source_url": source_url,
        }
        for field, value in _non_empty.items():
            if not value.strip():
                raise ValueError(
                    f"{field} must be a non-empty str, got {value!r}"
                )

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
        """Write accumulated rows to a split-encoded CSV file.

        Output path: ``output_path`` (e.g. ``manifest_D1.csv``) so the
        split is recoverable from the artifact name without inspecting the
        file content. Overwrites any existing file at that path.
        Path.parent is created (mkdir parents=True, follows symlinks —
        operator-trusted input).
        """
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES))
            writer.writeheader()
            writer.writerows(self.rows)
