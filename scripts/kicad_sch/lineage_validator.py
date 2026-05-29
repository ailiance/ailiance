"""Per-split SPDX license and source_type gate — EU AI Act Annex IV §2.b.

This module is the **lineage validator**: it enforces per-split allowlists on
the ``license_spdx`` and ``source_type`` fields of a dataset manifest.  It is
intentionally kept separate from ``manifest.py`` (single-responsibility: the
manifest *writes*, the validator *validates*).

EU AI Act context
-----------------
Art. 10(2)(b) of the EU AI Act requires that training datasets undergo
"relevant data preparation processing steps, such as annotation, labelling,
cleaning, enrichment and aggregation".  Lineage validation is the gate that
ensures only provenance-tracked, license-compatible samples enter each split:

* **D1** (scraped real schematics) — strict permissive set: MIT, Apache-2.0,
  CC0-1.0, GPL-3.0.  This is the authoritative set from issue #39 and the
  ``scrape_d1`` CLI ``--license-allowlist`` default.
* **D2** (synthetic) — CC0-1.0 only; all synthetic outputs are released
  under CC0-1.0 by the project (``synth_d2.py`` always emits this value).
* **D3** (50/50 mix of D1+D2) — union of D1 and D2 sets, since mixed rows
  inherit the licenses of their component splits.

  .. note::
     D2 and D3 allowlists are derived from the scrapers' ``source_type=``
     calls and the D3 mixer's ``license_spdx=`` field.  They are marked
     ``# PLACEHOLDER — confirm with spec`` and should be reviewed if the
     D2/D3 data pipeline changes.

Source-type allowlists
----------------------
Values are taken from the three pipeline scripts:

* ``github_scrape`` — emitted by ``scrape_d1.py``
* ``synth``         — emitted by ``synth_d2.py``
* ``mix``           — emitted by ``mix_d3.py``

Unknown ``source_type`` values are rejected; the caller decides the
escalation strategy (use ``assert_clean`` for hard stop, inspect
``validate_rows`` results for reporting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.kicad_sch.manifest import DatasetManifest


# ---------------------------------------------------------------------------
# Per-split SPDX allowlists
# ---------------------------------------------------------------------------

#: D1 — real scraped schematics.
#: Authoritative source: issue #39 + ``scrape_d1 --license-allowlist`` default.
#: Strict permissive set; CERN-OHL and other community licenses that the
#: scraper's DEFAULT_LICENSE_ALLOWLIST accepts are **not** in the D1
#: validator set (they are accepted at scrape-time but not for training).
SPDX_D1: frozenset[str] = frozenset({"MIT", "Apache-2.0", "CC0-1.0", "GPL-3.0"})

#: D2 — fully synthetic circuits.  synth_d2.py always emits CC0-1.0.
#: PLACEHOLDER — confirm with spec if D2 ever adds non-CC0 outputs.
SPDX_D2: frozenset[str] = frozenset({"CC0-1.0"})

#: D3 — 50/50 mix of D1 and D2 rows.  Union of both source sets.
#: PLACEHOLDER — confirm with spec if D3 policy diverges from D1∪D2.
SPDX_D3: frozenset[str] = SPDX_D1 | SPDX_D2

_SPDX_ALLOWLISTS: dict[str, frozenset[str]] = {
    "D1": SPDX_D1,
    "D2": SPDX_D2,
    "D3": SPDX_D3,
}

# ---------------------------------------------------------------------------
# Per-split source_type allowlists
# ---------------------------------------------------------------------------

#: Sourced from scrape_d1.py: ``source_type="github_scrape"``
SOURCE_TYPE_D1: frozenset[str] = frozenset({"github_scrape"})

#: Sourced from synth_d2.py: ``source_type="synth"``
#: PLACEHOLDER — confirm with spec if new synth backends add other values.
SOURCE_TYPE_D2: frozenset[str] = frozenset({"synth"})

#: Sourced from mix_d3.py: ``source_type="mix"``
#: PLACEHOLDER — confirm with spec if D3 ever carries raw D1/D2 source_types.
SOURCE_TYPE_D3: frozenset[str] = frozenset({"mix"})

_SOURCE_TYPE_ALLOWLISTS: dict[str, frozenset[str]] = {
    "D1": SOURCE_TYPE_D1,
    "D2": SOURCE_TYPE_D2,
    "D3": SOURCE_TYPE_D3,
}

_VALID_SPLITS: frozenset[str] = frozenset({"D1", "D2", "D3"})


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineageViolation:
    """A single policy violation found in a manifest row.

    Attributes:
        row_index:  Zero-based index of the offending row in the input list.
        field:      Name of the field that failed (``license_spdx`` or
                    ``source_type``).
        value:      The actual value that was rejected.
        reason:     Human-readable explanation including the split name and
                    the allowed set, suitable for audit logs.
    """

    row_index: int
    field: str
    value: str
    reason: str


class LineageError(Exception):
    """Raised by ``assert_clean`` when one or more violations are found.

    The error message contains a summary of all violations.
    """


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------


def validate_rows(rows: list[dict], split: str) -> list[LineageViolation]:
    """Validate every row's ``license_spdx`` and ``source_type`` for *split*.

    Parameters
    ----------
    rows:
        List of manifest row dicts (as stored in ``DatasetManifest.rows``).
        Each dict must have at least ``license_spdx`` and ``source_type`` keys.
    split:
        One of ``"D1"``, ``"D2"``, ``"D3"``.

    Returns
    -------
    list[LineageViolation]
        All violations found.  Empty list means the rows are clean.
        Does **not** raise on violations — the caller decides what to do.

    Raises
    ------
    ValueError
        If *split* is not a known split identifier.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}; valid splits are "
            f"{sorted(_VALID_SPLITS)!r}"
        )

    spdx_allow = _SPDX_ALLOWLISTS[split]
    st_allow = _SOURCE_TYPE_ALLOWLISTS[split]
    violations: list[LineageViolation] = []

    for idx, row in enumerate(rows):
        spdx = row.get("license_spdx", "")
        source_type = row.get("source_type", "")

        if spdx not in spdx_allow:
            violations.append(
                LineageViolation(
                    row_index=idx,
                    field="license_spdx",
                    value=str(spdx),
                    reason=(
                        f"license_spdx {spdx!r} is not allowed for split {split}. "
                        f"Allowed: {sorted(spdx_allow)!r}"
                    ),
                )
            )

        if source_type not in st_allow:
            violations.append(
                LineageViolation(
                    row_index=idx,
                    field="source_type",
                    value=str(source_type),
                    reason=(
                        f"source_type {source_type!r} is not allowed for split {split}. "
                        f"Allowed: {sorted(st_allow)!r}"
                    ),
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def validate_manifest(manifest: "DatasetManifest") -> list[LineageViolation]:
    """Validate a ``DatasetManifest`` instance.

    Reads ``manifest.rows`` and ``manifest.split``.  Returns all violations.

    This is the primary entry point for scrapers that hold a manifest object.
    """
    return validate_rows(manifest.rows, split=manifest.split)


def assert_clean(rows: list[dict], split: str) -> None:
    """Assert that all rows pass lineage validation for *split*.

    Parameters
    ----------
    rows:
        Manifest rows to validate.
    split:
        Target split identifier.

    Raises
    ------
    ValueError
        If *split* is unknown (propagated from ``validate_rows``).
    LineageError
        If any violations are found.  The error message lists all violations.
    """
    violations = validate_rows(rows, split=split)
    if violations:
        detail = "; ".join(
            f"[row {v.row_index}] {v.field}={v.value!r}: {v.reason}"
            for v in violations
        )
        raise LineageError(
            f"{len(violations)} lineage violation(s) in split {split!r}: {detail}"
        )
