#!/usr/bin/env python3
"""Verify that EU AI Act transparency doc matches the active router checkpoint.

Drift between docs/eu-ai-act-transparency.md and the production router
meta.json is a real compliance risk under EU AI Act Article 13. This
script reads the active router metadata and asserts the transparency
doc section 3.1 mentions the same embedding model and dimension.

Exit codes:
* 0 — aligned
* 1 — drift detected (script prints what diverged)
* 2 — files missing or unparseable (configuration error)

Single source of truth: the meta.json of the active router checkpoint.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_META = REPO_ROOT / "output" / "router-v7-multimodel" / "meta.json"
TRANSPARENCY_DOC = REPO_ROOT / "docs" / "eu-ai-act-transparency.md"
SECTION_RE = re.compile(r"### 3\.1 Current.*?(?=### 3\.2)", re.DOTALL)


def _load_meta() -> dict:
    if not ROUTER_META.exists():
        print(f"FAIL: router meta.json not found at {ROUTER_META}", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(ROUTER_META.read_text())
    except json.JSONDecodeError as exc:
        print(f"FAIL: router meta.json invalid JSON: {exc}", file=sys.stderr)
        sys.exit(2)


def _load_section() -> str:
    if not TRANSPARENCY_DOC.exists():
        print(f"FAIL: transparency doc not found at {TRANSPARENCY_DOC}", file=sys.stderr)
        sys.exit(2)
    doc = TRANSPARENCY_DOC.read_text()
    match = SECTION_RE.search(doc)
    if not match:
        print(
            "FAIL: section 3.1 (Current) not found in transparency doc — "
            "ensure heading format ### 3.1 Current... ### 3.2 is preserved",
            file=sys.stderr,
        )
        sys.exit(2)
    return match.group(0)


def main() -> int:
    meta = _load_meta()
    section = _load_section()

    model = meta.get("embedding_model")
    dim = meta.get("embedding_dim")
    if not model or dim is None:
        print(
            "FAIL: router meta.json missing embedding_model or embedding_dim",
            file=sys.stderr,
        )
        return 2

    # Accept either the full HF id or the bare model name.
    bare = model.rsplit("/", 1)[-1]
    model_ok = model in section or bare in section
    dim_ok = str(dim) in section

    if model_ok and dim_ok:
        print(
            f"OK: transparency doc section 3.1 mentions "
            f"{bare} ({dim}d), matches {ROUTER_META.name}"
        )
        return 0

    print("FAIL: router doc drift detected", file=sys.stderr)
    if not model_ok:
        print(
            f"  meta.json embedding_model={model!r} not found in section 3.1",
            file=sys.stderr,
        )
    if not dim_ok:
        print(
            f"  meta.json embedding_dim={dim} not found in section 3.1",
            file=sys.stderr,
        )
    print(
        "Update docs/eu-ai-act-transparency.md section 3.1 to match "
        "the active router checkpoint, or change ROUTER_META at the top "
        "of this script if the active checkpoint moved.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
