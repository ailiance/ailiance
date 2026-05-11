"""D3: 50/50 D1+D2 mixer, stratified by compiler (D2) / hash (D1)."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch.manifest import DatasetManifest


def stratify(files: list[Path], n: int, key_re: str) -> list[Path]:
    """Return up to ``n`` files balanced across the regex capture group."""
    rx = re.compile(key_re)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = rx.search(f.name)
        k = m.group(1) if m else "_"
        buckets[k].append(f)
    keys = sorted(buckets)
    per = max(1, n // len(keys)) if keys else 0
    picked: list[Path] = []
    rng = random.Random(0)
    for k in keys:
        rng.shuffle(buckets[k])
        picked.extend(buckets[k][:per])
    return picked[:n]


def mix(
    d1: Path,
    d2: Path,
    d3: Path,
    n_total: int,
    seed: int,
    manifest_path: Path,
) -> int:
    """Build D3 as ``n_total`` symlinks split 50/50 across D1 and D2."""
    d3.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    d1_files = list(d1.glob("*.kicad_sch"))
    d2_files = stratify(
        list(d2.glob("*.kicad_sch")),
        n=n_total // 2,
        key_re=r"-(skidl|atopile|circuit-synth)-",
    )
    rng.shuffle(d1_files)
    d1_pick = d1_files[: n_total - len(d2_files)]
    manifest = DatasetManifest(manifest_path, split="D3")
    idx = 0
    for src in d1_pick + d2_files:
        link = d3 / f"{idx:06d}.kicad_sch"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        manifest.add(
            source_type="mix",
            source_url=str(src),
            commit_sha="",
            license_spdx="",
            dedup_hash=link.stem,
            file_size_bytes=src.stat().st_size,
            kicad_version_before="10.0.2",
            kicad_version_after="10.0.2",
        )
        idx += 1
    manifest.write()
    return idx


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--d1",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-scraped",
    )
    p.add_argument(
        "--d2",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-synth",
    )
    p.add_argument(
        "--d3",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-mixed",
    )
    p.add_argument("--n-total", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11/d3_manifest.csv",
    )
    a = p.parse_args(argv)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = a.audit_dir / f"d3-{run_stamp}.ndjson"
    log = AuditLogger(audit_path)
    n = mix(a.d1, a.d2, a.d3, a.n_total, a.seed, a.manifest)
    log.log("d3_done", linked=n)
    print(f"D3: {n} symlinks in {a.d3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
