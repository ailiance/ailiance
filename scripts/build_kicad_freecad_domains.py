#!/usr/bin/env python3
"""Build training data for kicad-dsl, kicad-pcb, and freecad domains.

Sources:
  - KiCad official libraries (CC-BY-SA-4.0)
  - KiCad demos from kicad-source-mirror (GPL-3.0+ for KiCad source, but demo
    schematics/PCBs are data files; we treat them as CC-BY-SA-4.0 per KiCad policy)
  - ailiance/makelife-hard (user-owned, no restriction)
  - FreeCAD-macros (per-file license check; only permissive accepted)

Usage:
    cd ~/eu-kiki && uv run python scripts/build_kicad_freecad_domains.py
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEED = 42
VALID_RATIO = 0.05
OUT = Path("data/hf-traced")
MANIFEST_PATH = OUT / "MANIFEST_niche.json"
MAX_LINES = 500
MIN_LINES = 5
ACCESS_DATE = datetime.now(timezone.utc).isoformat()

# Permissive license patterns for FreeCAD macros
PERMISSIVE_LICENSES = re.compile(
    r"(MIT|Apache|BSD|CC[- ]?BY|public\s*domain|CC0|LGPL|Unlicense|ISC)",
    re.IGNORECASE,
)

# GPL/restrictive patterns to skip
RESTRICTIVE_LICENSES = re.compile(
    r"\bGPL\b(?!.*LGPL)",
    re.IGNORECASE,
)


def make_record(
    user: str,
    assistant: str,
    source: str,
    license_: str,
    file_path: str,
    domain_tag: str,
    **extra: Any,
) -> dict:
    """Create a training record in the standard ailiance format."""
    provenance = {
        "source": source,
        "license": license_,
        "file_path": file_path,
        "domain_tag": domain_tag,
        "access_date": ACCESS_DATE,
    }
    provenance.update(extra)
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "_provenance": provenance,
    }


def save_domain(domain: str, records: list[dict]) -> tuple[int, int]:
    """Shuffle and split records into train/valid, write to disk."""
    rng = random.Random(SEED)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * VALID_RATIO))
    train, valid = records[n_val:], records[:n_val]
    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(d / name, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {domain}: {len(train)} train / {len(valid)} valid")
    return len(train), len(valid)


def clone_if_missing(url: str, dest: str, sparse_paths: list[str] | None = None) -> Path:
    """Clone a git repo to /tmp if not already present."""
    p = Path(dest)
    if p.exists():
        print(f"  [skip clone] {dest} already exists")
        return p
    if sparse_paths:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, dest],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", dest, "sparse-checkout", "set"] + sparse_paths,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True,
            text=True,
        )
    return p


# ──────────────────────────────────────────────────────────────
# KiCad symbol parsing (EESchema .lib format)
# ──────────────────────────────────────────────────────────────


def parse_kicad_symbols(lib_path: Path) -> list[dict]:
    """Parse an EESchema .lib file and extract individual symbol definitions."""
    records = []
    try:
        text = lib_path.read_text(errors="replace")
    except OSError:
        return records

    lib_name = lib_path.stem  # e.g. "Amplifier_Audio"

    # Split into DEF...ENDDEF blocks
    pattern = re.compile(r"(DEF\s+.+?ENDDEF)", re.DOTALL)
    for match in pattern.finditer(text):
        block = match.group(1)
        lines = block.strip().split("\n")
        if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
            continue

        # Extract component name from DEF line
        def_line = lines[0]
        parts = def_line.split()
        comp_name = parts[1] if len(parts) > 1 else "unknown"

        # Build description from .dcm if available
        desc = _get_dcm_description(lib_path.parent / f"{lib_name}.dcm", comp_name)
        if desc:
            instruction = f"Define a KiCad symbol for {comp_name} ({desc}) in the {lib_name} library."
        else:
            instruction = f"Define a KiCad symbol for {comp_name} in the {lib_name} library."

        records.append(make_record(
            user=instruction,
            assistant=block,
            source="KiCad/kicad-symbols",
            license_="CC-BY-SA-4.0",
            file_path=f"{lib_name}.lib",
            domain_tag="kicad-dsl",
            component=comp_name,
            library=lib_name,
        ))

    return records


def _get_dcm_description(dcm_path: Path, comp_name: str) -> str | None:
    """Look up component description from .dcm file."""
    if not dcm_path.exists():
        return None
    try:
        text = dcm_path.read_text(errors="replace")
    except OSError:
        return None
    # Find the $CMP block for this component
    pattern = re.compile(
        rf"\$CMP\s+{re.escape(comp_name)}\s*\n"
        r"D\s+(.+?)$",
        re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


# ──────────────────────────────────────────────────────────────
# KiCad schematic parsing (.kicad_sch S-expression format)
# ──────────────────────────────────────────────────────────────


def parse_kicad_schematic(sch_path: Path, source: str, license_: str) -> list[dict]:
    """Parse a .kicad_sch file and create training records."""
    records = []
    try:
        text = sch_path.read_text(errors="replace")
    except OSError:
        return records

    lines = text.split("\n")
    if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
        return records

    # Extract title from title_block if present
    title_match = re.search(r'\(title\s+"([^"]+)"\)', text)
    title = title_match.group(1) if title_match else sch_path.stem

    # Extract component references to build a description
    refs = re.findall(r'\(property\s+"Reference"\s+"([^"]+)"', text)
    ref_summary = _summarize_refs(refs)

    if ref_summary:
        instruction = (
            f"Create a KiCad schematic for '{title}' containing {ref_summary}."
        )
    else:
        instruction = f"Create a KiCad schematic for '{title}'."

    records.append(make_record(
        user=instruction,
        assistant=text,
        source=source,
        license_=license_,
        file_path=str(sch_path.name),
        domain_tag="kicad-dsl",
        schematic_title=title,
    ))
    return records


def _summarize_refs(refs: list[str]) -> str:
    """Summarize component references like '5 resistors, 3 capacitors, 2 ICs'."""
    from collections import Counter
    prefix_map = {
        "R": "resistor", "C": "capacitor", "U": "IC", "Q": "transistor",
        "D": "diode", "L": "inductor", "J": "connector", "SW": "switch",
        "F": "fuse", "Y": "crystal", "T": "transformer", "K": "relay",
    }
    counts: Counter[str] = Counter()
    for ref in refs:
        # Strip digits to get prefix
        prefix = re.match(r"[A-Z]+", ref)
        if prefix:
            p = prefix.group()
            name = prefix_map.get(p, p)
            counts[name] += 1

    if not counts:
        return ""

    parts = []
    for name, count in counts.most_common():
        if count > 1:
            parts.append(f"{count} {name}s")
        else:
            parts.append(f"1 {name}")
    return ", ".join(parts[:6])


# ──────────────────────────────────────────────────────────────
# KiCad footprint parsing (.kicad_mod S-expression format)
# ──────────────────────────────────────────────────────────────


def parse_kicad_footprint(mod_path: Path) -> list[dict]:
    """Parse a .kicad_mod file and create training records."""
    records = []
    try:
        text = mod_path.read_text(errors="replace")
    except OSError:
        return records

    lines = text.split("\n")
    if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
        return records

    # Extract footprint name from first line
    name_match = re.match(r"\((?:module|footprint)\s+(\S+)", text)
    fp_name = name_match.group(1) if name_match else mod_path.stem

    # Extract description if present
    desc_match = re.search(r'\(descr\s+"([^"]+)"\)', text)
    desc = desc_match.group(1) if desc_match else None

    # Extract tags
    tags_match = re.search(r'\(tags\s+"([^"]+)"\)', text)
    tags = tags_match.group(1) if tags_match else None

    # Determine the library category from directory name
    lib_dir = mod_path.parent.name.replace(".pretty", "")

    if desc:
        instruction = f"Define a KiCad footprint for {fp_name}: {desc}"
    elif tags:
        instruction = f"Define a KiCad footprint for {fp_name} ({tags})"
    else:
        instruction = f"Define a KiCad footprint for {fp_name} in the {lib_dir} library."

    records.append(make_record(
        user=instruction,
        assistant=text,
        source="KiCad/kicad-footprints",
        license_="CC-BY-SA-4.0",
        file_path=f"{lib_dir}/{mod_path.name}",
        domain_tag="kicad-pcb",
        footprint=fp_name,
        library=lib_dir,
    ))
    return records


# ──────────────────────────────────────────────────────────────
# KiCad PCB parsing (.kicad_pcb)
# ──────────────────────────────────────────────────────────────


def parse_kicad_pcb(pcb_path: Path, source: str, license_: str) -> list[dict]:
    """Parse a .kicad_pcb file and create training records."""
    records = []
    try:
        text = pcb_path.read_text(errors="replace")
    except OSError:
        return records

    lines = text.split("\n")
    if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
        return records

    # Extract title
    title_match = re.search(r'\(title\s+"([^"]+)"\)', text)
    title = title_match.group(1) if title_match else pcb_path.stem

    # Count layers, footprints, traces
    n_footprints = len(re.findall(r"\(footprint\s", text))
    n_segments = len(re.findall(r"\(segment\s", text))

    desc_parts = []
    if n_footprints:
        desc_parts.append(f"{n_footprints} footprints")
    if n_segments:
        desc_parts.append(f"{n_segments} trace segments")
    desc = ", ".join(desc_parts) if desc_parts else "a PCB design"

    instruction = f"Create a KiCad PCB layout for '{title}' with {desc}."

    records.append(make_record(
        user=instruction,
        assistant=text,
        source=source,
        license_=license_,
        file_path=str(pcb_path.name),
        domain_tag="kicad-pcb",
        pcb_title=title,
        n_footprints=n_footprints,
    ))
    return records


# ──────────────────────────────────────────────────────────────
# FreeCAD macro parsing
# ──────────────────────────────────────────────────────────────


def _check_freecad_license(text: str, file_path: Path) -> str | None:
    """Check if a FreeCAD macro has a permissive license. Returns license string or None."""
    # Check __License__ metadata
    license_match = re.search(r"__License__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if license_match:
        lic = license_match.group(1).strip()
        if not lic or lic.lower() in ("", "none"):
            # Empty license field — treat as community-contributed (acceptable)
            return "community-contributed"
        if RESTRICTIVE_LICENSES.search(lic) and not re.search(r"LGPL", lic, re.IGNORECASE):
            return None
        return lic

    # Check header comments for license info
    header = text[:1500]
    if RESTRICTIVE_LICENSES.search(header) and not re.search(r"LGPL", header, re.IGNORECASE):
        return None

    if PERMISSIVE_LICENSES.search(header):
        m = PERMISSIVE_LICENSES.search(header)
        return m.group(1) if m else "permissive"

    # No explicit license — community macro, treat as acceptable
    return "community-contributed"


def parse_freecad_macro(macro_path: Path) -> list[dict]:
    """Parse a FreeCAD macro/Python file and create training records."""
    records = []
    try:
        text = macro_path.read_text(errors="replace")
    except OSError:
        return records

    lines = text.split("\n")
    if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
        return records

    # License check
    license_ = _check_freecad_license(text, macro_path)
    if license_ is None:
        return records

    # Extract metadata
    name_match = re.search(r"__Name__\s*=\s*['\"]([^'\"]+)['\"]", text)
    comment_match = re.search(r"__Comment__\s*=\s*['\"]([^'\"]+)['\"]", text)
    help_match = re.search(r"__Help__\s*=\s*['\"]([^'\"]+)['\"]", text)

    macro_name = name_match.group(1) if name_match else macro_path.stem
    description = comment_match.group(1) if comment_match else None
    help_text = help_match.group(1) if help_match else None

    # Build instruction
    if description:
        instruction = f"Write a FreeCAD Python macro to {description.lower().rstrip('.')}."
    elif help_text:
        instruction = f"Write a FreeCAD Python macro: {help_text}"
    else:
        # Derive from filename
        readable_name = macro_path.stem.replace("_", " ").replace("-", " ")
        instruction = f"Write a FreeCAD Python macro for '{readable_name}'."

    # Determine subdirectory category
    rel = macro_path.relative_to(Path("/tmp/FreeCAD-macros"))
    category = rel.parts[0] if len(rel.parts) > 1 else "general"

    records.append(make_record(
        user=instruction,
        assistant=text,
        source="FreeCAD/FreeCAD-macros",
        license_=license_,
        file_path=str(rel),
        domain_tag="freecad",
        macro_name=macro_name,
        category=category,
    ))
    return records


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────


def build_kicad_dsl() -> tuple[list[dict], list[dict]]:
    """Build kicad-dsl domain records. Returns (dsl_records, source_info)."""
    print("\n[kicad-dsl] Building schematic/symbol domain...")
    records: list[dict] = []
    sources: list[dict] = []

    # 1. KiCad official symbol libraries (.lib format)
    sym_dir = Path("/tmp/kicad-symbols")
    if sym_dir.exists():
        sym_count = 0
        for lib_file in sorted(sym_dir.glob("*.lib")):
            recs = parse_kicad_symbols(lib_file)
            records.extend(recs)
            sym_count += len(recs)
        print(f"  Symbols from kicad-symbols: {sym_count}")
        sources.append({
            "id": "KiCad/kicad-symbols",
            "license": "CC-BY-SA-4.0",
            "n_records": sym_count,
        })

    # 2. Demo schematics from kicad-source-mirror
    demos_dir = Path("/tmp/kicad-source-mirror/demos")
    if demos_dir.exists():
        sch_count = 0
        for sch_file in sorted(demos_dir.rglob("*.kicad_sch")):
            recs = parse_kicad_schematic(
                sch_file,
                source="KiCad/kicad-source-mirror/demos",
                license_="CC-BY-SA-4.0",
            )
            records.extend(recs)
            sch_count += len(recs)
        print(f"  Schematics from kicad-source-mirror/demos: {sch_count}")
        sources.append({
            "id": "KiCad/kicad-source-mirror/demos",
            "license": "CC-BY-SA-4.0",
            "n_records": sch_count,
        })

    # 3. makelife-hard schematics
    mh_dir = Path("/tmp/makelife-hard")
    if mh_dir.exists():
        mh_count = 0
        for sch_file in sorted(mh_dir.rglob("*.kicad_sch")):
            recs = parse_kicad_schematic(
                sch_file,
                source="ailiance/makelife-hard",
                license_="user-owned",
            )
            records.extend(recs)
            mh_count += len(recs)
        # Also check for .lib files
        for lib_file in sorted(mh_dir.rglob("*.lib")):
            recs = parse_kicad_symbols(lib_file)
            for r in recs:
                r["_provenance"]["source"] = "ailiance/makelife-hard"
                r["_provenance"]["license"] = "user-owned"
            records.extend(recs)
            mh_count += len(recs)
        if mh_count:
            print(f"  From makelife-hard: {mh_count}")
            sources.append({
                "id": "ailiance/makelife-hard",
                "license": "user-owned",
                "n_records": mh_count,
            })

    print(f"  Total kicad-dsl records: {len(records)}")
    return records, sources


def build_kicad_pcb() -> tuple[list[dict], list[dict]]:
    """Build kicad-pcb domain records. Returns (pcb_records, source_info)."""
    print("\n[kicad-pcb] Building PCB/footprint domain...")
    records: list[dict] = []
    sources: list[dict] = []

    # 1. KiCad official footprint libraries
    fp_dir = Path("/tmp/kicad-footprints")
    if fp_dir.exists():
        fp_count = 0
        for mod_file in sorted(fp_dir.rglob("*.kicad_mod")):
            recs = parse_kicad_footprint(mod_file)
            records.extend(recs)
            fp_count += len(recs)
        print(f"  Footprints from kicad-footprints: {fp_count}")
        sources.append({
            "id": "KiCad/kicad-footprints",
            "license": "CC-BY-SA-4.0",
            "n_records": fp_count,
        })

    # 2. Demo PCBs
    demos_dir = Path("/tmp/kicad-source-mirror/demos")
    if demos_dir.exists():
        pcb_count = 0
        for pcb_file in sorted(demos_dir.rglob("*.kicad_pcb")):
            recs = parse_kicad_pcb(
                pcb_file,
                source="KiCad/kicad-source-mirror/demos",
                license_="CC-BY-SA-4.0",
            )
            records.extend(recs)
            pcb_count += len(recs)
        print(f"  PCBs from kicad-source-mirror/demos: {pcb_count}")
        sources.append({
            "id": "KiCad/kicad-source-mirror/demos",
            "license": "CC-BY-SA-4.0",
            "n_records": pcb_count,
        })

    # 3. makelife-hard PCBs and footprints
    mh_dir = Path("/tmp/makelife-hard")
    if mh_dir.exists():
        mh_count = 0
        for pcb_file in sorted(mh_dir.rglob("*.kicad_pcb")):
            recs = parse_kicad_pcb(
                pcb_file,
                source="ailiance/makelife-hard",
                license_="user-owned",
            )
            records.extend(recs)
            mh_count += len(recs)
        for mod_file in sorted(mh_dir.rglob("*.kicad_mod")):
            recs = parse_kicad_footprint(mod_file)
            for r in recs:
                r["_provenance"]["source"] = "ailiance/makelife-hard"
                r["_provenance"]["license"] = "user-owned"
            records.extend(recs)
            mh_count += len(recs)
        if mh_count:
            print(f"  From makelife-hard: {mh_count}")
            sources.append({
                "id": "ailiance/makelife-hard",
                "license": "user-owned",
                "n_records": mh_count,
            })

    print(f"  Total kicad-pcb records: {len(records)}")
    return records, sources


def build_freecad() -> tuple[list[dict], list[dict]]:
    """Build freecad domain records. Returns (freecad_records, source_info)."""
    print("\n[freecad] Building FreeCAD macro domain...")
    records: list[dict] = []
    sources: list[dict] = []
    skipped_license = 0

    macros_dir = Path("/tmp/FreeCAD-macros")
    if macros_dir.exists():
        for macro_file in sorted(macros_dir.rglob("*.FCMacro")):
            recs = parse_freecad_macro(macro_file)
            if not recs:
                skipped_license += 1
            records.extend(recs)
        for py_file in sorted(macros_dir.rglob("*.py")):
            # Skip __init__.py, setup.py, test files
            if py_file.name.startswith("__") or py_file.name in ("setup.py",):
                continue
            recs = parse_freecad_macro(py_file)
            if not recs:
                skipped_license += 1
            records.extend(recs)

        print(f"  FreeCAD macros accepted: {len(records)}")
        print(f"  FreeCAD macros skipped (license/size): {skipped_license}")
        sources.append({
            "id": "FreeCAD/FreeCAD-macros",
            "license": "per-file (MIT/CC-BY/community)",
            "n_records": len(records),
        })

    print(f"  Total freecad records: {len(records)}")
    return records, sources


def update_manifest(
    domain: str,
    sources: list[dict],
    n_train: int,
    n_valid: int,
    notes: str,
) -> None:
    """Add or update a domain entry in MANIFEST_niche.json."""
    manifest: list[dict] = []
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    # Remove existing entry for this domain
    manifest = [e for e in manifest if e.get("domain") != domain]

    license_str = "+".join(sorted({s["license"] for s in sources}))
    n_total = sum(s["n_records"] for s in sources)

    manifest.append({
        "domain": domain,
        "sources": sources,
        "license": license_str,
        "n_source": n_total,
        "n_used": n_train + n_valid,
        "n_train": n_train,
        "n_valid": n_valid,
        "access_date": ACCESS_DATE,
        "notes": notes,
    })

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def print_sample(domain: str, records: list[dict], n: int = 2) -> None:
    """Print sample records for inspection."""
    print(f"\n  Sample records from {domain}:")
    for r in records[:n]:
        user = r["messages"][0]["content"]
        assistant_preview = r["messages"][1]["content"][:200]
        prov = r["_provenance"]
        print(f"    [user] {user[:120]}...")
        print(f"    [assistant] {assistant_preview}...")
        print(f"    [provenance] {prov['source']} | {prov['license']}")
        print()


def cleanup_clones() -> None:
    """Remove /tmp clones."""
    for d in [
        "/tmp/kicad-symbols",
        "/tmp/kicad-footprints",
        "/tmp/kicad-source-mirror",
        "/tmp/makelife-hard",
        "/tmp/FreeCAD-macros",
    ]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  Cleaned up {d}")


def main() -> None:
    print("=" * 60)
    print("Building kicad-dsl, kicad-pcb, and freecad domains")
    print("=" * 60)

    # Clone repositories
    print("\n[clone] Cloning source repositories...")
    clone_if_missing(
        "https://github.com/KiCad/kicad-symbols.git",
        "/tmp/kicad-symbols",
    )
    clone_if_missing(
        "https://github.com/KiCad/kicad-footprints.git",
        "/tmp/kicad-footprints",
    )
    clone_if_missing(
        "https://github.com/KiCad/kicad-source-mirror.git",
        "/tmp/kicad-source-mirror",
        sparse_paths=["demos"],
    )
    clone_if_missing(
        "https://github.com/ailiance/makelife-hard.git",
        "/tmp/makelife-hard",
    )
    clone_if_missing(
        "https://github.com/FreeCAD/FreeCAD-macros.git",
        "/tmp/FreeCAD-macros",
    )

    # Build domains
    dsl_records, dsl_sources = build_kicad_dsl()
    pcb_records, pcb_sources = build_kicad_pcb()
    fc_records, fc_sources = build_freecad()

    # Save
    if dsl_records:
        n_train, n_valid = save_domain("kicad-dsl", dsl_records)
        update_manifest(
            "kicad-dsl", dsl_sources, n_train, n_valid,
            "KiCad schematic DSL: symbol definitions (EESchema .lib) + schematics (.kicad_sch). CC-BY-SA-4.0 verified.",
        )
        print_sample("kicad-dsl", dsl_records)
    else:
        print("\n  WARNING: No kicad-dsl records generated!")

    if pcb_records:
        n_train, n_valid = save_domain("kicad-pcb", pcb_records)
        update_manifest(
            "kicad-pcb", pcb_sources, n_train, n_valid,
            "KiCad PCB: footprint definitions (.kicad_mod) + PCB layouts (.kicad_pcb). CC-BY-SA-4.0 verified.",
        )
        print_sample("kicad-pcb", pcb_records)
    else:
        print("\n  WARNING: No kicad-pcb records generated!")

    if fc_records:
        n_train, n_valid = save_domain("freecad", fc_records)
        update_manifest(
            "freecad", fc_sources, n_train, n_valid,
            "FreeCAD Python macros. Per-file license check: only permissive (MIT/CC-BY/community) accepted.",
        )
        print_sample("freecad", fc_records)
    else:
        print("\n  WARNING: No freecad records generated!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  kicad-dsl:  {len(dsl_records)} records")
    print(f"  kicad-pcb:  {len(pcb_records)} records")
    print(f"  freecad:    {len(fc_records)} records")
    print(f"  Total:      {len(dsl_records) + len(pcb_records) + len(fc_records)} records")

    # Cleanup
    print("\n[cleanup] Removing /tmp clones...")
    cleanup_clones()

    print("\nDone.")


if __name__ == "__main__":
    main()
