#!/usr/bin/env python3
"""Rebuild cpp domain with clean, domain-pure, EU AI Act compliant data.

Sources (priority order, all verified SPDX):
  A) iamtarun/code_instructions_120k_alpaca  (Apache-2.0)
  B) sahil2801/CodeAlpaca-20k                (CC-BY-4.0)
  C) ise-uiuc/Magicoder-OSS-Instruct-75K    (MIT)

Filtering: at least 2 C++ markers per record, exclude Java/Python contamination.

Usage:
    cd ~/ailiance && uv run python scripts/rebuild_cpp.py
"""
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

SEED = 42
TARGET = 3000
VALID_RATIO = 0.05
OUT = Path("data/hf-traced")
MANIFEST_PATH = OUT / "MANIFEST_niche.json"

# C++ positive markers — need at least 2 to qualify
CPP_MARKERS: list[str] = [
    "#include",
    "std::",
    "cout",
    "cin",
    "int main",
    "nullptr",
    "template<",
    "class ",
    "namespace",
    "vector<",
]

# Exclusion markers for Java contamination
JAVA_MARKERS: list[str] = [
    "public static void",
    "System.out",
    "import java",
    "public class ",
    "String[]",
]

# Exclusion markers for Python contamination
PYTHON_MARKERS: list[str] = [
    "def ",
    "import ",
    "print(",
]

# Exclusion markers for C#/.NET/UWP contamination
CSHARP_MARKERS: list[str] = [
    "using System",
    "using Windows",
    "public sealed",
    "IActionResult",
    "Console.WriteLine",
    "namespace ",  # C# namespace declarations (also valid C++ but combined with others)
]


def count_cpp_markers(text: str) -> tuple[int, list[str]]:
    """Count distinct C++ markers found in text."""
    found = [m for m in CPP_MARKERS if m in text]
    return len(found), found


def is_java_contaminated(text: str) -> bool:
    """Check if text is primarily Java."""
    return sum(1 for m in JAVA_MARKERS if m in text) >= 2


def is_python_contaminated(text: str) -> bool:
    """Check if text is primarily Python without C++ context."""
    py_count = sum(1 for m in PYTHON_MARKERS if m in text)
    cpp_count, _ = count_cpp_markers(text)
    # Python contamination: has Python markers but few/no C++ markers
    return py_count >= 2 and cpp_count < 2


def is_csharp_contaminated(text: str) -> bool:
    """Check if text is primarily C#/.NET.

    C# code uses 'using' directives and lacks '#include'.
    If we see C# markers and NO #include, it's almost certainly C#.
    """
    cs_count = sum(1 for m in CSHARP_MARKERS if m in text)
    has_include = "#include" in text
    # Strong C# signal without C++ includes
    if cs_count >= 2 and not has_include:
        return True
    # Very strong C# signal overrides even #include presence
    if cs_count >= 3:
        return True
    return False


def make_msg(user: str, assistant: str, provenance: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "_provenance": provenance,
    }


def load_source_a() -> tuple[list[dict], dict]:
    """Load iamtarun/code_instructions_120k_alpaca, filter for C++."""
    source_id = "iamtarun/code_instructions_120k_alpaca"
    license_ = "Apache-2.0"
    print(f"\n{'='*60}")
    print(f"Source A: {source_id} ({license_})")
    print(f"{'='*60}")

    ds = load_dataset(source_id, split="train")
    print(f"  Raw records: {len(ds)}")

    records: list[dict] = []
    stats = {"total": len(ds), "cpp_pass": 0, "java_reject": 0, "python_reject": 0, "csharp_reject": 0}

    for idx, row in enumerate(ds):
        instruction = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        output = row.get("output", "") or ""

        # Combine instruction + input for user prompt
        user_text = instruction.strip()
        if inp.strip():
            user_text = f"{user_text}\n\n{inp.strip()}"

        if not user_text or not output.strip():
            continue

        combined = f"{user_text}\n{output}"

        # Exclude Java
        if is_java_contaminated(combined):
            stats["java_reject"] += 1
            continue

        # Exclude C#/.NET
        if is_csharp_contaminated(combined):
            stats["csharp_reject"] += 1
            continue

        # Exclude Python without C++ context
        if is_python_contaminated(combined):
            stats["python_reject"] += 1
            continue

        # Require at least 2 C++ markers
        cpp_count, _ = count_cpp_markers(combined)
        if cpp_count < 2:
            continue

        stats["cpp_pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
        }
        records.append(make_msg(user_text, output, provenance))

    print(f"  C++ pass (>=2 markers): {stats['cpp_pass']}")
    print(f"  Java rejected: {stats['java_reject']}")
    print(f"  C# rejected: {stats['csharp_reject']}")
    print(f"  Python rejected: {stats['python_reject']}")
    return records, stats


def load_source_b() -> tuple[list[dict], dict]:
    """Load sahil2801/CodeAlpaca-20k, filter for C++."""
    source_id = "sahil2801/CodeAlpaca-20k"
    license_ = "CC-BY-4.0"
    print(f"\n{'='*60}")
    print(f"Source B: {source_id} ({license_})")
    print(f"{'='*60}")

    ds = load_dataset(source_id, split="train")
    print(f"  Raw records: {len(ds)}")

    records: list[dict] = []
    stats = {"total": len(ds), "cpp_pass": 0, "java_reject": 0, "python_reject": 0, "csharp_reject": 0}

    for idx, row in enumerate(ds):
        instruction = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        output = row.get("output", "") or ""

        user_text = instruction.strip()
        if inp.strip():
            user_text = f"{user_text}\n\n{inp.strip()}"

        if not user_text or not output.strip():
            continue

        combined = f"{user_text}\n{output}"

        if is_java_contaminated(combined):
            stats["java_reject"] += 1
            continue

        if is_csharp_contaminated(combined):
            stats["csharp_reject"] += 1
            continue

        if is_python_contaminated(combined):
            stats["python_reject"] += 1
            continue

        cpp_count, _ = count_cpp_markers(combined)
        if cpp_count < 2:
            continue

        stats["cpp_pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
        }
        records.append(make_msg(user_text, output, provenance))

    print(f"  C++ pass (>=2 markers): {stats['cpp_pass']}")
    print(f"  Java rejected: {stats['java_reject']}")
    print(f"  C# rejected: {stats['csharp_reject']}")
    print(f"  Python rejected: {stats['python_reject']}")
    return records, stats


def load_source_c() -> tuple[list[dict], dict]:
    """Load ise-uiuc/Magicoder-OSS-Instruct-75K, filter for C++."""
    source_id = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    license_ = "MIT"
    print(f"\n{'='*60}")
    print(f"Source C: {source_id} ({license_})")
    print(f"{'='*60}")

    ds = load_dataset(source_id, split="train")
    print(f"  Raw records: {len(ds)}")

    records: list[dict] = []
    stats = {"total": len(ds), "cpp_pass": 0, "java_reject": 0, "python_reject": 0, "csharp_reject": 0}

    for idx, row in enumerate(ds):
        # Magicoder uses 'problem' and 'solution' columns
        instruction = (
            row.get("problem", "")
            or row.get("instruction", "")
            or row.get("input", "")
            or ""
        )
        output = (
            row.get("solution", "")
            or row.get("output", "")
            or row.get("response", "")
            or ""
        )

        if not instruction.strip() or not output.strip():
            continue

        combined = f"{instruction}\n{output}"

        if is_java_contaminated(combined):
            stats["java_reject"] += 1
            continue

        if is_csharp_contaminated(combined):
            stats["csharp_reject"] += 1
            continue

        if is_python_contaminated(combined):
            stats["python_reject"] += 1
            continue

        cpp_count, _ = count_cpp_markers(combined)
        if cpp_count < 2:
            continue

        stats["cpp_pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
        }
        records.append(make_msg(instruction.strip(), output, provenance))

    print(f"  C++ pass (>=2 markers): {stats['cpp_pass']}")
    print(f"  Java rejected: {stats['java_reject']}")
    print(f"  C# rejected: {stats['csharp_reject']}")
    print(f"  Python rejected: {stats['python_reject']}")
    return records, stats


def save_split(records: list[dict], domain: str = "cpp") -> tuple[int, int]:
    """Shuffle, split, and write train/valid JSONL."""
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

    print(f"\n  → {domain}: {len(train)} train / {len(valid)} valid")
    return len(train), len(valid)


def update_manifest(
    hf_id: str,
    license_: str,
    n_source: int,
    n_used: int,
    n_train: int,
    n_valid: int,
    notes: str,
) -> None:
    """Update the cpp entry in MANIFEST_niche.json."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Remove existing cpp entry
    manifest = [e for e in manifest if e.get("domain") != "cpp"]

    manifest.append(
        {
            "domain": "cpp",
            "hf_id": hf_id,
            "license": license_,
            "n_source": n_source,
            "n_used": n_used,
            "n_train": n_train,
            "n_valid": n_valid,
            "access_date": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
    )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  → MANIFEST_niche.json updated (cpp entry)")


def print_samples(records: list[dict], n: int = 5) -> None:
    """Print sample records for quality validation."""
    print(f"\n{'='*60}")
    print(f"Sample records ({n}):")
    print(f"{'='*60}")
    for i, rec in enumerate(records[:n]):
        user = rec["messages"][0]["content"]
        assistant = rec["messages"][1]["content"]
        print(f"\n--- Sample {i+1} ---")
        print(f"USER: {user[:200]}")
        print(f"ASSISTANT: {assistant[:400]}")
        print(f"PROVENANCE: {rec['_provenance']}")


def print_marker_stats(records: list[dict]) -> None:
    """Print C++ marker distribution across all records."""
    marker_counts: Counter = Counter()
    for rec in records:
        combined = (
            rec["messages"][0]["content"] + "\n" + rec["messages"][1]["content"]
        )
        _, found = count_cpp_markers(combined)
        for m in found:
            marker_counts[m] += 1

    print(f"\n{'='*60}")
    print("C++ Marker Distribution:")
    print(f"{'='*60}")
    for marker, count in marker_counts.most_common():
        pct = count / len(records) * 100
        print(f"  {marker:20s}: {count:5d} ({pct:5.1f}%)")


def main() -> None:
    all_records: list[dict] = []
    sources_used: list[str] = []
    all_stats: list[dict] = []

    # --- Source A ---
    records_a, stats_a = load_source_a()
    all_records.extend(records_a)
    sources_used.append("iamtarun/code_instructions_120k_alpaca")
    all_stats.append({"source": "A", **stats_a})

    if len(all_records) < TARGET:
        # --- Source B ---
        records_b, stats_b = load_source_b()
        all_records.extend(records_b)
        sources_used.append("sahil2801/CodeAlpaca-20k")
        all_stats.append({"source": "B", **stats_b})

    if len(all_records) < TARGET:
        # --- Source C ---
        records_c, stats_c = load_source_c()
        all_records.extend(records_c)
        sources_used.append("ise-uiuc/Magicoder-OSS-Instruct-75K")
        all_stats.append({"source": "C", **stats_c})

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Aggregation summary:")
    print(f"{'='*60}")
    print(f"  Total collected: {len(all_records)}")
    print(f"  Target: {TARGET}")

    if len(all_records) < TARGET:
        print(f"  WARNING: Only {len(all_records)} records, below target {TARGET}")
        print(f"  Using all available records")
    else:
        # Cap to TARGET
        all_records = random.Random(SEED).sample(all_records, TARGET)
        print(f"  Capped to {TARGET}")

    n_used = len(all_records)

    # Save
    n_train, n_valid = save_split(all_records)

    # Update manifest
    hf_id = "+".join(sources_used)
    licenses = "Apache-2.0+CC-BY-4.0+MIT" if len(sources_used) == 3 else (
        "Apache-2.0+CC-BY-4.0" if len(sources_used) == 2 else "Apache-2.0"
    )
    update_manifest(
        hf_id=hf_id,
        license_=licenses,
        n_source=sum(s["total"] for s in all_stats),
        n_used=n_used,
        n_train=n_train,
        n_valid=n_valid,
        notes=f"C++ filtered (>=2 markers, Java/Python excluded) from {len(sources_used)} sources",
    )

    # Validate samples
    print_samples(all_records)

    # Marker stats
    print_marker_stats(all_records)

    # Final report
    print(f"\n{'='*60}")
    print("FINAL REPORT:")
    print(f"{'='*60}")
    for s in all_stats:
        print(
            f"  Source {s['source']}: {s['total']} raw → "
            f"{s['cpp_pass']} C++ pass, "
            f"{s['java_reject']} Java rejected, "
            f"{s.get('csharp_reject', 0)} C# rejected, "
            f"{s['python_reject']} Python rejected"
        )
    print(f"  Combined: {n_used} used → {n_train} train / {n_valid} valid")
    print(f"  Sources: {hf_id}")
    print(f"  Licenses: {licenses}")


if __name__ == "__main__":
    main()
