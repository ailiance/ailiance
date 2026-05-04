#!/usr/bin/env python3
"""
Curriculum Learning: Split long sequences and sort JSONL training data short -> long.

For each record:
  - If tokens <= max_seq: keep as-is
  - If tokens <= max_seq * 2: keep as-is (will use higher max_seq setting)
  - If tokens > max_seq * 2: split by C/C++ logical boundaries

After splitting, all records (including splits) are sorted by length ascending
for curriculum learning (easy -> hard).

Usage:
    uv run python scripts/prepare_curriculum.py --domains cpp --max-seq 8192
    uv run python scripts/prepare_curriculum.py --domains cpp --max-seq 8192 --stats-only
    uv run python scripts/prepare_curriculum.py --domains emc-dsp-power,security-fenrir --max-seq 4096
    uv run python scripts/prepare_curriculum.py --all --max-seq 4096
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
from pathlib import Path

HF_DATA = Path(__file__).resolve().parent.parent / "data" / "hf-traced"

THRESHOLDS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Regex for C/C++ function boundaries
FUNC_BOUNDARY_RE = re.compile(
    r"\n(?=(?:void|int|static|bool|uint\w+_t|char|float|double|struct|enum|class|"
    r"unsigned|signed|long|short|size_t|ssize_t|const|volatile|extern|inline|"
    r"HAL_StatusTypeDef|FRESULT|BaseType_t|TaskHandle_t|esp_err_t|"
    r"__STATIC_INLINE|__attribute__)\s+\w+\s*\()"
)

OVERLAP_LINES = 10


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (chars / 3.5)."""
    return int(len(text) / 3.5)


def concat_messages(record: dict) -> str:
    """Concatenate all message contents."""
    return "".join(msg.get("content", "") for msg in record.get("messages", []))


def estimate_record_tokens(record: dict) -> int:
    """Estimate token count for an entire record."""
    return estimate_tokens(concat_messages(record))


def get_user_prompt(record: dict) -> str:
    """Extract the user (instruction) message from a record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_assistant_content(record: dict) -> str:
    """Extract the assistant (code) response from a record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def split_by_functions(code: str) -> list[str]:
    """Split code at C/C++ function boundaries."""
    parts = FUNC_BOUNDARY_RE.split(code)
    # Filter out empty parts
    return [p for p in parts if p.strip()]


def split_by_blocks(code: str) -> list[str]:
    """Split code at double-newline boundaries."""
    parts = re.split(r"\n\n+", code)
    return [p for p in parts if p.strip()]


def merge_chunks_to_target(
    parts: list[str], target_chars: int
) -> list[str]:
    """Merge small parts into chunks targeting a character budget."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part)
        if current and (current_len + part_len) > target_chars:
            chunks.append("\n".join(current))
            current = [part]
            current_len = part_len
        else:
            current.append(part)
            current_len += part_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def add_overlap(chunks: list[str]) -> list[str]:
    """Add last N lines of previous chunk to start of next chunk as context."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_lines = chunks[i - 1].split("\n")
        overlap = "\n".join(prev_lines[-OVERLAP_LINES:])
        result.append(f"// ... (continued from previous part)\n{overlap}\n\n{chunks[i]}")

    return result


def split_record(
    record: dict, max_seq: int
) -> list[dict]:
    """
    Split a record that exceeds max_seq * 2 tokens into multiple records.

    Each split chunk keeps:
    - The original user prompt with "[Part N/M]" prefix
    - A chunk of code as assistant response
    - Overlap context from previous chunk
    - Provenance with split metadata
    """
    user_prompt = get_user_prompt(record)
    code = get_assistant_content(record)

    # Target chars for each chunk (max_seq * 0.9 tokens * 3.5 chars/token,
    # minus the user prompt)
    prompt_chars = len(user_prompt)
    target_chars = int(max_seq * 0.9 * 3.5) - prompt_chars
    target_chars = max(target_chars, 1000)  # minimum safety

    # Step 1: Try splitting on function boundaries
    func_parts = split_by_functions(code)

    if len(func_parts) > 1:
        chunks = merge_chunks_to_target(func_parts, target_chars)
    else:
        # Step 2: Fall back to double-newline block boundaries
        block_parts = split_by_blocks(code)
        if len(block_parts) > 1:
            chunks = merge_chunks_to_target(block_parts, target_chars)
        else:
            chunks = [code]

    # Step 3: Re-split any chunks still exceeding target
    # Handles giant array literals, binary data, long lines, etc.
    refined: list[str] = []
    for chunk in chunks:
        if len(chunk) <= target_chars * 1.5:
            refined.append(chunk)
            continue
        # Try line-based splitting first
        lines = chunk.split("\n")
        if len(lines) > 10:
            target_lines = max(target_chars // max(len(chunk) // len(lines), 80), 5)
            for i in range(0, len(lines), target_lines):
                sub = "\n".join(lines[i : i + target_lines])
                if sub.strip():
                    refined.append(sub)
        else:
            # Very few lines but huge content (e.g., hex arrays on single lines)
            # Split by comma boundaries within the text
            comma_parts = chunk.split(", ")
            if len(comma_parts) > 10:
                parts_per_chunk = max(target_chars // 10, 100)  # ~10 chars per "0xNN, "
                for i in range(0, len(comma_parts), parts_per_chunk):
                    sub = ", ".join(comma_parts[i : i + parts_per_chunk])
                    if sub.strip():
                        refined.append(sub)
            else:
                # Last resort: raw character split
                for i in range(0, len(chunk), target_chars):
                    sub = chunk[i : i + target_chars]
                    if sub.strip():
                        refined.append(sub)
    chunks = refined

    # Add overlap between chunks
    chunks = add_overlap(chunks)

    # Build output records
    total_parts = len(chunks)
    if total_parts <= 1:
        return [record]

    provenance = record.get("_provenance", {})
    results: list[dict] = []

    for idx, chunk in enumerate(chunks, 1):
        new_record = copy.deepcopy(record)
        new_messages = []
        for msg in new_record.get("messages", []):
            if msg["role"] == "user":
                new_messages.append(
                    {
                        "role": "user",
                        "content": f"[Part {idx}/{total_parts}] {msg['content']}",
                    }
                )
            elif msg["role"] == "assistant":
                new_messages.append({"role": "assistant", "content": chunk})
            else:
                new_messages.append(msg)

        new_record["messages"] = new_messages
        new_provenance = {**provenance, "split_part": idx, "split_total": total_parts}
        new_record["_provenance"] = new_provenance
        results.append(new_record)

    return results


def get_all_domains() -> list[str]:
    """Return all domain directories that contain a train.jsonl."""
    domains = []
    for d in sorted(HF_DATA.iterdir()):
        if d.is_dir() and (d / "train.jsonl").exists():
            domains.append(d.name)
    return domains


def process_domain(
    domain: str,
    *,
    max_seq: int = 4096,
    stats_only: bool = False,
    output_suffix: str = "curriculum",
) -> dict:
    """Process a single domain: analyze, split if needed, sort by length."""
    data_dir = HF_DATA / domain
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        print(f"  ERROR: {train_file} not found, skipping")
        return {}

    # Read all records
    records: list[dict] = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n_original = len(records)
    if n_original == 0:
        print(f"  ERROR: {train_file} is empty, skipping")
        return {}

    # Analyze original distribution
    original_tokens = [estimate_record_tokens(r) for r in records]
    original_sorted = sorted(original_tokens)

    # Classify records
    keep_as_is = 0
    keep_higher_seq = 0
    needs_split = 0
    split_threshold = max_seq * 2

    for tok in original_tokens:
        if tok <= max_seq:
            keep_as_is += 1
        elif tok <= split_threshold:
            keep_higher_seq += 1
        else:
            needs_split += 1

    # Perform splitting
    final_records: list[dict] = []
    total_splits_generated = 0
    records_actually_split = 0

    for record in records:
        tok = estimate_record_tokens(record)
        if tok <= split_threshold:
            final_records.append(record)
        else:
            parts = split_record(record, max_seq)
            if len(parts) > 1:
                records_actually_split += 1
                total_splits_generated += len(parts)
            final_records.extend(parts)

    n_final = len(final_records)

    # Compute final token distribution
    final_tokens = [estimate_record_tokens(r) for r in final_records]
    final_sorted = sorted(final_tokens)

    # Print stats
    def _percentile(data: list[int], p: float) -> int:
        idx = int(len(data) * p)
        idx = min(idx, len(data) - 1)
        return data[idx]

    print(f"\n  Domain: {domain}")
    print(f"  max_seq threshold: {max_seq}")
    print(f"  split threshold (max_seq * 2): {split_threshold}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ORIGINAL: {n_original} records")
    print(f"    min={original_sorted[0]}  median={_percentile(original_sorted, 0.5)}"
          f"  mean={int(statistics.mean(original_tokens))}  max={original_sorted[-1]}")
    print(f"    p90={_percentile(original_sorted, 0.9)}"
          f"  p95={_percentile(original_sorted, 0.95)}"
          f"  p99={_percentile(original_sorted, 0.99)}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Classification:")
    print(f"    keep as-is   (<= {max_seq} tokens): {keep_as_is:5d} ({keep_as_is / n_original * 100:.1f}%)")
    print(f"    keep higher  (<= {split_threshold} tokens): {keep_higher_seq:5d} ({keep_higher_seq / n_original * 100:.1f}%)")
    print(f"    needs split  (> {split_threshold} tokens): {needs_split:5d} ({needs_split / n_original * 100:.1f}%)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Splitting results:")
    print(f"    records split: {records_actually_split}")
    print(f"    chunks generated: {total_splits_generated}")
    print(f"    net new records: {n_final - n_original}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL: {n_final} records (after splitting)")
    print(f"    min={final_sorted[0]}  median={_percentile(final_sorted, 0.5)}"
          f"  mean={int(statistics.mean(final_tokens))}  max={final_sorted[-1]}")
    print(f"    p90={_percentile(final_sorted, 0.9)}"
          f"  p95={_percentile(final_sorted, 0.95)}"
          f"  p99={_percentile(final_sorted, 0.99)}")
    print(f"  Threshold analysis (final):")
    for threshold in THRESHOLDS:
        exceeds = sum(1 for t in final_tokens if t > threshold)
        pct = (exceeds / n_final) * 100 if n_final else 0
        marker = " *** TRUNCATION" if pct > 5 else ""
        print(f"    >{threshold:6d} tokens: {exceeds:5d} ({pct:5.1f}%){marker}")

    # Memory estimate (rough): tokens * 2 bytes * batch_size * grad_accum * overhead
    peak_tokens = final_sorted[-1]
    # For MLX LoRA: ~4 bytes per token per layer activation, 32 layers typical
    # Very rough: peak_mem_gb ~ max_seq * 4 * 32 * batch * grad_accum / 1e9
    # Simplified: just show tokens for user to reason about
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Memory estimate (max record = {peak_tokens} tokens):")
    for model, ga, seq in [("Devstral-24B", 16, 8192), ("Apertus-70B", 16, 4096)]:
        # Rough MLX LoRA memory: model_size + activations
        # activations ~ max_seq * hidden_dim * n_layers * 4 bytes * batch
        if "24B" in model:
            base_gb, hidden, layers = 48, 6144, 56
        else:
            base_gb, hidden, layers = 140, 8192, 80
        act_gb = (min(peak_tokens, seq) * hidden * layers * 4) / (1024**3)
        total_gb = base_gb + act_gb * ga
        print(f"    {model} (ga={ga}, seq={seq}): ~{total_gb:.0f} GB peak est.")

    result = {
        "domain": domain,
        "original": n_original,
        "split_count": records_actually_split,
        "chunks_generated": total_splits_generated,
        "final": n_final,
    }

    if stats_only:
        return result

    # Sort all records by token count ascending (curriculum order)
    final_with_tokens = [(estimate_record_tokens(r), r) for r in final_records]
    final_with_tokens.sort(key=lambda x: x[0])

    curriculum_file = data_dir / f"train_{output_suffix}.jsonl"
    with open(curriculum_file, "w") as f:
        for _, record in final_with_tokens:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n  Wrote {curriculum_file} ({n_final} records, sorted short->long)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum learning: split long sequences and sort by length"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domains",
        type=str,
        help="Comma-separated domain names (e.g., cpp,emc-dsp-power,security-fenrir)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all domains",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=4096,
        help="Max sequence length threshold for splitting (default: 4096)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print analysis, do not write files",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="curriculum",
        help="Suffix for output file (default: curriculum -> train_curriculum.jsonl)",
    )
    args = parser.parse_args()

    if args.all:
        domains = get_all_domains()
    else:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    print("=" * 60)
    print(" Curriculum Learning — Intelligent Sequence Splitting")
    print(f" Data: {HF_DATA}")
    print(f" Domains: {', '.join(domains)}")
    print(f" Max seq: {args.max_seq}")
    print(f" Split threshold: {args.max_seq * 2}")
    print(f" Mode: {'stats only' if args.stats_only else 'write curriculum files'}")
    print("=" * 60)

    all_results: list[dict] = []
    for domain in domains:
        result = process_domain(
            domain,
            max_seq=args.max_seq,
            stats_only=args.stats_only,
            output_suffix=args.output_suffix,
        )
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    if len(all_results) > 1:
        print(" SUMMARY")
        total_orig = sum(r["original"] for r in all_results)
        total_final = sum(r["final"] for r in all_results)
        total_split = sum(r["split_count"] for r in all_results)
        total_chunks = sum(r["chunks_generated"] for r in all_results)
        print(f"   Total original records: {total_orig}")
        print(f"   Total records split: {total_split}")
        print(f"   Total chunks generated: {total_chunks}")
        print(f"   Total final records: {total_final} (+{total_final - total_orig})")

    if args.stats_only:
        print(" Stats complete. Use without --stats-only to write curriculum files.")
    else:
        print(" Curriculum files written. Ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
