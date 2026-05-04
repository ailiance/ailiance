#!/usr/bin/env python3
"""
Prepare KiCad PCB dataset with full-sequence integrity.

Problem: max_seq=8192 truncates large .kicad_mod footprints mid-component.
Solution: Raise target to 16384 and use S-expression-aware splitting for
any record that still exceeds the budget.

KiCad .kicad_pcb files are nested S-expressions with top-level blocks like
(footprint ...), (segment ...), (via ...), (zone ...), (gr_line ...).
The file header — (kicad_pcb (version ...) (generator ...) (general ...)
(layers ...) (setup ...)) — is shared context prepended to every chunk.

.kicad_mod files are single self-contained footprints (typically 200-2000
tokens). They should almost never need splitting.

Usage:
    uv run python scripts/prepare_kicad_pcb_fullseq.py
    uv run python scripts/prepare_kicad_pcb_fullseq.py --stats-only
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

HF_DATA = Path(__file__).resolve().parent.parent / "data" / "hf-traced"
DOMAIN = "kicad-pcb"

MAX_SEQ = 16384
CHARS_PER_TOKEN = 3.5
# Leave room for user prompt + header when chunking
CHUNK_TOKEN_BUDGET = 14000
OVERLAP_LINES = 8

THRESHOLDS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return int(len(text) / CHARS_PER_TOKEN)


def concat_messages(record: dict) -> str:
    """Concatenate all message contents."""
    return "".join(msg.get("content", "") for msg in record.get("messages", []))


def get_user_prompt(record: dict) -> str:
    """Extract the user (instruction) message."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_assistant_content(record: dict) -> str:
    """Extract the assistant (code) response."""
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def find_matching_paren(text: str, start: int) -> int:
    """Find the index of the closing paren matching the open paren at `start`."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return len(text) - 1


def extract_top_level_sexps(text: str) -> tuple[str, list[str]]:
    """
    Parse KiCad S-expression text into header + list of top-level blocks.

    The header includes everything from the start through (layers ...) and
    (setup ...) blocks. Remaining top-level S-expressions become the block list.

    For .kicad_mod files (single footprint), the entire content is one block
    so header is empty and blocks = [full text].
    """
    text = text.strip()

    # Detect .kicad_mod files — they start with (module or (footprint at top level
    if text.startswith("(module ") or text.startswith("(footprint "):
        return "", [text]

    # For .kicad_pcb files, parse the outer wrapper
    # Find "(kicad_pcb" at the start
    if not text.startswith("(kicad_pcb"):
        # Unknown format — treat as single block
        return "", [text]

    # Scan for top-level S-expressions inside the outer (kicad_pcb ...)
    # Skip past the opening "(kicad_pcb" to find inner blocks
    header_keywords = {"version", "generator", "generator_version", "general", "layers", "setup", "page", "title_block"}
    header_parts: list[str] = []
    body_blocks: list[str] = []
    header_done = False

    i = len("(kicad_pcb")
    # Skip whitespace
    while i < len(text) and text[i] in (' ', '\n', '\r', '\t'):
        i += 1

    while i < len(text) - 1:  # -1 to skip final closing paren
        # Skip whitespace
        while i < len(text) and text[i] in (' ', '\n', '\r', '\t'):
            i += 1
        if i >= len(text) - 1:
            break
        if text[i] == ')':
            break  # End of outer kicad_pcb
        if text[i] != '(':
            i += 1
            continue

        # Found opening paren — extract the block keyword
        end = find_matching_paren(text, i)
        block = text[i:end + 1]

        # Determine keyword (first word after open paren)
        keyword_end = i + 1
        while keyword_end < end and text[keyword_end] not in (' ', '\n', '\t', ')'):
            keyword_end += 1
        keyword = text[i + 1:keyword_end]

        if not header_done and keyword in header_keywords:
            header_parts.append(block)
        else:
            header_done = True
            body_blocks.append(block)

        i = end + 1

    # Reconstruct header as a valid partial kicad_pcb
    if header_parts:
        header = "(kicad_pcb\n  " + "\n  ".join(header_parts)
    else:
        header = ""

    return header, body_blocks


def merge_blocks_to_budget(
    blocks: list[str],
    budget_chars: int,
) -> list[str]:
    """Merge consecutive blocks into chunks that fit within the character budget."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for block in blocks:
        block_len = len(block)
        if current and (current_len + block_len + 1) > budget_chars:
            chunks.append("\n  ".join(current))
            current = [block]
            current_len = block_len
        else:
            current.append(block)
            current_len += block_len + 1  # +1 for newline

    if current:
        chunks.append("\n  ".join(current))

    return chunks


def build_chunk_with_header(header: str, chunk_body: str, close: bool = True) -> str:
    """Combine header and body into a valid KiCad S-expression."""
    if not header:
        return chunk_body
    if close:
        return f"{header}\n\n  {chunk_body}\n)"
    return f"{header}\n\n  {chunk_body}"


def split_record_kicad(record: dict) -> list[dict]:
    """
    Split a KiCad PCB record using S-expression-aware chunking.

    Records <= MAX_SEQ tokens: kept as-is.
    Records > MAX_SEQ tokens: split into chunks with shared header.
    """
    total_tokens = estimate_tokens(concat_messages(record))

    if total_tokens <= MAX_SEQ:
        return [record]

    user_prompt = get_user_prompt(record)
    code = get_assistant_content(record)

    # Budget in characters for the code portion of each chunk
    prompt_tokens = estimate_tokens(user_prompt) + 20  # 20 for [Part N/M] prefix
    header_budget = 0

    header, blocks = extract_top_level_sexps(code)

    if header:
        header_budget = estimate_tokens(header) + 10  # closing paren etc.

    available_tokens = CHUNK_TOKEN_BUDGET - prompt_tokens - header_budget
    budget_chars = int(available_tokens * CHARS_PER_TOKEN)
    budget_chars = max(budget_chars, 3000)  # safety floor

    if len(blocks) <= 1:
        # Single block (e.g. a huge footprint) — cannot split meaningfully
        return [record]

    chunks = merge_blocks_to_budget(blocks, budget_chars)

    if len(chunks) <= 1:
        return [record]

    # Add overlap context between chunks
    for i in range(1, len(chunks)):
        prev_lines = chunks[i - 1].split("\n")
        overlap = "\n".join(prev_lines[-OVERLAP_LINES:])
        chunks[i] = f"// ... (continued from previous part)\n{overlap}\n\n{chunks[i]}"

    # Build final chunks with header
    final_chunks: list[str] = []
    for i, chunk_body in enumerate(chunks):
        is_last = i == len(chunks) - 1
        final_chunks.append(build_chunk_with_header(header, chunk_body, close=is_last))

    # Build output records
    provenance = record.get("_provenance", {})
    total_parts = len(final_chunks)
    results: list[dict] = []

    for idx, chunk in enumerate(final_chunks, 1):
        new_record = copy.deepcopy(record)
        new_messages = []
        for msg in new_record.get("messages", []):
            if msg["role"] == "user":
                new_messages.append({
                    "role": "user",
                    "content": f"[Part {idx}/{total_parts}] {msg['content']}",
                })
            elif msg["role"] == "assistant":
                new_messages.append({"role": "assistant", "content": chunk})
            else:
                new_messages.append(msg)

        new_record["messages"] = new_messages
        new_record["_provenance"] = {
            **provenance,
            "split_part": idx,
            "split_total": total_parts,
        }
        results.append(new_record)

    return results


def print_distribution(lengths: list[int], label: str) -> None:
    """Print token length distribution with buckets."""
    n = len(lengths)
    if n == 0:
        print(f"  {label}: empty")
        return

    sorted_lens = sorted(lengths)
    print(f"\n  {label} ({n} records)")
    print(f"    Min: {sorted_lens[0]:,}   Max: {sorted_lens[-1]:,}")
    print(f"    Mean: {statistics.mean(sorted_lens):,.0f}   Median: {statistics.median(sorted_lens):,.0f}")

    if n > 10:
        p90 = sorted_lens[int(n * 0.90)]
        p95 = sorted_lens[int(n * 0.95)]
        p99 = sorted_lens[int(n * 0.99)]
        print(f"    p90: {p90:,}   p95: {p95:,}   p99: {p99:,}")

    print("    Buckets:")
    for t in THRESHOLDS:
        count = sum(1 for x in sorted_lens if x <= t)
        pct = count / n * 100
        print(f"      <= {t:>6,}: {count:>6,} ({pct:.1f}%)")

    over_max = sum(1 for x in sorted_lens if x > MAX_SEQ)
    print(f"      > {MAX_SEQ:>6,}: {over_max:>6,} ({'PROBLEM' if over_max else 'OK'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KiCad PCB fullseq dataset")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats, don't write")
    args = parser.parse_args()

    data_dir = HF_DATA / DOMAIN

    # Read from train_original.jsonl (the unsplit source)
    original_file = data_dir / "train_original.jsonl"
    if not original_file.exists():
        # Fallback: if train_original doesn't exist, use train.jsonl
        original_file = data_dir / "train.jsonl"
        print(f"WARNING: train_original.jsonl not found, using train.jsonl")

    print(f"Reading: {original_file}")
    records: list[dict] = []
    with open(original_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Original records: {len(records)}")

    # Analyze original distribution
    orig_lengths = [estimate_tokens(concat_messages(r)) for r in records]
    print_distribution(orig_lengths, "BEFORE splitting")

    over_threshold = sum(1 for t in orig_lengths if t > MAX_SEQ)
    print(f"\n  Records > {MAX_SEQ} tokens (need splitting): {over_threshold}")

    if args.stats_only:
        return

    # Process: split records exceeding MAX_SEQ
    output_records: list[dict] = []
    split_count = 0
    parts_generated = 0

    for record in records:
        splits = split_record_kicad(record)
        if len(splits) > 1:
            split_count += 1
            parts_generated += len(splits)
        output_records.extend(splits)

    # Sort by length (curriculum: short -> long)
    output_records.sort(key=lambda r: estimate_tokens(concat_messages(r)))

    # Analyze output distribution
    out_lengths = [estimate_tokens(concat_messages(r)) for r in output_records]
    print_distribution(out_lengths, "AFTER splitting")

    # Write output
    output_file = data_dir / "train_fullseq.jsonl"
    with open(output_file, "w") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input:  {len(records):,} records (from {original_file.name})")
    print(f"  Output: {len(output_records):,} records -> {output_file.name}")
    print(f"  Records split: {split_count}")
    print(f"  Parts generated from splits: {parts_generated}")
    print(f"  Max tokens after: {max(out_lengths):,}")
    print(f"  Records > {MAX_SEQ} after: {sum(1 for t in out_lengths if t > MAX_SEQ)}")
    print(f"  Sorted: curriculum (short -> long)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
