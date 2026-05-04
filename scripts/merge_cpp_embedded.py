#!/usr/bin/env python3
"""Merge scraped embedded C/C++ datasets with existing synthetic data into a
clean, microcontroller-focused cpp domain.

Sources:
  1. ESP-IDF examples (687 records, Apache-2.0)
  2. STM32 Cube examples (1812 records, BSD-3-Clause)
  3. Arduino built-in examples (99 records, CC0/MIT/Apache-2.0)
  4. Existing cpp/train.jsonl — only embedded-mcu + synthetic-embedded kept

Priority: real scraped > synthetic-embedded > generic
Target: 3000 records -> 2850 train / 150 valid
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
TARGET_TOTAL = 3000
TRAIN_SIZE = 2850
VALID_SIZE = 150

ROOT = Path(__file__).resolve().parent.parent
SCRAPED_DIR = ROOT / "data" / "scraped"
CPP_DIR = ROOT / "data" / "hf-traced" / "cpp"

SOURCES = [
    SCRAPED_DIR / "espidf-examples" / "train.jsonl",
    SCRAPED_DIR / "stm32-examples" / "train.jsonl",
    SCRAPED_DIR / "arduino-examples" / "train.jsonl",
]

KEEP_DOMAIN_TAGS = {"embedded-mcu", "synthetic-embedded"}


def assistant_fingerprint(rec: dict) -> str:
    """Return first 200 chars of the assistant message for dedup."""
    for msg in rec.get("messages", []):
        if msg.get("role") == "assistant":
            return msg["content"][:200].strip()
    return ""


def classify_platform(rec: dict) -> str:
    """Heuristic platform classification from provenance + content."""
    prov = rec.get("_provenance", {})
    source = prov.get("source", "")
    assistant_text = assistant_fingerprint(rec)
    user_text = ""
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            user_text = msg["content"][:500].lower()
            break

    combined = (source + " " + assistant_text + " " + user_text).lower()

    if "esp" in source.lower() or "esp_" in combined or "esp-idf" in combined or "esp32" in combined:
        return "ESP32"
    if "stm32" in source.lower() or "stm32" in combined or "hal_" in combined:
        return "STM32"
    if "arduino" in source.lower() or "arduino" in combined or ".ino" in combined:
        return "Arduino/AVR"
    return "generic"


def classify_peripheral(rec: dict) -> str:
    """Heuristic peripheral classification."""
    prov = rec.get("_provenance", {})
    if "peripheral" in prov:
        return prov["peripheral"]
    if "category" in prov:
        return prov["category"]

    combined = ""
    for msg in rec.get("messages", []):
        combined += " " + msg.get("content", "")[:600]
    combined = combined.lower()

    peripherals = [
        ("bluetooth", "bluetooth"), ("ble", "bluetooth"), ("wifi", "wifi"),
        ("i2c", "I2C"), ("spi", "SPI"), ("uart", "UART"), ("usart", "UART"),
        ("adc", "ADC"), ("dac", "DAC"), ("pwm", "PWM"), ("timer", "timer"),
        ("gpio", "GPIO"), ("dma", "DMA"), ("flash", "FLASH"), ("rtc", "RTC"),
        ("watchdog", "watchdog"), ("wdt", "watchdog"), ("can", "CAN"),
        ("ethernet", "ethernet"), ("usb", "USB"), ("i2s", "I2S"),
        ("led", "LED/GPIO"), ("interrupt", "interrupt"), ("exti", "interrupt"),
        ("rtos", "RTOS"), ("freertos", "RTOS"), ("sleep", "power-mgmt"),
        ("deep_sleep", "power-mgmt"), ("low_power", "power-mgmt"),
    ]

    for keyword, label in peripherals:
        if keyword in combined:
            return label
    return "misc"


def priority_key(rec: dict) -> int:
    """Lower = higher priority. Real scraped code first."""
    tag = rec.get("_provenance", {}).get("domain_tag", "")
    source = rec.get("_provenance", {}).get("source", "")

    # Real vendor examples
    if "espressif" in source or "STMicroelectronics" in source or "arduino/arduino" in source:
        return 0
    if tag == "embedded-mcu":
        return 1
    if tag == "synthetic-embedded":
        return 2
    return 3


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)

    # --- Load scraped sources ---
    scraped: list[dict] = []
    for src_path in SOURCES:
        records = load_jsonl(src_path)
        print(f"  Loaded {len(records):>5} from {src_path.relative_to(ROOT)}")
        scraped.extend(records)
    print(f"  Total scraped: {len(scraped)}")

    # --- Load existing cpp, filter to embedded-mcu + synthetic-embedded ---
    existing_all = load_jsonl(CPP_DIR / "train.jsonl")
    existing = [
        r for r in existing_all
        if r.get("_provenance", {}).get("domain_tag", "") in KEEP_DOMAIN_TAGS
    ]
    dropped = len(existing_all) - len(existing)
    print(f"  Loaded {len(existing_all):>5} from existing cpp/train.jsonl")
    print(f"  Kept {len(existing):>5} (embedded-mcu + synthetic-embedded), dropped {dropped} generic-cpp")

    # --- Combine ---
    pool = scraped + existing
    print(f"  Combined pool: {len(pool)}")

    # --- Deduplicate by assistant content (first 200 chars) ---
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0
    for rec in pool:
        fp = assistant_fingerprint(rec)
        if fp in seen:
            dup_count += 1
            continue
        seen.add(fp)
        deduped.append(rec)
    print(f"  After dedup: {len(deduped)} (removed {dup_count} duplicates)")

    # --- Sort by priority, then shuffle within priority groups ---
    deduped.sort(key=priority_key)

    # Take up to TARGET_TOTAL, priority-ordered
    selected = deduped[:TARGET_TOTAL] if len(deduped) >= TARGET_TOTAL else deduped

    if len(selected) < TARGET_TOTAL:
        print(f"  WARNING: Only {len(selected)} records available, target was {TARGET_TOTAL}")

    # Shuffle for training
    rng.shuffle(selected)

    # --- Split ---
    train = selected[:TRAIN_SIZE]
    valid = selected[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]

    # --- Save ---
    save_jsonl(train, CPP_DIR / "train.jsonl")
    save_jsonl(valid, CPP_DIR / "valid.jsonl")
    print(f"\n  Saved {len(train)} train + {len(valid)} valid to {CPP_DIR.relative_to(ROOT)}/")

    # --- Stats ---
    all_records = train + valid

    # Source distribution
    print("\n=== Records per source ===")
    source_counts: Counter[str] = Counter()
    for rec in all_records:
        source_counts[rec.get("_provenance", {}).get("source", "unknown")] += 1
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    # Platform distribution
    print("\n=== Platform distribution ===")
    platform_counts: Counter[str] = Counter()
    for rec in all_records:
        platform_counts[classify_platform(rec)] += 1
    for plat, count in platform_counts.most_common():
        pct = 100 * count / len(all_records)
        print(f"  {plat}: {count} ({pct:.1f}%)")

    # Peripheral distribution (top 15)
    print("\n=== Peripheral distribution (top 15) ===")
    periph_counts: Counter[str] = Counter()
    for rec in all_records:
        periph_counts[classify_peripheral(rec)] += 1
    for periph, count in periph_counts.most_common(15):
        pct = 100 * count / len(all_records)
        print(f"  {periph}: {count} ({pct:.1f}%)")

    # Real vs synthetic
    print("\n=== Real code vs synthetic ===")
    real = sum(1 for r in all_records if priority_key(r) <= 1)
    synthetic = sum(1 for r in all_records if priority_key(r) == 2)
    other = len(all_records) - real - synthetic
    total = len(all_records)
    print(f"  Real vendor code: {real} ({100*real/total:.1f}%)")
    print(f"  Synthetic-embedded: {synthetic} ({100*synthetic/total:.1f}%)")
    if other:
        print(f"  Other: {other} ({100*other/total:.1f}%)")

    # Domain tag distribution
    print("\n=== Domain tags ===")
    tag_counts: Counter[str] = Counter()
    for rec in all_records:
        tag_counts[rec.get("_provenance", {}).get("domain_tag", "unknown")] += 1
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    print("=== merge_cpp_embedded.py ===\n")
    main()
    print("\nDone.")
