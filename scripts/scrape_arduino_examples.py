#!/usr/bin/env python3
"""Scrape Arduino examples from cloned repos for embedded C++ training data.

Sources (permissive licenses only):
- arduino/arduino-examples  (CC0 1.0 — public domain)
- adafruit/Adafruit_Sensor  (Apache-2.0)
- adafruit/DHT-sensor-library (MIT)
- knolleary/pubsubclient     (MIT — MQTT)
- bblanchon/ArduinoJson      (MIT)

Excluded (LGPL — cautious for EU AI Act):
- arduino/ArduinoCore-avr    (LGPL-2.1)
- adafruit/Adafruit_NeoPixel (LGPL-3.0)
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Repos with their license and base path for example search
REPOS: list[dict[str, str]] = [
    {
        "name": "arduino/arduino-examples",
        "path": "/tmp/arduino-examples",
        "license": "CC0-1.0",
        "search_root": "/tmp/arduino-examples/examples",
    },
    {
        "name": "adafruit/Adafruit_Sensor",
        "path": "/tmp/Adafruit_Sensor",
        "license": "Apache-2.0",
        "search_root": "/tmp/Adafruit_Sensor",
    },
    {
        "name": "adafruit/DHT-sensor-library",
        "path": "/tmp/DHT-sensor-library",
        "license": "MIT",
        "search_root": "/tmp/DHT-sensor-library",
    },
    {
        "name": "knolleary/pubsubclient",
        "path": "/tmp/pubsubclient",
        "license": "MIT",
        "search_root": "/tmp/pubsubclient",
    },
    {
        "name": "bblanchon/ArduinoJson",
        "path": "/tmp/ArduinoJson",
        "license": "MIT",
        "search_root": "/tmp/ArduinoJson",
    },
]

OUTPUT = Path("data/scraped/arduino-examples")
MIN_LINES = 10
MAX_LINES = 400


def extract_header_comment(code: str) -> str | None:
    """Extract the leading block comment (/* ... */) or consecutive // lines."""
    # Try block comment first
    match = re.match(r"^\s*/\*(.+?)\*/", code, re.DOTALL)
    if match:
        raw = match.group(1)
        # Clean up: remove leading * from each line
        lines = [re.sub(r"^\s*\*\s?", "", line) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line.strip()).strip()

    # Try consecutive // lines at the top
    comment_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            comment_lines.append(stripped.lstrip("/ ").strip())
        elif stripped == "":
            if comment_lines:
                comment_lines.append("")
        else:
            break
    if len(comment_lines) >= 2:
        return "\n".join(comment_lines).strip()
    return None


def classify_category(file_path: str, code: str) -> str:
    """Classify the example into a domain category based on path and content.

    Serial.print is used as debug output in nearly all Arduino sketches, so
    'serial' is checked last (lowest priority).  Higher-priority categories
    are tested first so that an example using Serial merely for debug output
    gets classified by its *primary* topic.
    """
    path_lower = file_path.lower()
    code_lower = code.lower()

    # Ordered by priority: most specific first, serial/general last
    category_signals: list[tuple[str, list[str]]] = [
        ("json-parsing", ["json", "deserialize", "serialize", "arduinojson"]),
        ("communication-mqtt", ["mqtt", "pubsub"]),
        ("sensors", ["adafruit_sensor", "dht", "temperature", "humidity", "pressure", "accel", "gyro", "bme", "bmp"]),
        ("communication-spi", ["spi.begin", "spi.transfer", "spi.h"]),
        ("communication-i2c", ["wire.begin", "wire.request", "i2c", "twi"]),
        ("communication-wifi", ["wifi", "ethernet", "wificlient", "ethclient"]),
        ("servo-motor", ["servo", "motor", "stepper"]),
        ("display", ["lcd", "display", "oled", "tft", "screen"]),
        ("data-storage", ["eeprom", "sd.begin", "flash", "storage"]),
        ("analog-io", ["analogread", "analogwrite", "pwm", "fade"]),
        ("digital-io", ["digitalread", "digitalwrite", "button", "pinmode"]),
        ("timing", ["millis()", "micros()", "interrupt", "attachinterrupt"]),
        ("strings", ["string ", "indexof", "substring", "startswith", "charat"]),
        ("control-structures", ["arrays", "forloop", "whileloop", "switchcase"]),
        # Serial is the fallback — almost every sketch uses Serial for debug
        ("communication-serial", ["serial.begin", "uart", "usart"]),
    ]

    # Check path first (directory names are very reliable)
    for category, signals in category_signals:
        for signal in signals:
            if signal in path_lower:
                return category

    # Then check code content
    for category, signals in category_signals:
        for signal in signals:
            if signal in code_lower:
                return category

    return "general"


def build_user_prompt(
    filename: str,
    header_comment: str | None,
    category: str,
    repo_name: str,
) -> str:
    """Create a user prompt from the example metadata."""
    # Use filename (without .ino/.cpp) as the example name
    example_name = Path(filename).stem

    base = f"Write an Arduino sketch for: {example_name} (category: {category})"

    if header_comment:
        # Trim to first ~500 chars
        context = header_comment[:500]
        if len(header_comment) > 500:
            context += "..."
        return f"Context from example documentation:\n{context}\n\n{base}"

    return base


def find_examples(repo: dict[str, str]) -> list[Path]:
    """Find .ino and .cpp files under examples/ directories in a repo."""
    root = Path(repo["search_root"])
    if not root.exists():
        return []

    results: list[Path] = []
    # Look for files in examples/ directories specifically
    for examples_dir in root.rglob("examples"):
        if examples_dir.is_dir():
            for ext in (".ino", ".cpp"):
                results.extend(examples_dir.rglob(f"*{ext}"))

    # If no examples/ subdirectory found, the search_root itself might be examples/
    if not results and root.name == "examples":
        for ext in (".ino", ".cpp"):
            results.extend(root.rglob(f"*{ext}"))

    return sorted(set(results))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped_short = 0
    skipped_long = 0
    skipped_read_error = 0
    repo_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()

    for repo in REPOS:
        repo_path = Path(repo["path"])
        if not repo_path.exists():
            print(f"SKIP: {repo['name']} — directory {repo['path']} not found")
            continue

        example_files = find_examples(repo)
        print(f"  {repo['name']}: {len(example_files)} example files found")

        for src_file in example_files:
            try:
                code = src_file.read_text(errors="replace")
            except OSError:
                skipped_read_error += 1
                continue

            line_count = len(code.splitlines())
            if line_count < MIN_LINES:
                skipped_short += 1
                continue
            if line_count > MAX_LINES:
                skipped_long += 1
                continue

            rel_path = str(src_file.relative_to(repo_path))
            header_comment = extract_header_comment(code)
            category = classify_category(rel_path, code)

            user_prompt = build_user_prompt(
                src_file.name, header_comment, category, repo["name"]
            )

            records.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": code},
                ],
                "_provenance": {
                    "source": repo["name"],
                    "license": repo["license"],
                    "file_path": rel_path,
                    "domain_tag": "embedded-mcu",
                    "access_date": datetime.now(timezone.utc).isoformat(),
                    "category": category,
                },
            })
            repo_counts[repo["name"]] += 1
            category_counts[category] += 1
            license_counts[repo["license"]] += 1

    # Save
    out_file = OUTPUT / "train.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'=' * 60}")
    print("Arduino Examples Scrape Report")
    print(f"{'=' * 60}")
    print(f"Total example files scanned: {sum(repo_counts.values()) + skipped_short + skipped_long + skipped_read_error}")
    print(f"Skipped (< {MIN_LINES} lines):      {skipped_short}")
    print(f"Skipped (> {MAX_LINES} lines):     {skipped_long}")
    print(f"Skipped (read errors):       {skipped_read_error}")
    print(f"Training records saved:      {len(records)}")
    print(f"Output: {out_file}")

    print("\nBy source repo:")
    for repo_name, count in repo_counts.most_common():
        bar = "\u2588" * max(1, count // 2)
        print(f"  {repo_name:<35} {count:>4}  {bar}")

    print("\nBy license:")
    for lic, count in license_counts.most_common():
        print(f"  {lic:<20} {count:>4}")

    print("\nBy category:")
    for cat, count in category_counts.most_common():
        bar = "\u2588" * max(1, count)
        print(f"  {cat:<25} {count:>4}  {bar}")

    # Sample records
    if records:
        print(f"\n--- Sample record (first) ---")
        sample = records[0]
        print(f"  User prompt:  {sample['messages'][0]['content'][:120]}...")
        print(f"  Code length:  {len(sample['messages'][1]['content'])} chars")
        print(f"  Provenance:   {sample['_provenance']}")

        if len(records) > 1:
            mid = records[len(records) // 2]
            print(f"\n--- Sample record (middle) ---")
            print(f"  User prompt:  {mid['messages'][0]['content'][:120]}...")
            print(f"  Code length:  {len(mid['messages'][1]['content'])} chars")
            print(f"  Provenance:   {mid['_provenance']}")

        if len(records) > 2:
            last = records[-1]
            print(f"\n--- Sample record (last) ---")
            print(f"  User prompt:  {last['messages'][0]['content'][:120]}...")
            print(f"  Code length:  {len(last['messages'][1]['content'])} chars")
            print(f"  Provenance:   {last['_provenance']}")


if __name__ == "__main__":
    main()
