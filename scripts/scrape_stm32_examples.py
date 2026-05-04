#!/usr/bin/env python3
"""Scrape STM32 HAL/LL/MIX examples from ST's official GitHub repositories.

Source repos (BSD-3-Clause):
  - STM32CubeF4
  - STM32CubeH7
  - STM32CubeL4

Requires repos pre-cloned to /tmp/STM32Cube{F4,H7,L4} with Projects/ checked out.
EU AI Act: Article 53 compliant — open-source BSD-3-Clause, public GitHub repos.
"""

import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOS = {
    "STM32CubeF4": Path("/tmp/STM32CubeF4"),
    "STM32CubeH7": Path("/tmp/STM32CubeH7"),
    "STM32CubeL4": Path("/tmp/STM32CubeL4"),
}

OUTPUT = Path("data/scraped/stm32-examples")
MIN_LINES = 20
MAX_LINES = 500

EXAMPLE_DIR_PATTERN = re.compile(r"Examples(?:_LL|_MIX)?")
PERIPHERAL_PATTERN = re.compile(
    r"(GPIO|UART|USART|SPI|I2C|ADC|DAC|DMA|TIM|RTC|RCC|IWDG|WWDG|"
    r"CRC|PWR|FLASH|FMC|SDIO|SDMMC|CAN|FDCAN|USB|ETH|SAI|DFSDM|"
    r"COMP|OPAMP|LPTIM|LTDC|HASH|CRYP|RNG|TSC|LCD|QSPI|OCTOSPI|"
    r"CORTEXM|CORTEX|HAL|BSP|HRTIM|MDMA|BDMA|HSEM|IPCC)",
    re.IGNORECASE,
)


def extract_board_name(example_path: Path) -> str:
    """Extract board name from path like Projects/<board>/Examples/..."""
    parts = example_path.parts
    try:
        proj_idx = parts.index("Projects")
        return parts[proj_idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def extract_peripheral(example_path: Path) -> str:
    """Extract peripheral name from path like .../Examples/GPIO/GPIO_Toggle/..."""
    parts = example_path.parts
    for i, part in enumerate(parts):
        if EXAMPLE_DIR_PATTERN.fullmatch(part):
            if i + 1 < len(parts):
                return parts[i + 1].upper()
    return "OTHER"


def extract_example_name(example_dir: Path) -> str:
    """Extract example name from the example directory path."""
    parts = example_dir.parts
    for i, part in enumerate(parts):
        if EXAMPLE_DIR_PATTERN.fullmatch(part):
            remaining = parts[i + 1 :]
            return "/".join(remaining)
    return example_dir.name


def read_readme(example_dir: Path) -> str:
    """Read readme.txt or README.md from an example directory.

    ST readme.txt files use doxygen format with @par sections.
    We extract the Example Description section as the most useful content.
    """
    for name in ("readme.txt", "README.md", "readme.md", "Readme.txt"):
        readme_path = example_dir / name
        if readme_path.exists():
            try:
                text = readme_path.read_text(encoding="utf-8", errors="replace")

                # Try to extract @par Example Description section
                desc_match = re.search(
                    r"@par\s+Example\s+Description\s*\n(.*?)(?=@par\s|\Z)",
                    text,
                    re.DOTALL,
                )
                if desc_match:
                    desc = desc_match.group(1).strip()
                    # Clean leading comment markers
                    desc = re.sub(r"^\s*\*\s?", "", desc, flags=re.MULTILINE)
                    desc = re.sub(r"^\s*-\s*$", "", desc, flags=re.MULTILINE)
                    # Collapse excessive blank lines
                    desc = re.sub(r"\n{3,}", "\n\n", desc)
                    return desc.strip()

                # Fallback: extract @page title line
                page_match = re.search(r"@page\s+\S+\s+(.*)", text)
                if page_match:
                    return page_match.group(1).strip()

                # Last resort: strip boilerplate and return what's left
                text = re.sub(r"/\*\*.*?\*/", "", text, flags=re.DOTALL)
                text = re.sub(r"@verbatim.*?@endverbatim", "", text, flags=re.DOTALL)
                text = re.sub(r"@par\s+", "", text)
                text = re.sub(r"\*{3,}.*?\*{3,}", "", text, flags=re.DOTALL)
                text = re.sub(r"Copyright.*?AS-IS\.", "", text, flags=re.DOTALL)
                lines = [ln.strip() for ln in text.splitlines()]
                text = "\n".join(ln for ln in lines if ln)
                return text.strip()
            except Exception:
                pass
    return ""


def find_example_dirs(repo_root: Path) -> list[Path]:
    """Find all example directories (directories containing Src/main.c)."""
    results = []
    projects_dir = repo_root / "Projects"
    if not projects_dir.exists():
        return results

    for example_type_dir in projects_dir.rglob("*"):
        if not example_type_dir.is_dir():
            continue
        if not EXAMPLE_DIR_PATTERN.fullmatch(example_type_dir.name):
            continue
        # Walk 2 levels deep: peripheral/example_name
        for peripheral_dir in sorted(example_type_dir.iterdir()):
            if not peripheral_dir.is_dir():
                continue
            for example_dir in sorted(peripheral_dir.iterdir()):
                if not example_dir.is_dir():
                    continue
                src_dir = example_dir / "Src"
                if src_dir.is_dir():
                    results.append(example_dir)

    return results


def collect_source_files(example_dir: Path) -> list[Path]:
    """Collect main.c, main.cpp, and Src/*.c files."""
    src_dir = example_dir / "Src"
    files = []
    if not src_dir.exists():
        return files

    for f in sorted(src_dir.iterdir()):
        if f.suffix in (".c", ".cpp") and f.is_file():
            # Skip system/interrupt boilerplate
            if f.name in (
                "system_stm32f4xx.c",
                "system_stm32h7xx.c",
                "system_stm32l4xx.c",
                "stm32f4xx_it.c",
                "stm32h7xx_it.c",
                "stm32l4xx_it.c",
                "stm32f4xx_hal_msp.c",
                "stm32h7xx_hal_msp.c",
                "stm32l4xx_hal_msp.c",
            ):
                continue
            files.append(f)

    return files


def read_source(path: Path) -> tuple[str, int]:
    """Read source file and return (content, line_count)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        line_count = len(content.splitlines())
        return content, line_count
    except Exception:
        return "", 0


def build_user_prompt(readme: str, example_name: str, board: str, example_type: str) -> str:
    """Build user prompt from readme and metadata."""
    type_label = {
        "Examples": "HAL",
        "Examples_LL": "LL (Low-Layer)",
        "Examples_MIX": "HAL/LL mixed",
    }.get(example_type, "HAL")

    prompt_parts = []
    if readme:
        # Truncate readme if too long
        if len(readme) > 1500:
            readme = readme[:1500] + "..."
        prompt_parts.append(readme)
        prompt_parts.append("")

    prompt_parts.append(
        f"Write STM32 {type_label} code for {example_name} on {board}."
    )
    return "\n".join(prompt_parts)


def get_example_type(example_dir: Path) -> str:
    """Get the example type (Examples, Examples_LL, Examples_MIX)."""
    for part in example_dir.parts:
        if EXAMPLE_DIR_PATTERN.fullmatch(part):
            return part
    return "Examples"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skipped_short = 0
    skipped_long = 0
    peripheral_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()

    for repo_name, repo_path in REPOS.items():
        if not repo_path.exists():
            print(f"  SKIP {repo_name}: not found at {repo_path}")
            continue

        print(f"Processing {repo_name}...")
        example_dirs = find_example_dirs(repo_path)
        print(f"  Found {len(example_dirs)} example directories")

        for example_dir in example_dirs:
            board = extract_board_name(example_dir)
            peripheral = extract_peripheral(example_dir)
            example_name = extract_example_name(example_dir)
            example_type = get_example_type(example_dir)
            readme = read_readme(example_dir)

            source_files = collect_source_files(example_dir)
            if not source_files:
                continue

            # Combine all source files for this example
            combined_source_parts = []
            total_lines = 0
            file_paths = []

            for src_file in source_files:
                content, line_count = read_source(src_file)
                if not content:
                    continue
                total_lines += line_count
                rel_path = src_file.relative_to(repo_path)
                file_paths.append(str(rel_path))
                if len(source_files) > 1:
                    combined_source_parts.append(f"// === {src_file.name} ===")
                combined_source_parts.append(content)

            if total_lines < MIN_LINES:
                skipped_short += 1
                continue
            if total_lines > MAX_LINES:
                skipped_long += 1
                continue

            combined_source = "\n\n".join(combined_source_parts)
            user_prompt = build_user_prompt(readme, example_name, board, example_type)

            record = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": combined_source},
                ],
                "_provenance": {
                    "source": f"STMicroelectronics/{repo_name}",
                    "license": "BSD-3-Clause",
                    "file_path": file_paths[0] if len(file_paths) == 1 else file_paths,
                    "domain_tag": "embedded-mcu",
                    "peripheral": peripheral,
                    "board": board,
                    "example_type": example_type,
                    "access_date": datetime.now(timezone.utc).isoformat(),
                },
            }
            records.append(record)
            peripheral_counts[peripheral] += 1
            repo_counts[repo_name] += 1

    # Save
    out_path = OUTPUT / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'='*60}")
    print(f"Total examples: {len(records)}")
    print(f"Skipped (< {MIN_LINES} lines): {skipped_short}")
    print(f"Skipped (> {MAX_LINES} lines): {skipped_long}")
    print(f"Output: {out_path}")
    print(f"\nBy repo:")
    for repo, count in repo_counts.most_common():
        print(f"  {repo}: {count}")
    print(f"\nBy peripheral (top 20):")
    for periph, count in peripheral_counts.most_common(20):
        print(f"  {periph:12s}: {count}")

    # Show sample record
    if records:
        print(f"\n{'='*60}")
        print("Sample record (first):")
        sample = records[0]
        print(f"  User prompt ({len(sample['messages'][0]['content'])} chars):")
        print(f"    {sample['messages'][0]['content'][:200]}...")
        print(f"  Assistant ({len(sample['messages'][1]['content'])} chars)")
        print(f"  Provenance: {json.dumps(sample['_provenance'], indent=4)}")


if __name__ == "__main__":
    main()
