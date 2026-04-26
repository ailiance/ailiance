# scripts/build_router_data.py
"""Build router training data from micro-kiki classified data + new EU domains."""

import json
import random
from pathlib import Path

CLASSIFIED_DIR = Path.home() / "KIKI-Mac_tunner/data/micro-kiki/classified"
OUTPUT_DIR = Path("data/router")
SEED = 42
TRAIN_RATIO = 0.8


def load_classified(directory: Path) -> list[dict]:
    rows = []
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        domain = jsonl_file.stem
        with open(jsonl_file) as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("instruction", "")
                if prompt.strip():
                    rows.append({"prompt": prompt.strip(), "domain": domain})
    return rows


def split_and_write(rows: list[dict], output_dir: Path) -> None:
    random.seed(SEED)
    by_domain: dict[str, list] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    train, valid = [], []
    for domain, items in sorted(by_domain.items()):
        random.shuffle(items)
        cut = max(1, int(len(items) * TRAIN_RATIO))
        train.extend(items[:cut])
        valid.extend(items[cut:])
        print(f"  {domain:<25s} train={cut:>5d}  valid={len(items)-cut:>5d}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(output_dir / "valid.jsonl", "w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal: train={len(train)}, valid={len(valid)}")


if __name__ == "__main__":
    rows = load_classified(CLASSIFIED_DIR)
    print(f"Loaded {len(rows)} rows from {CLASSIFIED_DIR}")
    split_and_write(rows, OUTPUT_DIR)
