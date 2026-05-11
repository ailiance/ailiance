# scripts/build_router_data.py
"""Build router train/valid splits.

By default reads the AI-Act-traceable corpus produced by
scripts/rebuild_router_dataset.py at data/router-clean/. Set the env var
ROUTER_LEGACY=1 to fall back to the legacy noisy corpus at
~/KIKI-Mac_tunner/data/micro-kiki/classified/ (deprecated, kept for
reproducibility of pre-2026-05 router checkpoints)."""

import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = Path.home() / "ailiance-mac-tuner/data/micro-kiki/classified"
CLEAN_DIR = REPO_ROOT / "data/router-clean"
CLASSIFIED_DIR = LEGACY_DIR if os.environ.get("ROUTER_LEGACY") == "1" else CLEAN_DIR
OUTPUT_DIR = REPO_ROOT / "data/router"
SEED = 42
TRAIN_RATIO = 0.8


def load_classified(directory: Path) -> list[dict]:
    rows = []
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        domain = jsonl_file.stem
        with open(jsonl_file) as f:
            for line in f:
                obj = json.loads(line)
                # Support both formats: {prompt:...} and {messages:[{role,content}]}
                prompt = obj.get("prompt") or obj.get("instruction", "")
                if not prompt and "messages" in obj:
                    for msg in obj["messages"]:
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break
                if prompt and prompt.strip():
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
