#!/usr/bin/env python3
"""Encode router train/valid with MiniLM-L6-v2 → save NumPy embeddings.

Adapted from encode_router_jina.py but uses sentence-transformers/
all-MiniLM-L6-v2 (384d, 22M params) to match the production router
encoder. Runs on MPS / CUDA / CPU automatically.

Usage:
    .venv/bin/python scripts/encode_router_minilm.py \\
        --output-dir data/router-minilm-v6
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/router")
    ap.add_argument("--output-dir", default="data/router-minilm-v6")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-seq", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    encoder = SentenceTransformer(args.model, device=device)
    encoder.max_seq_length = args.max_seq

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        rows = load_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        prompts = [r["prompt"] for r in rows]
        domains = [r["domain"] for r in rows]
        print(f"{split}: encoding {len(prompts)} prompts...")
        embs = encoder.encode(
            prompts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        np.save(out / f"{split}_embs.npy", embs.astype(np.float32))
        (out / f"{split}_domains.json").write_text(json.dumps(domains))
        print(f"  wrote {split}_embs.npy shape={embs.shape}")

    meta = {
        "encoder": args.model,
        "embedding_dim": int(embs.shape[1]),
        "max_seq_length": args.max_seq,
        "device": device,
    }
    (out / "encode_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
