#!/usr/bin/env python3
"""Encode router training data with Jina v3 and save embeddings.

Run with KIKI-Mac_tunner venv (Python 3.12) for SentenceTransformer compat:
    ~/KIKI-Mac_tunner/.venv/bin/python scripts/encode_router_jina.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch Jina compat
cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/jinaai")
for root, dirs, files in os.walk(cache):
    for f in files:
        if f in ("modeling_lora.py", "modeling_xlm_roberta.py"):
            path = os.path.join(root, f)
            txt = open(path).read()
            marker = "self.all_tied_weights_keys = {}"
            if marker not in txt:
                txt = txt.replace(
                    "super().__init__(config)\n",
                    f"super().__init__(config)\n        {marker}\n",
                    1,
                )
                open(path, "w").write(txt)
                import shutil
                pc = os.path.join(os.path.dirname(path), "__pycache__")
                if os.path.isdir(pc):
                    shutil.rmtree(pc)
                print(f"Patched {f}")

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/router")
OUTPUT_DIR = Path("data/router-jina-v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Jina v3...")
encoder = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
dim = encoder.get_sentence_embedding_dimension()
print(f"Jina v3 loaded: dim={dim}")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


for split in ("train", "valid"):
    data = load_jsonl(DATA_DIR / f"{split}.jsonl")
    texts = [r["prompt"] for r in data]
    domains = [r["domain"] for r in data]

    print(f"Encoding {split}: {len(texts)} texts...")
    embs = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=32)

    np.save(str(OUTPUT_DIR / f"{split}_embs.npy"), embs)
    with open(OUTPUT_DIR / f"{split}_domains.json", "w") as f:
        json.dump(domains, f)

    print(f"  Saved: {embs.shape} → {OUTPUT_DIR}/{split}_embs.npy")

print("Done!")
