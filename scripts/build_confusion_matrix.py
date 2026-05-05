"""Compute the router confusion matrix on the validation split.

Outputs:
  output/confusion-<sha>.csv      — full matrix (domains × domains)
  output/confusion-<sha>-top10.md — markdown summary of worst pairs

The script reuses the same head loading code as calibrate_threshold.py.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "output/router"
VALID = REPO / "data/router/valid.jsonl"
OUT_DIR = REPO / "output"


def load_head() -> tuple[torch.nn.Module, list[str], str]:
    meta = json.loads((WEIGHTS / "meta.json").read_text())
    domains = meta["domains"]
    encoder_name = meta["embedding_model"]
    dim = meta["embedding_dim"]
    hidden = meta["hidden_dim"]
    n = len(domains)
    import torch.nn as tnn
    class RouterMLP(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = tnn.Sequential(
                tnn.Linear(dim, hidden), tnn.GELU(),
                tnn.Dropout(0.1), tnn.Linear(hidden, n),
            )
        def forward(self, x): return torch.sigmoid(self.net(x))
    mlp = RouterMLP()
    state = load_file(str(WEIGHTS / "router.safetensors"))
    if not any(k.startswith("net.") for k in state):
        state = {f"net.{k}": v for k, v in state.items()}
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp, domains, encoder_name


def main() -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    mlp, domains, enc_name = load_head()
    domain_idx = {d: i for i, d in enumerate(domains)}
    n = len(domains)

    pairs: list[tuple[str, str]] = []
    with VALID.open() as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["prompt"], obj["domain"]))

    enc = SentenceTransformer(enc_name)
    embs = enc.encode([p for p, _ in pairs], normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=False)
    with torch.no_grad():
        scores = mlp(embs.cpu())

    matrix = [[0] * n for _ in range(n)]
    confusions: Counter = Counter()
    for i, (_, target) in enumerate(pairs):
        ti = domain_idx.get(target)
        if ti is None:
            continue
        pi = int(scores[i].argmax().item())
        matrix[ti][pi] += 1
        if pi != ti:
            confusions[(target, domains[pi])] += 1

    # CSV
    csv_path = OUT_DIR / "confusion.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["target↓ / predicted→"] + domains)
        for ti, target in enumerate(domains):
            w.writerow([target] + matrix[ti])
    print(f"  wrote {csv_path}")

    # Markdown top-10
    md_path = OUT_DIR / "confusion-top10.md"
    with md_path.open("w") as f:
        f.write("# Top 10 confusion pairs\n\n")
        f.write("| Target | Predicted | Count |\n|---|---|---|\n")
        for (t, p), c in confusions.most_common(10):
            f.write(f"| {t} | {p} | {c} |\n")
    print(f"  wrote {md_path}")
    print()
    print("Top 10 confusions:")
    for (t, p), c in confusions.most_common(10):
        print(f"  {t:18s} -> {p:18s}  ({c})")


if __name__ == "__main__":
    main()
