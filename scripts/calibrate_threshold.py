"""Calibrate the router confidence threshold against the validation split.

For each candidate threshold, compute:
  - top-1 accuracy on prompts where the head is confident enough
  - coverage: fraction of prompts where any score >= threshold
  - fallback rate: 1 - coverage (these would route to Gemma)

Pick the threshold that maximises top-1 accuracy while keeping
coverage >= 95 % (i.e. we don't bail to Gemma more than 5 % of the
time on the labelled validation set).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "output/router"  # whichever checkpoint is currently active
VALID = REPO / "data/router/valid.jsonl"


def load_head() -> tuple[torch.nn.Module, list[str], str, int]:
    """Return (mlp, domains, encoder_name, embedding_dim)."""
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
    return mlp, domains, encoder_name, dim


def main() -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    mlp, domains, enc_name, dim = load_head()
    print(f"Loaded head: {len(domains)} domains, encoder={enc_name}")

    pairs: list[tuple[str, str]] = []
    with VALID.open() as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["prompt"], obj["domain"]))
    print(f"Validation set: {len(pairs)} prompts")

    enc = SentenceTransformer(enc_name)
    embs = enc.encode([p for p, _ in pairs], normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=False)
    with torch.no_grad():
        scores = mlp(embs.cpu())  # (N, num_domains)

    domain_idx = {d: i for i, d in enumerate(domains)}

    print("\n  threshold | coverage | top-1 (covered) | top-1 (overall)")
    print("  ----------+----------+-----------------+----------------")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        n_total = len(pairs)
        n_covered = 0
        n_correct_covered = 0
        n_correct_overall = 0
        for i, (_, target) in enumerate(pairs):
            target_idx = domain_idx.get(target)
            if target_idx is None:
                continue
            row = scores[i]
            top_idx = int(row.argmax().item())
            top_score = float(row[top_idx].item())
            covered = top_score >= thr
            if covered:
                n_covered += 1
                if top_idx == target_idx:
                    n_correct_covered += 1
                    n_correct_overall += 1
            # if not covered the gateway falls back to Gemma → counted wrong
        cov = n_covered / n_total
        top1_cov = n_correct_covered / n_covered if n_covered else 0.0
        top1_all = n_correct_overall / n_total
        print(f"   {thr:.2f}     |  {cov:.3f}   |    {top1_cov:.3f}        |   {top1_all:.3f}")


if __name__ == "__main__":
    main()
