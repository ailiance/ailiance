#!/usr/bin/env python3
"""Train MLP router from pre-computed embeddings.

Usage:
    uv run python scripts/train_router_from_embeddings.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as tnn
from safetensors.torch import save_file


def train(args):
    emb_dir = Path(args.emb_dir)
    train_embs = torch.from_numpy(np.load(str(emb_dir / "train_embs.npy")))
    valid_embs = torch.from_numpy(np.load(str(emb_dir / "valid_embs.npy")))
    train_domains = json.loads((emb_dir / "train_domains.json").read_text())
    valid_domains = json.loads((emb_dir / "valid_domains.json").read_text())

    all_domains = sorted(set(train_domains))
    label_map = {d: i for i, d in enumerate(all_domains)}
    num_domains = len(all_domains)
    dim = train_embs.shape[1]

    train_labels = torch.tensor([label_map[d] for d in train_domains])
    valid_labels = torch.tensor([label_map[d] for d in valid_domains])

    print(f"Domains: {num_domains}, dim: {dim}")
    print(f"Train: {len(train_embs)}, Valid: {len(valid_embs)}")

    counts = torch.bincount(train_labels, minlength=num_domains).float()
    weights = (counts.sum() / (num_domains * counts.clamp(min=1))).clamp(max=10.0)

    hidden = args.hidden_dim
    mlp = tnn.Sequential(
        tnn.Linear(dim, hidden), tnn.GELU(), tnn.Dropout(0.1),
        tnn.Linear(hidden, num_domains),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = tnn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        perm = torch.randperm(len(train_embs))
        total_loss = 0.0
        for i in range(0, len(perm), args.batch_size):
            batch_idx = perm[i : i + args.batch_size]
            logits = mlp(train_embs[batch_idx])
            loss = loss_fn(logits, train_labels[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        mlp.eval()
        with torch.no_grad():
            v_logits = mlp(valid_embs)
            v_preds = v_logits.argmax(dim=1)
            top1 = (v_preds == valid_labels).float().mean().item()
            top3_hits = sum(
                1 for j in range(len(valid_labels))
                if valid_labels[j] in v_logits[j].topk(3).indices
            )
            top3 = top3_hits / len(valid_labels)

        n_batches = max(1, len(perm) / args.batch_size)
        print(f"epoch {epoch:>2d}  loss={total_loss / n_batches:.4f}  top1={top1:.3f}  top3={top3:.3f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(mlp.state_dict(), str(out / "router.safetensors"))
    meta = {
        "embedding_model": args.embedding_model,
        "embedding_dim": dim,
        "hidden_dim": hidden,
        "num_domains": num_domains,
        "domains": all_domains,
        "label_map": label_map,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", default="data/router-jina-v3")
    p.add_argument("--output-dir", default="output/router")
    p.add_argument("--embedding-model", default="jinaai/jina-embeddings-v3")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
