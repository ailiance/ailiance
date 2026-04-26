# scripts/train_router.py
"""Train Jina v3 + MLP domain router for eu-kiki."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as tnn
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def train(args):
    train_data = load_jsonl(Path(args.data_dir) / "train.jsonl")
    valid_data = load_jsonl(Path(args.data_dir) / "valid.jsonl")

    all_domains = sorted({r["domain"] for r in train_data})
    label_map = {d: i for i, d in enumerate(all_domains)}
    num_domains = len(all_domains)
    print(f"Domains: {num_domains}")

    encoder = SentenceTransformer(args.embedding_model)
    dim = encoder.get_sentence_embedding_dimension()
    print(f"Embedding dim: {dim}")

    print("Encoding train...")
    train_embs = encoder.encode([r["prompt"] for r in train_data], show_progress_bar=True,
                                 convert_to_tensor=True, normalize_embeddings=True)
    train_labels = torch.tensor([label_map[r["domain"]] for r in train_data])

    print("Encoding valid...")
    valid_embs = encoder.encode([r["prompt"] for r in valid_data], show_progress_bar=True,
                                 convert_to_tensor=True, normalize_embeddings=True)
    valid_labels = torch.tensor([label_map[r["domain"]] for r in valid_data])

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
            batch_idx = perm[i:i + args.batch_size]
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
            top3_hits = 0
            for j in range(len(valid_labels)):
                top3 = v_logits[j].topk(3).indices
                if valid_labels[j] in top3:
                    top3_hits += 1
            top3 = top3_hits / len(valid_labels)

        avg_loss = total_loss / (len(perm) / args.batch_size)
        print(f"epoch {epoch:>2d}  loss={avg_loss:.4f}  top1={top1:.3f}  top3={top3:.3f}")

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
    print(f"Saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/router")
    p.add_argument("--output-dir", default="output/router")
    p.add_argument("--embedding-model", default="jinaai/jina-embeddings-v3")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
