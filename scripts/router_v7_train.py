#!/usr/bin/env python3
"""Train router v7: MiniLM-L6-v2 (384d) + 256-hidden MLP, 47-label.

Same arch as v6 prod (drop-in compatible).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as tnn
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def encode(model: SentenceTransformer, texts: list[str], bs: int = 128) -> np.ndarray:
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/router-v7")
    ap.add_argument("--out-dir", default="output/router-v7")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-seq", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train = load_jsonl(data_dir / "train.jsonl")
    valid = load_jsonl(data_dir / "valid.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")

    domains = sorted({r["domain"] for r in train + valid + test})
    label_map = {d: i for i, d in enumerate(domains)}
    print(f"Domains: {len(domains)}")
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    enc = SentenceTransformer(args.encoder, device=device)
    enc.max_seq_length = args.max_seq
    dim = enc.get_sentence_embedding_dimension()
    print(f"Encoder dim: {dim}")

    print("Encoding train...")
    Xtr = torch.tensor(encode(enc, [r["prompt"] for r in train])).float()
    ytr = torch.tensor([label_map[r["domain"]] for r in train])
    print("Encoding valid...")
    Xv = torch.tensor(encode(enc, [r["prompt"] for r in valid])).float()
    yv = torch.tensor([label_map[r["domain"]] for r in valid])
    print("Encoding test...")
    Xt = torch.tensor(encode(enc, [r["prompt"] for r in test])).float()
    yt = torch.tensor([label_map[r["domain"]] for r in test])

    del enc

    counts = torch.bincount(ytr, minlength=len(domains)).float()
    weights = (counts.sum() / (len(domains) * counts.clamp(min=1))).clamp(max=10.0)

    mlp = tnn.Sequential(
        tnn.Linear(dim, args.hidden), tnn.GELU(), tnn.Dropout(0.1),
        tnn.Linear(args.hidden, len(domains)),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = tnn.CrossEntropyLoss(weight=weights)

    best_v_top1 = 0.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        mlp.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = mlp(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        mlp.eval()
        with torch.no_grad():
            vp = mlp(Xv)
            v_top1 = (vp.argmax(1) == yv).float().mean().item()
            v_top3 = sum(yv[j].item() in vp[j].topk(3).indices.tolist() for j in range(len(yv))) / len(yv)
        if v_top1 > best_v_top1:
            best_v_top1 = v_top1
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
        print(f"epoch {ep:>2d}  loss={total/max(1,nb):.4f}  val_top1={v_top1:.4f}  val_top3={v_top3:.4f}")

    # Restore best
    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()

    # Test eval
    with torch.no_grad():
        tp = mlp(Xt)
        t_top1 = (tp.argmax(1) == yt).float().mean().item()
        t_top3 = sum(yt[j].item() in tp[j].topk(3).indices.tolist() for j in range(len(yt))) / len(yt)
    print(f"\nTEST: top1={t_top1:.4f}, top3={t_top3:.4f}, best_val_top1={best_v_top1:.4f}")

    # Per-label F1 on test
    from sklearn.metrics import classification_report
    test_pred = tp.argmax(1).numpy()
    rep = classification_report(yt.numpy(), test_pred,
                                 target_names=domains, output_dict=True,
                                 zero_division=0)
    print("\nPer-label F1 (test):")
    for d in domains:
        row = rep.get(d, {})
        print(f"  {d}: f1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")

    macro_f1 = rep["macro avg"]["f1-score"]
    weighted_f1 = rep["weighted avg"]["f1-score"]
    print(f"\nmacro F1={macro_f1:.4f}, weighted F1={weighted_f1:.4f}")

    # Save in v6-compatible format
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(mlp.state_dict(), str(out / "router.safetensors"))
    meta = {
        "embedding_model": args.encoder,
        "embedding_dim": dim,
        "hidden_dim": args.hidden,
        "num_domains": len(domains),
        "domains": domains,
        "label_map": label_map,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "top1_acc": t_top1,
        "top3_acc": t_top3,
        "best_val_top1": best_v_top1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rebuilt_from": "data/router-v7 (47 domains, LLM-generated + edge-case curated)",
        "parent_checkpoint": "router-v6",
        "per_label_f1": {d: rep.get(d, {}).get("f1-score", 0) for d in domains},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2))
    print(f"\nSaved to {out}")

    # 4 obligatory smoke tests
    print("\n=== Obligatory smoke tests ===")
    enc2 = SentenceTransformer(args.encoder, device=device)
    enc2.max_seq_length = args.max_seq
    cases = [
        ("Think step by step about why the sky is blue", "reasoning"),
        ("Write Python fibonacci", "python"),
        ("Translate this technical doc to French", "traduction-tech"),
        ("Compile une fonction C++ avec un bug template", "cpp"),
    ]
    embs = enc2.encode([c[0] for c in cases], normalize_embeddings=True, convert_to_numpy=True)
    with torch.no_grad():
        logits = mlp(torch.tensor(embs).float())
        preds = logits.argmax(1).tolist()
    smoke = []
    for (text, expected), pid in zip(cases, preds):
        got = domains[pid]
        ok = (got == expected)
        smoke.append({"text": text, "expected": expected, "got": got, "ok": ok})
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] '{text[:50]}...' -> {got} (expected {expected})")

    (out / "smoke_tests.json").write_text(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
