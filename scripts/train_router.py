# scripts/train_router.py
"""Train Jina v3 + MLP domain router for ailiance.

Uses native JinaEmbeddingsV3Model from transformers (not SentenceTransformer)
to avoid custom code compatibility issues.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as tnn
from safetensors.torch import save_file


def _patch_jina_compat():
    """Patch Jina v3 custom code for transformers 5.6+ compatibility.

    Jina v3's XLMRobertaLoRA and XLMRobertaModel don't define
    all_tied_weights_keys, which transformers 5.6+ expects.
    """
    import os
    import glob
    cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/jinaai")
    for pattern in ["**/modeling_lora.py", "**/modeling_xlm_roberta.py"]:
        for path in glob.glob(os.path.join(cache, pattern), recursive=True):
            txt = open(path).read()
            marker = "self.all_tied_weights_keys = {}"
            if marker not in txt:
                txt = txt.replace(
                    "super().__init__(config)\n",
                    f"super().__init__(config)\n        {marker}\n",
                    1,
                )
                open(path, "w").write(txt)
                # Clear pycache
                pycache = os.path.join(os.path.dirname(path), "__pycache__")
                if os.path.isdir(pycache):
                    import shutil
                    shutil.rmtree(pycache)
                print(f"  Patched {os.path.basename(path)}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def train(args):
    train_data = load_jsonl(Path(args.data_dir) / "train.jsonl")
    valid_data = load_jsonl(Path(args.data_dir) / "valid.jsonl")

    all_domains = sorted({r["domain"] for r in train_data + valid_data})
    label_map = {d: i for i, d in enumerate(all_domains)}
    num_domains = len(all_domains)
    print(f"Domains: {num_domains}")

    # Load Jina v3 natively (no SentenceTransformer — avoids multiprocessing crash)
    _patch_jina_compat()
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading {args.embedding_model}...")
    jina_tok = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    jina_model = AutoModel.from_pretrained(args.embedding_model, trust_remote_code=True)
    jina_model.eval()

    def _encode_batch(texts: list[str], batch_size: int = 32) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = jina_tok(batch, padding=True, truncation=True,
                              return_tensors="pt", max_length=512)
            with torch.no_grad():
                out = jina_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs)
            if i % (batch_size * 50) == 0:
                print(f"  {i + len(batch)}/{len(texts)}")
        return torch.cat(all_embs, dim=0)

    # Detect embedding dim from a test encode
    with torch.no_grad():
        test_in = jina_tok(["test"], return_tensors="pt", truncation=True, max_length=512)
        test_out = jina_model(**test_in)
        dim = test_out.last_hidden_state.shape[-1]
    print(f"Embedding dim: {dim}")

    print("Encoding train...")
    train_embs = _encode_batch([r["prompt"] for r in train_data])
    train_labels = torch.tensor([label_map[r["domain"]] for r in train_data])

    print("Encoding valid...")
    valid_embs = _encode_batch([r["prompt"] for r in valid_data])
    valid_labels = torch.tensor([label_map[r["domain"]] for r in valid_data])

    # Free encoder memory
    del jina_model, jina_tok

    # Class weights (inverse frequency, clamped)
    counts = torch.bincount(train_labels, minlength=num_domains).float()
    weights = (counts.sum() / (num_domains * counts.clamp(min=1))).clamp(max=10.0)

    # MLP
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
            top3_hits = 0
            for j in range(len(valid_labels)):
                top3 = v_logits[j].topk(3).indices
                if valid_labels[j] in top3:
                    top3_hits += 1
            top3 = top3_hits / len(valid_labels)

        n_batches = max(1, len(perm) / args.batch_size)
        avg_loss = total_loss / n_batches
        print(f"epoch {epoch:>2d}  loss={avg_loss:.4f}  top1={top1:.3f}  top3={top3:.3f}")

    # Save
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
    p.add_argument("--data-dir", default="data/router")
    p.add_argument("--output-dir", default="output/router")
    p.add_argument("--embedding-model", default="jinaai/jina-embeddings-v3")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
