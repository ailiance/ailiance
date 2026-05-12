#!/usr/bin/env python3
"""Eval router v7 on test set + 4 mandatory test cases + compare vs v6 prod on shared 32 labels."""
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as tnn
from safetensors.torch import load_file
from collections import defaultdict

V7_DIR = Path("/home/electron/ailiance/output/router-v7-multimodel")
V6_DIR = Path("/home/electron/ailiance/output/router-v6")
DATA_DIR = Path("/home/electron/ailiance/data/router-v7-multimodel")

TEST_CASES = [
    ("Write a Python function for binary search", "python"),
    ("Bonjour, comment ça va aujourd'hui ?", "chat-fr"),
    ("Think step by step about the trolley problem", "reasoning"),
    ("KiCad ERC error: pin not connected to any other pin", "kicad"),
]


def load_router(out_dir: Path):
    meta = json.loads((out_dir / "meta.json").read_text())
    state = load_file(str(out_dir / "router.safetensors"))
    dim = meta["embedding_dim"]
    hidden = meta["hidden_dim"]
    nd = meta["num_domains"]
    mlp = tnn.Sequential(
        tnn.Linear(dim, hidden), tnn.GELU(), tnn.Dropout(0.1),
        tnn.Linear(hidden, nd),
    )
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp, meta


def encode_minilm(texts: list[str], model_name: str) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    out = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        inp = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            o = mdl(**inp)
        mask = inp["attention_mask"].unsqueeze(-1).float()
        emb = (o.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb)
    return torch.cat(out, 0)


def main():
    print("=== Eval v7 multi-model router ===")
    mlp7, meta7 = load_router(V7_DIR)
    mlp6, meta6 = load_router(V6_DIR)
    print(f"v7: {meta7['num_domains']} labels, hidden={meta7['hidden_dim']}, embedding={meta7['embedding_model']}")
    print(f"v6: {meta6['num_domains']} labels, hidden={meta6['hidden_dim']}, embedding={meta6['embedding_model']}")

    test = [json.loads(l) for l in (DATA_DIR / "test.jsonl").read_text().splitlines()]
    print(f"Test set: {len(test)} examples")

    # Encode test set with v7 embedding model
    emb_model = meta7["embedding_model"]
    test_texts = [r["prompt"] for r in test]
    test_embs = encode_minilm(test_texts, emb_model)

    label_map7 = meta7["label_map"]
    inv7 = {v: k for k, v in label_map7.items()}

    with torch.no_grad():
        logits = mlp7(test_embs)
        preds = logits.argmax(1).tolist()

    # Overall + per-label F1
    correct = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for i, ex in enumerate(test):
        true = ex["domain"]
        pred = inv7[preds[i]]
        if pred == true:
            per_label[true]["tp"] += 1
            correct += 1
        else:
            per_label[true]["fn"] += 1
            per_label[pred]["fp"] += 1
    top1 = correct / len(test)
    print(f"\nOverall top1: {top1:.4f}")

    # Top3
    top3_hits = 0
    top3_idx = logits.topk(3, dim=1).indices
    for i, ex in enumerate(test):
        true_idx = label_map7.get(ex["domain"])
        if true_idx in top3_idx[i].tolist():
            top3_hits += 1
    print(f"Overall top3: {top3_hits/len(test):.4f}")

    # 15 new labels (in v7 but not in v6)
    new_labels = sorted(set(label_map7) - set(meta6["label_map"]))
    print(f"\n--- {len(new_labels)} new labels (not in v6) ---")
    new_f1 = []
    for lab in new_labels:
        s = per_label[lab]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        new_f1.append(f1)
        print(f"  {lab:25s}  P={prec:.2f} R={rec:.2f} F1={f1:.2f}  (n_test={tp+fn})")
    print(f"new-labels avg F1: {sum(new_f1)/len(new_f1):.3f}")

    # Mandatory test cases
    print("\n--- 4 mandatory test cases ---")
    tc_texts = [t for t, _ in TEST_CASES]
    tc_embs = encode_minilm(tc_texts, emb_model)
    with torch.no_grad():
        tc_logits = mlp7(tc_embs)
        tc_preds = tc_logits.argmax(1).tolist()
    passed = 0
    for (txt, exp), p_idx in zip(TEST_CASES, tc_preds):
        pred = inv7[p_idx]
        ok = pred == exp
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] '{txt[:50]}' -> {pred} (exp {exp})")
    print(f"Test cases: {passed}/{len(TEST_CASES)}")

    # Compare on shared 32 labels: subset test examples whose true label is in v6
    shared = set(label_map7) & set(meta6["label_map"])
    sub_idx = [i for i, ex in enumerate(test) if ex["domain"] in shared]
    print(f"\n--- v6 vs v7 on shared 32 labels ({len(sub_idx)} test ex) ---")
    label_map6 = meta6["label_map"]
    inv6 = {v: k for k, v in label_map6.items()}
    # encode with v6 embedding model (same MiniLM, can reuse)
    if meta6["embedding_model"] == emb_model:
        sub_embs = test_embs[sub_idx]
    else:
        sub_embs = encode_minilm([test[i]["prompt"] for i in sub_idx], meta6["embedding_model"])
    with torch.no_grad():
        v6_logits = mlp6(sub_embs)
        v6_preds = v6_logits.argmax(1).tolist()
    v6_correct = 0
    v7_correct = 0
    for k, i in enumerate(sub_idx):
        true = test[i]["domain"]
        if inv6[v6_preds[k]] == true:
            v6_correct += 1
        if inv7[preds[i]] == true:
            v7_correct += 1
    print(f"v6 top1: {v6_correct/len(sub_idx):.4f}")
    print(f"v7 top1: {v7_correct/len(sub_idx):.4f}")

    report = {
        "test_top1": top1,
        "test_top3": top3_hits / len(test),
        "new_labels_avg_f1": sum(new_f1) / len(new_f1) if new_f1 else None,
        "new_labels": {lab: {"f1": new_f1[i]} for i, lab in enumerate(new_labels)},
        "test_cases_pass": passed,
        "test_cases_total": len(TEST_CASES),
        "v6_vs_v7_shared": {
            "n": len(sub_idx),
            "v6_top1": v6_correct / len(sub_idx) if sub_idx else None,
            "v7_top1": v7_correct / len(sub_idx) if sub_idx else None,
        },
    }
    (V7_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: {V7_DIR}/eval_report.json")


if __name__ == "__main__":
    main()
