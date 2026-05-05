"""Side-by-side benchmark: MiniLM L6 vs Jina v3 on the eu-kiki router task.

Both encode the same prompts; we measure:
  - encoding latency
  - confidence of the (correct) target domain via the existing router head
    (only with MiniLM since the head was trained against MiniLM 384d; Jina is
    1024d so the existing head can't be reused)
  - prompt-pair cosine: same-domain vs cross-domain pairs (encoder-only signal)

For an honest end-to-end Jina comparison we'd need to retrain the MLP head
against Jina embeddings — out of scope here. This script focuses on what's
measurable now.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent  # eu-kiki/

# Test set: prompt + ground-truth target domain
TESTS = [
    ("Écris un script Python pour parser un CSV", "python"),
    ("Implement a Rust TCP server with tokio", "rust"),
    ("How do I dockerize a FastAPI app?", "docker"),
    ("Comment dimensionner une self pour un buck converter ?", "power"),
    ("Aide-moi à debugger ce schéma KiCad avec ESP32", "stm32"),  # multi-tag
    ("Simule un filtre RC dans NGSPICE", "spice"),
    ("Translate 'Hello world' to French", "chat-fr"),
    ("Bonjour comment ça va ?", "chat-fr"),
    ("Write a SQL query joining users and orders", "sql"),
    ("Quelle est la norme IEC 61010 pour la sécurité ?", "calcul-normatif"),  # may not be a label
]

# Pairs for same-domain vs cross-domain cosine signal (encoder-only)
PAIRS_SAME = [
    ("Écris un test pytest", "Comment mocker un appel HTTP en pytest ?"),
    ("Bonjour", "Salut, comment vas-tu ?"),
    ("Implement a Rust binary", "Cargo new --bin"),
]
PAIRS_DIFF = [
    ("Écris un test pytest", "Bonjour comment ça va ?"),
    ("Implement a Rust binary", "Translate to French"),
    ("Buck converter design", "SQL join syntax"),
]


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def bench_encoder(name, model_id, load_kwargs=None, encode_kwargs=None):
    load_kwargs = load_kwargs or {}
    encode_kwargs = encode_kwargs or {}
    print(f"\n=== {name} ===  ({model_id})")
    t0 = time.time()
    enc = SentenceTransformer(model_id, **load_kwargs)
    print(f"  load       : {time.time() - t0:.1f}s")

    # Warmup
    enc.encode("warmup", normalize_embeddings=True, **encode_kwargs)

    # Encode
    queries = [q for q, _ in TESTS]
    t0 = time.time()
    embs = enc.encode(
        queries, normalize_embeddings=True, convert_to_numpy=True, **encode_kwargs
    )
    dt = time.time() - t0
    print(
        f"  encode {len(queries):2d} prompts: {dt*1000:.0f}ms total, "
        f"{dt*1000/len(queries):.1f}ms/prompt   dim={embs.shape[1]}"
    )

    # Pair similarity test
    same_cos = []
    diff_cos = []
    all_pairs = [(a, b, "same") for a, b in PAIRS_SAME] + [
        (a, b, "diff") for a, b in PAIRS_DIFF
    ]
    pair_embs = enc.encode(
        [t for pair in all_pairs for t in pair[:2]],
        normalize_embeddings=True,
        convert_to_numpy=True,
        **encode_kwargs,
    )
    for i, (_, _, kind) in enumerate(all_pairs):
        c = cos(pair_embs[2 * i], pair_embs[2 * i + 1])
        (same_cos if kind == "same" else diff_cos).append(c)
    print(
        f"  cos same-domain pairs: mean={np.mean(same_cos):.3f}  "
        f"diff-domain: mean={np.mean(diff_cos):.3f}  "
        f"separation Δ={np.mean(same_cos) - np.mean(diff_cos):.3f}"
    )
    return embs


def head_predict_with_minilm(embs):
    """Run the MiniLM-trained MLP head on MiniLM embeddings."""
    weights_dir = REPO / "output" / "router"
    meta = json.loads((weights_dir / "meta.json").read_text())
    domains = meta["domains"]

    from safetensors.torch import load_file
    state = load_file(str(weights_dir / "router.safetensors"))
    if not any(k.startswith("net.") for k in state):
        state = {f"net.{k}": v for k, v in state.items()}

    import torch.nn as tnn
    mlp = tnn.Sequential(
        tnn.Linear(384, 256), tnn.GELU(), tnn.Dropout(0.1), tnn.Linear(256, 32)
    )
    # Wrap to match keys
    full = tnn.Module()
    full.net = mlp
    full.load_state_dict(state)
    full.eval()

    print("\n--- MiniLM head predictions ---")
    print(f"{'prompt':52s} | top-1 (score)        | target ok?")
    with torch.no_grad():
        scores = torch.sigmoid(full.net(torch.tensor(embs, dtype=torch.float32)))
    for i, (q, target) in enumerate(TESTS):
        s = scores[i]
        idx = int(torch.argmax(s).item())
        top = domains[idx]
        sc = float(s[idx])
        ok = "✓" if top == target else f"✗ (target={target})"
        print(f"{q[:50]:52s} | {top:18s} {sc:.2f} | {ok}")


def main():
    embs_minilm = bench_encoder(
        "MiniLM L6 v2", "sentence-transformers/all-MiniLM-L6-v2"
    )
    head_predict_with_minilm(embs_minilm)

    print(
        "\n(Jina head not retrained — only encoder-only signals reportable for Jina)"
    )
    # Jina v3 is task-conditioned. For routing/classification we want the
    # `classification` task adapter (LoRA) so embeddings are tuned for label
    # discrimination rather than general retrieval similarity.
    embs_jina_class = bench_encoder(
        "Jina v3 (task=classification)",
        "jinaai/jina-embeddings-v3",
        load_kwargs={"trust_remote_code": True},
        encode_kwargs={"task": "classification"},
    )
    embs_jina_sep = bench_encoder(
        "Jina v3 (task=separation)",
        "jinaai/jina-embeddings-v3",
        load_kwargs={"trust_remote_code": True},
        encode_kwargs={"task": "separation"},
    )
    print(
        "\nNote: top-1 accuracy with Jina would require retraining the MLP head "
        "(384d → 1024d). The encoder-only Δ above is the cleanest comparable signal."
    )


if __name__ == "__main__":
    main()
