# src/router/classifier.py
"""Jina v3 + MLP domain classifier.

Encodes user query with Jina Embeddings v3 (1024d),
classifies into one of 40 domains via 2-layer MLP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as tnn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterConfig:
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dim: int = 1024
    hidden_dim: int = 512
    num_domains: int = 40
    threshold: float = 0.12
    max_active: int = 4


def _build_mlp(cfg: RouterConfig) -> "tnn.Module":
    import torch.nn as tnn

    class RouterMLP(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = tnn.Sequential(
                tnn.Linear(cfg.embedding_dim, cfg.hidden_dim),
                tnn.GELU(),
                tnn.Dropout(0.1),
                tnn.Linear(cfg.hidden_dim, cfg.num_domains),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            import torch
            return torch.sigmoid(self.net(x))

    return RouterMLP()


class DomainRouter:
    """Encodes text with Jina v3, classifies with MLP head."""

    def __init__(self, cfg: RouterConfig, weights_dir: Path):
        self._cfg = cfg
        self._encoder = None
        self._mlp = None
        self._domains: list[str] = []
        self._load(weights_dir)

    def _load(self, weights_dir: Path) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from safetensors.torch import load_file

        meta_path = weights_dir / "meta.json"
        weights_path = weights_dir / "router.safetensors"

        meta = json.loads(meta_path.read_text())
        self._domains = meta["domains"]

        self._encoder = SentenceTransformer(self._cfg.embedding_model)
        self._mlp = _build_mlp(self._cfg)
        state = load_file(str(weights_path))
        # Remap keys: training saves "0.weight" but MLP expects "net.0.weight"
        if any(k.startswith("net.") for k in state):
            remapped = state
        else:
            remapped = {f"net.{k}": v for k, v in state.items()}
        self._mlp.load_state_dict(remapped)
        self._mlp.eval()
        log.info("Router loaded: %d domains, %s encoder", len(self._domains), self._cfg.embedding_model)

    def route(self, query: str) -> list[tuple[str, float]]:
        import torch

        with torch.no_grad():
            emb = self._encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            emb = emb.cpu()  # MLP is on CPU
            scores = self._mlp(emb.unsqueeze(0)).squeeze(0)

        results = []
        for idx in torch.argsort(scores, descending=True):
            i = idx.item()
            s = scores[i].item()
            if s < self._cfg.threshold:
                break
            results.append((self._domains[i], s))
            if len(results) >= self._cfg.max_active:
                break
        return results
