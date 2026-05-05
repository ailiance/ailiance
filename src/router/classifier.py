# src/router/classifier.py
"""Jina v3 + MLP domain classifier.

Encodes user query with Jina Embeddings v3 (1024d),
classifies into one of 40 domains via 2-layer MLP.

Includes a per-process L1 LRU cache keyed on sha256(user_msg) so
repeated prompts skip the ~50-100ms Jina embedding compute.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as tnn

log = logging.getLogger(__name__)


# Optional Prometheus metrics — fall back to no-ops if unavailable.
try:
    from prometheus_client import Counter as _PromCounter

    _ROUTER_CACHE_HITS = _PromCounter(
        "eu_kiki_router_cache_hits_total",
        "Number of L1 LRU cache hits in DomainRouter.route()",
    )
    _ROUTER_CACHE_MISSES = _PromCounter(
        "eu_kiki_router_cache_misses_total",
        "Number of L1 LRU cache misses in DomainRouter.route()",
    )
except Exception:  # pragma: no cover - optional dep / duplicate registration
    class _NoopCounter:
        def inc(self, _amount: float = 1) -> None:
            return None

    _ROUTER_CACHE_HITS = _NoopCounter()
    _ROUTER_CACHE_MISSES = _NoopCounter()

try:
    from prometheus_client import Counter as _PromCounter2  # noqa: F401

    _ROUTER_L2_HITS = _PromCounter(
        "eu_kiki_router_l2_hits_total",
        "L2 semantic-cache hits (cosine match) in DomainRouter.route()",
    )
except Exception:  # pragma: no cover
    class _NoopCounter2:
        def inc(self, _amount: float = 1) -> None:
            return None

    _ROUTER_L2_HITS = _NoopCounter2()


_CACHE_MAXSIZE = 1024

# Default prompts used to warm L1 LRU at boot (eliminates p95 spike on first
# real query). Cover the most-frequent French/English greetings and a few
# common task starts.
DEFAULT_WARMUP_PROMPTS: list[str] = [
    "coucou", "salut", "bonjour", "hello", "hi",
    "merci", "ok", "comment ça va",
    "explique moi", "donne moi un exemple",
    "écris du code python", "give me code", "write a function",
    "traduis", "translate",
]


@dataclass(frozen=True)
class RouterConfig:
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dim: int = 1024
    hidden_dim: int = 512
    num_domains: int = 40
    threshold: float = 0.12
    max_active: int = 4
    # Encoder device: "mps" on Apple Silicon, "cuda" on NVIDIA, "cpu" fallback.
    # Auto-resolved at load time when set to "auto".
    encoder_device: str = "auto"
    # Cap input length to keep encoding fast — routing decisions stabilize
    # well before the full 8192 tokens that Jina v3 supports.
    max_seq_length: int = 128
    # L2 semantic cache: cosine threshold for embedding-similarity hit.
    # Catches paraphrases ("coucou" / "salut" / "hello"). Set to 0 to disable.
    l2_cache_threshold: float = 0.95
    l2_cache_size: int = 256


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
    """Encodes text with Jina v3, classifies with MLP head.

    The route() method is wrapped by a per-instance LRU cache keyed on
    sha256(query). Cache is per-process; reload the router instance to
    flush. functools.lru_cache is thread-safe on CPython.
    """

    def __init__(self, cfg: RouterConfig, weights_dir: Path):
        self._cfg = cfg
        self._encoder = None
        self._mlp = None
        self._domains: list[str] = []
        # L2 semantic cache: parallel ring of (embedding, route) pairs.
        # Set to None to disable. _l2_embs is a tensor on the encoder device
        # so the cosine similarity stays on GPU.
        self._l2_embs = None  # torch.Tensor [N, D] or None
        self._l2_routes: list[tuple[tuple[str, float], ...]] = []
        self._load(weights_dir)
        # Bind a fresh lru_cache per-instance so reloading flushes it.
        self._cached_route_by_hash = lru_cache(maxsize=_CACHE_MAXSIZE)(
            self._route_by_hash
        )
        # Warm caches at construction time. Eliminates the ~250 ms p95 spike
        # observed on first call to Jina v3 (LoRA-task lazy init). Cheap on
        # any encoder; ~150-300 ms one-time cost paid at boot, not per-query.
        try:
            self.prewarm()
        except Exception:
            log.exception("router prewarm failed (non-fatal)")

    def _load(self, weights_dir: Path) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from safetensors.torch import load_file

        meta_path = weights_dir / "meta.json"
        weights_path = weights_dir / "router.safetensors"

        meta = json.loads(meta_path.read_text())
        self._domains = meta["domains"]

        device = self._cfg.encoder_device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        log.info("Router encoder device: %s", device)
        self._encoder_device = device
        self._encoder = SentenceTransformer(
            self._cfg.embedding_model, device=device,
        )
        # Truncate inputs to bound per-query encode cost (attention is O(n²))
        try:
            self._encoder.max_seq_length = self._cfg.max_seq_length
        except Exception:
            pass
        self._mlp = _build_mlp(self._cfg)
        state = load_file(str(weights_path))
        # Remap keys: training saves "0.weight" but MLP expects "net.0.weight"
        if any(k.startswith("net.") for k in state):
            remapped = state
        else:
            remapped = {f"net.{k}": v for k, v in state.items()}
        self._mlp.load_state_dict(remapped)
        # Keep MLP on the same device as the encoder to avoid GPU→CPU copy
        # on every query (~0.3 ms saved per route).
        self._mlp.to(device)
        self._mlp.eval()
        log.info("Router loaded: %d domains, %s encoder", len(self._domains), self._cfg.embedding_model)

    def _compute_route(self, query: str) -> list[tuple[str, float]]:
        """Uncached embedding+MLP path. Also feeds the L2 semantic cache."""
        import torch

        with torch.no_grad():
            emb = self._encoder.encode(
                query, convert_to_tensor=True, normalize_embeddings=True,
            )

            # L2 semantic cache lookup BEFORE the MLP — saves the MLP forward
            # AND the unnecessary CPU transfer when a near-duplicate hits.
            if (
                self._cfg.l2_cache_threshold > 0
                and self._l2_embs is not None
                and self._l2_embs.shape[0] > 0
            ):
                sim = torch.matmul(self._l2_embs, emb)
                best = torch.argmax(sim).item()
                if sim[best].item() >= self._cfg.l2_cache_threshold:
                    _ROUTER_L2_HITS.inc()
                    return list(self._l2_routes[best])

            # MLP is co-located with the encoder; final scores moved to CPU
            # only for argsort/index ops below.
            scores = self._mlp(emb.unsqueeze(0)).squeeze(0).cpu()

        results: list[tuple[str, float]] = []
        for idx in torch.argsort(scores, descending=True):
            i = idx.item()
            s = scores[i].item()
            if s < self._cfg.threshold:
                break
            results.append((self._domains[i], s))
            if len(results) >= self._cfg.max_active:
                break

        # Push (emb, results) into the L2 ring (FIFO).
        if self._cfg.l2_cache_threshold > 0:
            self._l2_push(emb.detach(), tuple(results))

        return results

    def _l2_push(self, emb, route: tuple[tuple[str, float], ...]) -> None:
        """Append (emb, route) to L2 cache, evicting oldest when full."""
        import torch

        e = emb.unsqueeze(0)
        if self._l2_embs is None:
            self._l2_embs = e
            self._l2_routes = [route]
            return
        if self._l2_embs.shape[0] < self._cfg.l2_cache_size:
            self._l2_embs = torch.cat([self._l2_embs, e], dim=0)
            self._l2_routes.append(route)
        else:
            # Ring buffer: roll and overwrite oldest slot.
            self._l2_embs = torch.cat([self._l2_embs[1:], e], dim=0)
            self._l2_routes = self._l2_routes[1:] + [route]

    def _route_by_hash(self, _query_hash: str, query: str) -> tuple[tuple[str, float], ...]:
        """Cached helper — keyed on the sha256 hash to bound memory.

        Returns a tuple so it remains hashable/immutable inside lru_cache.
        """
        return tuple(self._compute_route(query))

    def route(self, query: str) -> list[tuple[str, float]]:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        info_before = self._cached_route_by_hash.cache_info()
        result = self._cached_route_by_hash(query_hash, query)
        info_after = self._cached_route_by_hash.cache_info()
        if info_after.hits > info_before.hits:
            _ROUTER_CACHE_HITS.inc()
        else:
            _ROUTER_CACHE_MISSES.inc()
        return list(result)

    def cache_info(self) -> dict[str, int]:
        """Expose LRU cache stats for observability."""
        info = self._cached_route_by_hash.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "currsize": info.currsize,
            "maxsize": info.maxsize,
        }

    def cache_clear(self) -> None:
        """Flush the L1 cache (useful after reloads or in tests)."""
        self._cached_route_by_hash.cache_clear()

    def prewarm(self, prompts: list[str] | None = None) -> int:
        """Populate the L1+L2 caches by routing each prompt once.

        With prompts=None, uses DEFAULT_WARMUP_PROMPTS to kill the cold-call
        p95 spike (Jina v3 LoRA-task lazy init costs ~250 ms on first call).
        Pass an explicit list to extend or replace.

        Returns the number of prompts processed.
        """
        if prompts is None:
            prompts = DEFAULT_WARMUP_PROMPTS
        n = 0
        for p in prompts:
            try:
                self.route(p)
                n += 1
            except Exception:  # pragma: no cover
                log.exception("prewarm failed for prompt: %r", p[:80])
        log.info("Router prewarmed on %d prompts (L1=%d, L2=%d)",
                 n, self._cached_route_by_hash.cache_info().currsize,
                 0 if self._l2_embs is None else self._l2_embs.shape[0])
        return n
