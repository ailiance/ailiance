"""MLX model runtime with LoRA adapter hot-swap.

Loads a single MLX model and manages a pool of LoRA adapters.
Each worker process runs one instance of this runtime.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    model_path: str
    adapters_dir: str
    domains: list[str]
    port: int = 9201
    precision: str = "bf16"
    memory_limit_gb: int = 460
    cache_limit_gb: int = 32


class MLXWorkerRuntime:
    """Manages one MLX model + N LoRA adapters."""

    def __init__(self, cfg: WorkerConfig):
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._adapter_cache: dict[str, dict] = {}
        self._active_domain: str | None = None

    def load_model(self) -> None:
        from mlx_lm import load as mlx_load

        mx.set_memory_limit(self._cfg.memory_limit_gb * 1024**3)
        mx.set_cache_limit(self._cfg.cache_limit_gb * 1024**3)

        log.info("Loading model from %s", self._cfg.model_path)
        self._model, self._tokenizer = mlx_load(self._cfg.model_path)
        log.info("Model loaded")

    def preload_adapters(self) -> int:
        adapters_root = Path(self._cfg.adapters_dir)
        count = 0
        for domain in self._cfg.domains:
            adapter_path = adapters_root / domain / "adapters.safetensors"
            if adapter_path.exists():
                weights = mx.load(str(adapter_path))
                mx.eval(weights)
                self._adapter_cache[domain] = weights
                count += 1
                log.info("Preloaded adapter: %s", domain)
            else:
                log.warning("No adapter for domain: %s", domain)
        return count

    def apply(self, domain: str) -> float:
        t0 = time.perf_counter()
        if domain == self._active_domain:
            return 0.0
        if self._active_domain is not None:
            self._unpatch()
        if domain in self._adapter_cache:
            self._model.load_weights(list(self._adapter_cache[domain].items()))
        self._active_domain = domain
        return time.perf_counter() - t0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        from mlx_lm import generate as mlx_generate

        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response, {"domain": self._active_domain}

    def _unpatch(self) -> None:
        self._active_domain = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
