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
        model_keys = {k for k, _ in nn.utils.tree_flatten(self._model.parameters())}
        count = 0
        for domain in self._cfg.domains:
            adapter_path = adapters_root / domain / "adapters.safetensors"
            if not adapter_path.exists():
                log.warning("No adapter for domain: %s", domain)
                continue
            raw = mx.load(str(adapter_path))
            weights = self._remap_adapter_keys(raw, model_keys)
            # Skip LoRA-only adapters: load_weights is strict; lora_a/lora_b keys
            # are not in the base model unless linear_to_lora_layers wrapped it
            # first (it doesn't here). Filter to keys present in the model.
            filtered = {k: v for k, v in weights.items() if k in model_keys}
            skipped = len(weights) - len(filtered)
            if not filtered:
                log.warning(
                    "Adapter %s: 0 matching keys (%d skipped LoRA params); "
                    "running on base model", domain, skipped,
                )
                continue
            mx.eval(filtered)
            self._adapter_cache[domain] = filtered
            count += 1
            log.info(
                "Preloaded adapter: %s (%d keys, %d skipped)",
                domain, len(filtered), skipped,
            )
        return count

    @staticmethod
    def _remap_adapter_keys(
        raw: dict, model_keys: set[str],
    ) -> dict:
        """Remap adapter weight keys to match the loaded MLX model.

        Handles prefix mismatches from different training pipelines:
        - 'language_model.model.layers.*' → 'model.layers.*'
        - 'model.layers.*' → 'layers.*'
        """
        # Try as-is first
        if any(k in model_keys or k.rsplit(".", 1)[0] + ".weight" in model_keys for k in raw):
            return dict(raw)

        # Try stripping common prefixes
        prefixes = ["language_model.model.", "language_model.", "model."]
        for prefix in prefixes:
            remapped = {}
            matched = False
            for k, v in raw.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix):]
                    remapped[new_k] = v
                    base = new_k.rsplit(".", 1)[0] + ".weight"
                    if base in model_keys:
                        matched = True
                else:
                    remapped[k] = v
            if matched:
                log.info("  Remapped keys: stripped prefix '%s'", prefix)
                return remapped

        return dict(raw)

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
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        from mlx_lm import generate as mlx_generate

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
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
