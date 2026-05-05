"""MLX model runtime with LoRA adapter hot-swap.

Loads a single MLX model and manages a pool of LoRA adapters.
Each worker process runs one instance of this runtime.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)


def _infer_lora_config(adapter_path: Path) -> dict | None:
    """Infer LoRA config from adapter_config.json when present, else from safetensors.

    Returns dict {rank, scale, dropout, keys, num_layers} or None if no LoRA
    weights found.
    """
    import json

    # Prefer adapter_config.json (written by mlx_lm training — contains the
    # authoritative scale = alpha/rank used during training).
    config_path = adapter_path.parent / "adapter_config.json"
    lora_params: dict = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text())
            lora_params = raw.get("lora_parameters", {})
        except Exception as exc:
            log.warning("Could not parse %s: %s", config_path, exc)

    try:
        weights = mx.load(str(adapter_path))
    except Exception as exc:
        log.warning("Could not load %s: %s", adapter_path, exc)
        return None
    rank: int | None = lora_params.get("rank")
    num_layers = 0
    keys: set[str] = set()
    layer_re = re.compile(r"(?:model\.)?layers\.(\d+)\.(.+)")
    for key, val in weights.items():
        if not key.endswith(".lora_a"):
            continue
        if rank is None and len(val.shape) >= 2:
            # MLX LoRA stores lora_a as (input_dim, rank).
            rank = int(val.shape[1])
        base = key[: -len(".lora_a")]
        m = layer_re.search(base)
        if m:
            num_layers = max(num_layers, int(m.group(1)) + 1)
            keys.add(m.group(2))
    if rank is None or num_layers == 0 or not keys:
        return None
    scale = lora_params.get("scale", rank)  # default: alpha == rank -> scale = 1.0
    dropout = lora_params.get("dropout", 0.0)
    return {
        "rank": rank,
        "scale": scale,
        "dropout": dropout,
        "keys": sorted(keys),
        "num_layers": num_layers,
    }


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
        # Note: HF prints a warning about Mistral regex / fix_mistral_regex=True
        # but enabling it via tokenizer_config breaks generation in practice
        # (verified 2026-05-05). Leave default tokenizer config.
        self._model, self._tokenizer = mlx_load(self._cfg.model_path)
        log.info("Model loaded")

    def preload_adapters(self) -> int:
        """Cache adapter paths + inferred LoRA config per domain.

        Weights are NOT loaded here — adapter wrap + load happens on apply()
        so the model can be hot-swapped between domains without keeping
        every adapter resident at once.
        """
        adapters_root = Path(self._cfg.adapters_dir)
        count = 0
        for domain in self._cfg.domains:
            adapter_path = adapters_root / domain / "adapters.safetensors"
            if not adapter_path.exists():
                log.warning("No adapter for domain: %s", domain)
                continue
            cfg = _infer_lora_config(adapter_path)
            if cfg is None:
                log.warning(
                    "Adapter %s: could not infer LoRA config (no lora_a keys?); "
                    "running on base model", domain,
                )
                continue
            self._adapter_cache[domain] = {
                "adapter_path": str(adapter_path),
                "lora_config": cfg,
            }
            count += 1
            log.info(
                "Indexed adapter: %s (rank=%d, layers=%d, keys=%d)",
                domain, cfg["rank"], cfg["num_layers"], len(cfg["keys"]),
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
        """Hot-swap LoRA adapter for the given domain.

        Wraps the base model with LoRA layers (linear_to_lora_layers) using
        the inferred config from preload_adapters, then loads adapter
        weights with strict=False so any extra/missing keys are tolerated.
        Calls remove_lora_layers between switches to keep the active set
        clean.
        """
        from mlx_lm.tuner.utils import linear_to_lora_layers, remove_lora_layers

        t0 = time.perf_counter()
        if domain == self._active_domain:
            return 0.0
        if self._active_domain is not None:
            try:
                remove_lora_layers(self._model)
            except Exception as exc:
                log.warning("remove_lora_layers failed: %s", exc)
            self._active_domain = None
        entry = self._adapter_cache.get(domain)
        if entry is not None:
            try:
                cfg = entry["lora_config"]
                linear_to_lora_layers(self._model, cfg["num_layers"], cfg)
                self._model.load_weights(entry["adapter_path"], strict=False)
                self._active_domain = domain
            except Exception as exc:
                log.warning(
                    "apply(%s) failed (%s); falling back to base model",
                    domain, exc,
                )
                try:
                    remove_lora_layers(self._model)
                except Exception:
                    pass
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
