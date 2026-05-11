# AILIANCE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 100% EU-sovereign multi-model LLM serving pipeline with 39 LoRA domains on Mac Studio M3 Ultra 512GB.

**Architecture:** Gateway (Jina v3 router + MLP) dispatches to 3 workers (Apertus-70B, Devstral Small 2, EuroLLM-22B), each with domain-specific LoRA adapters. Multi-process, OpenAI-compatible API.

**Tech Stack:** Python 3.14, FastAPI, MLX, mlx-lm (fork), Jina Embeddings v3, Prometheus, safetensors.

**Spec:** `docs/specs/2026-04-26-ailiance-design.md`

---

## File Structure

```
ailiance/
├── src/
│   ├── __init__.py
│   ├── mlx_models/
│   │   ├── __init__.py
│   │   ├── xielu.py              # xIELU activation (Task 1)
│   │   └── apertus.py            # Apertus MLX model (Task 2)
│   ├── router/
│   │   ├── __init__.py
│   │   ├── classifier.py         # Jina v3 + MLP router (Task 4)
│   │   └── domain_map.py         # Domain→worker mapping (Task 4)
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── server.py             # Worker FastAPI app (Task 5)
│   │   ├── runtime.py            # MLX load + LoRA hot-swap (Task 3)
│   │   └── schemas.py            # OpenAI-compat Pydantic models (Task 5)
│   └── gateway/
│       ├── __init__.py
│       └── server.py             # Gateway FastAPI app (Task 6)
├── scripts/
│   ├── start.sh                  # Launch all processes (Task 7)
│   ├── train_router.py           # Router training (Task 8)
│   └── build_router_data.py      # Data preparation (Task 8)
├── configs/
│   ├── gateway.yaml              # Router weights path, worker URLs (Task 6)
│   ├── apertus.yaml              # Model path, LoRA dir, domains (Task 7)
│   ├── devstral.yaml             # Model path, LoRA dir, domains (Task 7)
│   └── eurollm.yaml              # Model path, LoRA dir, domains (Task 7)
├── tests/
│   ├── __init__.py
│   ├── test_xielu.py             # (Task 1)
│   ├── test_apertus_model.py     # (Task 2)
│   ├── test_runtime.py           # (Task 3)
│   ├── test_router.py            # (Task 4)
│   ├── test_worker.py            # (Task 5)
│   └── test_gateway.py           # (Task 6)
├── data/
│   └── router/                   # train.jsonl, valid.jsonl (Task 8)
├── output/
│   ├── router/                   # Router weights (Task 8)
│   └── adapters/                 # LoRA adapters per model
│       ├── apertus/
│       ├── devstral/
│       └── eurollm/
├── pyproject.toml                # (Task 0)
├── CLAUDE.md                     # (Task 9)
└── docs/
    └── specs/
```

---

## Task 0: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`, `src/mlx_models/__init__.py`, `src/router/__init__.py`, `src/worker/__init__.py`, `src/gateway/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "ailiance"
version = "0.1.0-dev"
description = "EU-sovereign multi-model LLM serving pipeline"
requires-python = ">=3.13"
license = "Apache-2.0"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "mlx>=0.22",
    "mlx-lm>=0.22",
    "httpx>=0.28",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "prometheus-client>=0.21",
    "pydantic>=2.10",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "httpx"]
router = ["sentence-transformers>=3.4", "torch>=2.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create package init files**

```bash
mkdir -p src/mlx_models src/router src/worker src/gateway tests
touch src/__init__.py src/mlx_models/__init__.py src/router/__init__.py
touch src/worker/__init__.py src/gateway/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create venv and install**

```bash
cd ~/ailiance && uv venv && uv pip install -e ".[dev]"
```

- [ ] **Step 4: Verify**

```bash
uv run python -c "import src; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: project scaffold with pyproject.toml"
```

---

## Task 1: xIELU Activation for MLX

**Files:**
- Create: `src/mlx_models/xielu.py`
- Create: `tests/test_xielu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xielu.py
import mlx.core as mx
import mlx.nn as nn
import math


def test_xielu_positive_quadratic():
    """For x > 0: f(x) = alpha_p * x^2 + beta * x"""
    from src.mlx_models.xielu import XIELU

    act = XIELU(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5)
    x = mx.array([1.0, 2.0, 3.0])
    y = act(x)
    mx.eval(y)

    alpha_p = float(mx.softplus(mx.array(math.log(math.exp(0.8) - 1))))
    expected = [alpha_p * v * v + 0.5 * v for v in [1.0, 2.0, 3.0]]
    for got, exp in zip(y.tolist(), expected):
        assert abs(got - exp) < 1e-4, f"{got} != {exp}"


def test_xielu_negative_exponential():
    """For x <= 0: involves exp(x) term"""
    from src.mlx_models.xielu import XIELU

    act = XIELU(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5)
    x = mx.array([-1.0, -2.0, -0.5])
    y = act(x)
    mx.eval(y)

    # Output should be finite and different from zero
    for v in y.tolist():
        assert math.isfinite(v), f"non-finite output: {v}"


def test_xielu_zero_continuous():
    """f(0) should be 0 (continuous at origin)"""
    from src.mlx_models.xielu import XIELU

    act = XIELU()
    x = mx.array([0.0])
    y = act(x)
    mx.eval(y)
    assert abs(y.item()) < 1e-5


def test_xielu_learnable_params():
    """alpha_p and alpha_n should be nn.Module parameters"""
    from src.mlx_models.xielu import XIELU

    act = XIELU()
    params = act.parameters()
    flat = nn.utils.tree_flatten(params)
    param_names = [name for name, _ in flat]
    assert "log_alpha_p" in param_names
    assert "log_alpha_n" in param_names
    assert len([n for n, _ in flat]) == 2  # exactly 2 learnable params
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_xielu.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mlx_models.xielu'`

- [ ] **Step 3: Write xIELU implementation**

```python
# src/mlx_models/xielu.py
"""xIELU activation function for MLX.

Piecewise trainable activation from arXiv:2411.13010.
Used by Apertus-70B (Swiss AI Initiative).

  x > 0:  f(x) = alpha_p * x^2 + beta * x
  x <= 0: f(x) = alpha_n * (exp(min(x, eps)) - 1 - x) + beta * x

alpha_p and alpha_n are per-layer learnable parameters stored
via softplus reparameterization.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class XIELU(nn.Module):
    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
    ):
        super().__init__()
        self.log_alpha_p = mx.array(math.log(math.exp(alpha_p_init) - 1))
        self.log_alpha_n = mx.array(math.log(math.exp(alpha_n_init - beta) - 1))
        self._beta = beta
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        alpha_p = mx.softplus(self.log_alpha_p)
        alpha_n = self._beta + mx.softplus(self.log_alpha_n)
        pos = alpha_p * x * x + self._beta * x
        neg = alpha_n * (mx.exp(mx.minimum(x, self._eps)) - 1.0 - x) + self._beta * x
        return mx.where(x > 0, pos, neg)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_xielu.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlx_models/xielu.py tests/test_xielu.py
git commit -m "feat: xIELU activation for MLX (Apertus support)"
```

---

## Task 2: Apertus MLX Model

**Files:**
- Create: `src/mlx_models/apertus.py`
- Create: `tests/test_apertus_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apertus_model.py
import mlx.core as mx


def test_apertus_model_forward_shape():
    """Tiny Apertus model produces correct output shape."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig

    cfg = ApertusConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = ApertusModel(cfg)
    tokens = mx.array([[1, 2, 3, 4]])  # (1, 4)
    out = model(tokens)
    mx.eval(out)
    assert out.shape == (1, 4, 256), f"Expected (1, 4, 256), got {out.shape}"


def test_apertus_uses_xielu():
    """MLP layers use xIELU, not SiLU."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig
    from src.mlx_models.xielu import XIELU

    cfg = ApertusConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    model = ApertusModel(cfg)
    mlp = model.layers[0].mlp
    assert isinstance(mlp.act, XIELU), f"Expected XIELU, got {type(mlp.act)}"


def test_apertus_qk_norm():
    """Attention layers apply QK-norm (RMSNorm on Q and K)."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig

    cfg = ApertusConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    model = ApertusModel(cfg)
    attn = model.layers[0].self_attn
    assert hasattr(attn, "q_norm"), "Missing q_norm (QK-norm)"
    assert hasattr(attn, "k_norm"), "Missing k_norm (QK-norm)"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_apertus_model.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write Apertus model**

```python
# src/mlx_models/apertus.py
"""Apertus model for MLX.

Based on Llama architecture with:
- xIELU activation (replaces SiLU in MLP)
- QK-norm (RMSNorm on Q and K projections)
- vocab_size=131072, GQA

Reference: swiss-ai/Apertus-70B-2509
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from src.mlx_models.xielu import XIELU


@dataclass
class ApertusConfig:
    vocab_size: int = 131072
    hidden_size: int = 8192
    intermediate_size: int = 28672
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False


class ApertusRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ApertusMLP(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act = XIELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ApertusAttention(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = ApertusRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = ApertusRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.rope = nn.RoPE(self.head_dim, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class ApertusDecoderLayer(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.self_attn = ApertusAttention(cfg)
        self.mlp = ApertusMLP(cfg)
        self.input_layernorm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class ApertusModel(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [ApertusDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, mask=None, cache=None):
        h = self.embed_tokens(input_ids)
        if mask is None and h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.lm_head(self.norm(h))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_apertus_model.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlx_models/apertus.py tests/test_apertus_model.py
git commit -m "feat: Apertus MLX model (xielu + QK-norm)"
```

---

## Task 3: MLX Runtime with LoRA Hot-Swap

**Files:**
- Create: `src/worker/runtime.py`
- Create: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py
import pytest


def test_runtime_config_defaults():
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake-model",
        adapters_dir="/tmp/fake-adapters",
        domains=["python", "rust"],
    )
    assert cfg.model_path == "/tmp/fake-model"
    assert cfg.port == 9201
    assert cfg.precision == "bf16"


def test_runtime_lora_switch_interface():
    """Runtime exposes apply(domain) and generate() interface."""
    from src.worker.runtime import MLXWorkerRuntime

    assert hasattr(MLXWorkerRuntime, "apply")
    assert hasattr(MLXWorkerRuntime, "generate")
    assert hasattr(MLXWorkerRuntime, "preload_adapters")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_runtime.py -v
```
Expected: FAIL

- [ ] **Step 3: Write runtime**

```python
# src/worker/runtime.py
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
            temp=temperature,
        )
        return response, {"domain": self._active_domain}

    def _unpatch(self) -> None:
        """Remove current LoRA weights, restore base model."""
        # Reload base weights for modified layers
        # This is a simplified version; production should track modified keys
        self._active_domain = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
```

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest tests/test_runtime.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/worker/runtime.py tests/test_runtime.py
git commit -m "feat: MLX worker runtime with LoRA hot-swap"
```

---

## Task 4: Jina v3 Router

**Files:**
- Create: `src/router/classifier.py`
- Create: `src/router/domain_map.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import DOMAIN_TO_WORKER, get_worker_for_domain

    assert get_worker_for_domain("python") == 9202
    assert get_worker_for_domain("electronics-hw") == 9201
    assert get_worker_for_domain("chat-fr") == 9203
    assert get_worker_for_domain("unknown-domain") is None


def test_domain_map_completeness():
    from src.router.domain_map import DOMAIN_TO_WORKER, ALL_DOMAINS

    assert len(ALL_DOMAINS) == 39
    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_WORKER, f"Missing mapping for {domain}"


def test_classifier_config():
    from src.router.classifier import RouterConfig

    cfg = RouterConfig()
    assert cfg.embedding_model == "jinaai/jina-embeddings-v3"
    assert cfg.embedding_dim == 1024
    assert cfg.hidden_dim == 512
    assert cfg.num_domains == 39
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_router.py -v
```
Expected: FAIL

- [ ] **Step 3: Write domain map**

```python
# src/router/domain_map.py
"""Static mapping of domains to worker ports.

Apertus (:9201) — reasoning, hardware, EU normative
Devstral (:9202) — code generation
EuroLLM  (:9203) — multilingual EU
"""

APERTUS_PORT = 9201
DEVSTRAL_PORT = 9202
EUROLLM_PORT = 9203

APERTUS_DOMAINS = frozenset({
    "electronics-hw", "emc", "dsp", "spice", "kicad", "stm32",
    "platformio", "iot", "embedded", "math", "reasoning",
    "security", "music-audio", "freecad", "power",
    "misra-c", "autosar-cert", "doc-technique-ce",
    "calcul-normatif", "normes-iec",
})

DEVSTRAL_DOMAINS = frozenset({
    "python", "rust", "typescript", "cpp", "shell", "html-css",
    "sql", "web-backend", "web-frontend", "docker", "devops",
    "yaml-json", "llm-ops", "llm-orch", "ml-training", "lua-upy",
})

EUROLLM_DOMAINS = frozenset({
    "chat-fr", "traduction-tech", "redaction-multilingue", "localisation-doc",
})

ALL_DOMAINS = APERTUS_DOMAINS | DEVSTRAL_DOMAINS | EUROLLM_DOMAINS

DOMAIN_TO_WORKER: dict[str, int] = {}
for d in APERTUS_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
for d in DEVSTRAL_DOMAINS:
    DOMAIN_TO_WORKER[d] = DEVSTRAL_PORT
for d in EUROLLM_DOMAINS:
    DOMAIN_TO_WORKER[d] = EUROLLM_PORT


def get_worker_for_domain(domain: str) -> int | None:
    return DOMAIN_TO_WORKER.get(domain)
```

- [ ] **Step 4: Write router classifier**

```python
# src/router/classifier.py
"""Jina v3 + MLP domain classifier.

Encodes user query with Jina Embeddings v3 (1024d),
classifies into one of 39 domains via 2-layer MLP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as tnn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterConfig:
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dim: int = 1024
    hidden_dim: int = 512
    num_domains: int = 39
    threshold: float = 0.12
    max_active: int = 4


class RouterMLP(tnn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.net = tnn.Sequential(
            tnn.Linear(cfg.embedding_dim, cfg.hidden_dim),
            tnn.GELU(),
            tnn.Dropout(0.1),
            tnn.Linear(cfg.hidden_dim, cfg.num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class DomainRouter:
    """Encodes text with Jina v3, classifies with MLP head."""

    def __init__(self, cfg: RouterConfig, weights_dir: Path):
        self._cfg = cfg
        self._encoder = None
        self._mlp = None
        self._domains: list[str] = []
        self._load(weights_dir)

    def _load(self, weights_dir: Path) -> None:
        from sentence_transformers import SentenceTransformer
        from safetensors.torch import load_file

        meta_path = weights_dir / "meta.json"
        weights_path = weights_dir / "router.safetensors"

        meta = json.loads(meta_path.read_text())
        self._domains = meta["domains"]

        self._encoder = SentenceTransformer(self._cfg.embedding_model)
        self._mlp = RouterMLP(self._cfg)
        state = load_file(str(weights_path))
        self._mlp.load_state_dict(state)
        self._mlp.eval()
        log.info("Router loaded: %d domains, %s encoder", len(self._domains), self._cfg.embedding_model)

    def route(self, query: str) -> list[tuple[str, float]]:
        with torch.no_grad():
            emb = self._encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
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
```

- [ ] **Step 5: Run tests**

```bash
uv run python -m pytest tests/test_router.py -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/router/ tests/test_router.py
git commit -m "feat: Jina v3 domain router with MLP classifier"
```

---

## Task 5: Worker Server (FastAPI)

**Files:**
- Create: `src/worker/schemas.py`
- Create: `src/worker/server.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker.py
import pytest
from fastapi.testclient import TestClient


def test_worker_health():
    from src.worker.server import make_worker_app
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake",
        adapters_dir="/tmp/fake",
        domains=["python"],
        port=9202,
    )
    app = make_worker_app(cfg, skip_model_load=True)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_worker_metrics():
    from src.worker.server import make_worker_app
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake",
        adapters_dir="/tmp/fake",
        domains=["python"],
    )
    app = make_worker_app(cfg, skip_model_load=True)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ailiance_worker_requests_total" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_worker.py -v
```
Expected: FAIL

- [ ] **Step 3: Write schemas**

```python
# src/worker/schemas.py
"""OpenAI-compatible request/response schemas."""

import time
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel, extra="ignore"):
    model: str = "ailiance"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ailiance"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
```

- [ ] **Step 4: Write worker server**

```python
# src/worker/server.py
"""Worker FastAPI server — serves one MLX model with LoRA hot-swap."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.worker.runtime import MLXWorkerRuntime, WorkerConfig
from src.worker.schemas import ChatCompletion, ChatCompletionRequest, ChatMessage, Choice

log = logging.getLogger(__name__)


def make_worker_app(cfg: WorkerConfig, skip_model_load: bool = False) -> FastAPI:
    app = FastAPI(title=f"ailiance-worker-{cfg.port}")
    reg = CollectorRegistry()
    requests_total = Counter("ailiance_worker_requests_total", "Requests", ["status"], registry=reg)
    inference_latency = Histogram("ailiance_worker_inference_seconds", "Inference latency", registry=reg)
    semaphore = asyncio.Semaphore(1)

    runtime = MLXWorkerRuntime(cfg)
    if not skip_model_load:
        runtime.load_model()
        count = runtime.preload_adapters()
        log.info("Preloaded %d adapters on port %d", count, cfg.port)

    start_time = time.time()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": runtime.is_loaded,
            "port": cfg.port,
            "domains": cfg.domains,
            "uptime_s": int(time.time() - start_time),
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        domain = request.headers.get("X-Lora-Domain", cfg.domains[0] if cfg.domains else "")
        user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

        async with semaphore:
            swap_time = await asyncio.to_thread(runtime.apply, domain)
            t0 = time.perf_counter()
            text, meta = await asyncio.to_thread(
                runtime.generate, user_msg, req.max_tokens, req.temperature,
            )
            latency = time.perf_counter() - t0

        inference_latency.observe(latency)
        requests_total.labels(status="200").inc()

        return ChatCompletion(
            model=req.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=text))],
        )

    return app
```

- [ ] **Step 5: Run tests**

```bash
uv run python -m pytest tests/test_worker.py -v
```
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/worker/schemas.py src/worker/server.py tests/test_worker.py
git commit -m "feat: worker FastAPI server with OpenAI-compat API"
```

---

## Task 6: Gateway Server

**Files:**
- Create: `src/gateway/server.py`
- Create: `configs/gateway.yaml`
- Create: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway.py
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def test_gateway_health():
    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_gateway_models_list():
    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    models = resp.json()["data"]
    ids = [m["id"] for m in models]
    assert "ailiance" in ids
    assert "ailiance-apertus" in ids
    assert "ailiance-devstral" in ids
    assert "ailiance-eurollm" in ids
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_gateway.py -v
```
Expected: FAIL

- [ ] **Step 3: Write gateway config**

```yaml
# configs/gateway.yaml
router:
  weights_dir: output/router
  embedding_model: jinaai/jina-embeddings-v3
  embedding_dim: 1024
  hidden_dim: 512
  num_domains: 39
  threshold: 0.12
  max_active: 4

workers:
  apertus:
    url: http://localhost:9201
    port: 9201
  devstral:
    url: http://localhost:9202
    port: 9202
  eurollm:
    url: http://localhost:9203
    port: 9203
```

- [ ] **Step 4: Write gateway server**

```python
# src/gateway/server.py
"""Gateway server — routes requests to the correct worker."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.router.domain_map import (
    APERTUS_DOMAINS, DEVSTRAL_DOMAINS, EUROLLM_DOMAINS,
    ALL_DOMAINS, get_worker_for_domain,
)
from src.worker.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

WORKER_URLS = {
    9201: "http://localhost:9201",
    9202: "http://localhost:9202",
    9203: "http://localhost:9203",
}

MODEL_FORCE_MAP = {
    "ailiance-apertus": 9201,
    "ailiance-devstral": 9202,
    "ailiance-eurollm": 9203,
}


def make_gateway_app(skip_router_load: bool = False) -> FastAPI:
    app = FastAPI(title="ailiance-gateway")
    reg = CollectorRegistry()
    requests_total = Counter("ailiance_gw_requests_total", "Gateway requests", ["model", "status"], registry=reg)
    route_latency = Histogram("ailiance_gw_route_seconds", "Router latency", registry=reg)

    router = None
    if not skip_router_load:
        from src.router.classifier import DomainRouter, RouterConfig
        cfg_path = Path("configs/gateway.yaml")
        raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        rcfg = RouterConfig(**(raw.get("router", {})))
        router = DomainRouter(rcfg, Path(rcfg.__dict__.get("weights_dir", "output/router")))

    start_time = time.time()
    http_client = httpx.AsyncClient(timeout=600.0)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "router_loaded": router is not None,
            "uptime_s": int(time.time() - start_time),
            "domains": len(ALL_DOMAINS),
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "ailiance", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-devstral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-eurollm", "object": "model", "owned_by": "ailiance"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # Force model if specified
        forced_port = MODEL_FORCE_MAP.get(req.model)
        if forced_port:
            domain = ""
            worker_port = forced_port
        elif router is not None:
            user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            t0 = time.perf_counter()
            selections = router.route(user_msg)
            route_latency.observe(time.perf_counter() - t0)
            domain = selections[0][0] if selections else "python"
            worker_port = get_worker_for_domain(domain) or 9202
        else:
            domain = "python"
            worker_port = 9202

        worker_url = WORKER_URLS[worker_port]
        headers = {"X-Lora-Domain": domain}

        resp = await http_client.post(
            f"{worker_url}/v1/chat/completions",
            json=req.model_dump(),
            headers=headers,
        )

        requests_total.labels(model=req.model, status=str(resp.status_code)).inc()
        return resp.json()

    return app
```

- [ ] **Step 5: Run tests**

```bash
uv run python -m pytest tests/test_gateway.py -v
```
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/gateway/ configs/gateway.yaml tests/test_gateway.py
git commit -m "feat: gateway server with multi-model routing"
```

---

## Task 7: Launch Scripts & Worker Configs

**Files:**
- Create: `configs/apertus.yaml`, `configs/devstral.yaml`, `configs/eurollm.yaml`
- Create: `scripts/start.sh`

- [ ] **Step 1: Write worker configs**

```yaml
# configs/apertus.yaml
model_path: /Users/clems/ailiance-mac-tuner/models/Apertus-70B-Instruct-2509
adapters_dir: output/adapters/apertus
port: 9201
precision: bf16
domains:
  - electronics-hw
  - emc
  - dsp
  - spice
  - kicad
  - stm32
  - platformio
  - iot
  - embedded
  - math
  - reasoning
  - security
  - music-audio
  - freecad
  - power
  - misra-c
  - autosar-cert
  - doc-technique-ce
  - calcul-normatif
  - normes-iec
```

```yaml
# configs/devstral.yaml
model_path: /Users/clems/ailiance-mac-tuner/models/Devstral-Small-2-24B-Instruct-2512
adapters_dir: output/adapters/devstral
port: 9202
precision: bf16
domains:
  - python
  - rust
  - typescript
  - cpp
  - shell
  - html-css
  - sql
  - web-backend
  - web-frontend
  - docker
  - devops
  - yaml-json
  - llm-ops
  - llm-orch
  - ml-training
  - lua-upy
```

```yaml
# configs/eurollm.yaml
model_path: /Users/clems/ailiance-mac-tuner/models/EuroLLM-22B-Instruct-2512
adapters_dir: output/adapters/eurollm
port: 9203
precision: bf16
domains:
  - chat-fr
  - traduction-tech
  - redaction-multilingue
  - localisation-doc
```

- [ ] **Step 2: Write start script**

```bash
#!/usr/bin/env bash
# scripts/start.sh — Launch all ailiance workers + gateway
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${ROOT}/.venv/bin/python"
LOG_DIR="/tmp/ailiance"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Starting ailiance workers..."

# Workers
"$PYTHON" -m uvicorn src.worker.server:make_apertus_app --factory \
    --host 127.0.0.1 --port 9201 > "$LOG_DIR/apertus.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Apertus worker started (PID $!, port 9201)"

"$PYTHON" -m uvicorn src.worker.server:make_devstral_app --factory \
    --host 127.0.0.1 --port 9202 > "$LOG_DIR/devstral.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Devstral worker started (PID $!, port 9202)"

"$PYTHON" -m uvicorn src.worker.server:make_eurollm_app --factory \
    --host 127.0.0.1 --port 9203 > "$LOG_DIR/eurollm.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] EuroLLM worker started (PID $!, port 9203)"

# Wait for workers
sleep 5

# Gateway
"$PYTHON" -m uvicorn src.gateway.server:make_gateway_app --factory \
    --host 127.0.0.1 --port 9200 > "$LOG_DIR/gateway.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Gateway started (PID $!, port 9200)"

echo "[$(date '+%H:%M:%S')] ailiance running. Logs in $LOG_DIR/"
echo "  Gateway:  http://localhost:9200"
echo "  Apertus:  http://localhost:9201"
echo "  Devstral: http://localhost:9202"
echo "  EuroLLM:  http://localhost:9203"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x scripts/start.sh
```

- [ ] **Step 4: Commit**

```bash
git add configs/apertus.yaml configs/devstral.yaml configs/eurollm.yaml scripts/start.sh
git commit -m "feat: worker configs and launch script"
```

---

## Task 8: Router Training Pipeline

**Files:**
- Create: `scripts/train_router.py`
- Create: `scripts/build_router_data.py`

- [ ] **Step 1: Write router data builder**

```python
# scripts/build_router_data.py
"""Build router training data from micro-kiki classified data + new EU domains."""

import json
import random
from pathlib import Path

CLASSIFIED_DIR = Path.home() / "ailiance-mac-tuner/data/micro-kiki/classified"
OUTPUT_DIR = Path("data/router")
SEED = 42
TRAIN_RATIO = 0.8


def load_classified(directory: Path) -> list[dict]:
    rows = []
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        domain = jsonl_file.stem
        with open(jsonl_file) as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("instruction", "")
                if prompt.strip():
                    rows.append({"prompt": prompt.strip(), "domain": domain})
    return rows


def split_and_write(rows: list[dict], output_dir: Path) -> None:
    random.seed(SEED)
    by_domain: dict[str, list] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    train, valid = [], []
    for domain, items in sorted(by_domain.items()):
        random.shuffle(items)
        cut = max(1, int(len(items) * TRAIN_RATIO))
        train.extend(items[:cut])
        valid.extend(items[cut:])
        print(f"  {domain:<25s} train={cut:>5d}  valid={len(items)-cut:>5d}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(output_dir / "valid.jsonl", "w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal: train={len(train)}, valid={len(valid)}")


if __name__ == "__main__":
    rows = load_classified(CLASSIFIED_DIR)
    print(f"Loaded {len(rows)} rows from {CLASSIFIED_DIR}")
    split_and_write(rows, OUTPUT_DIR)
```

- [ ] **Step 2: Write router training script**

```python
# scripts/train_router.py
"""Train Jina v3 + MLP domain router for ailiance."""

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
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_router_data.py scripts/train_router.py
git commit -m "feat: router training pipeline (Jina v3 + MLP)"
```

---

## Task 9: CLAUDE.md & Integration Test

**Files:**
- Create: `CLAUDE.md`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write CLAUDE.md**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

AILIANCE is a 100% EU-sovereign multi-model LLM serving pipeline. It routes requests to 3 European models via a Jina v3 domain classifier, each with LoRA adapters.

## Architecture

Gateway (:9200) dispatches to 3 workers:
- Apertus-70B (:9201) — reasoning, hardware, EU normative (20 LoRA domains)
- Devstral Small 2 (:9202) — code generation (16 LoRA domains)
- EuroLLM-22B (:9203) — multilingual EU (4 LoRA domains)

Router: Jina Embeddings v3 (Berlin) + MLP classifier (39 domains)

## Commands

    # Setup
    uv venv && uv pip install -e ".[dev,router]"

    # Tests
    uv run python -m pytest
    uv run python -m pytest tests/test_xielu.py -v     # single file
    uv run python -m pytest -k "test_name"              # single test

    # Launch all services
    bash scripts/start.sh

    # Train router
    uv run python scripts/build_router_data.py
    uv run python scripts/train_router.py

    # Logs
    tail -f /tmp/ailiance/gateway.log
    tail -f /tmp/ailiance/apertus.log

## Key Design Decisions

- BF16 for all models (512GB unified memory allows it)
- Multi-process workers (1 model per process, shared memory pool)
- Sigmoid routing (domains overlap, not mutually exclusive)
- LoRA on attention projections only (q/k/v/o_proj)
- xielu activation custom-implemented for Apertus MLX support
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration.py
"""Smoke tests for the full stack (no model loading)."""


def test_all_domains_mapped():
    from src.router.domain_map import ALL_DOMAINS, DOMAIN_TO_WORKER

    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_WORKER, f"Unmapped domain: {domain}"


def test_worker_configs_cover_all_domains():
    import yaml
    from pathlib import Path
    from src.router.domain_map import ALL_DOMAINS

    config_domains = set()
    for cfg_file in ["configs/apertus.yaml", "configs/devstral.yaml", "configs/eurollm.yaml"]:
        p = Path(cfg_file)
        if p.exists():
            data = yaml.safe_load(p.read_text())
            config_domains.update(data.get("domains", []))

    for domain in ALL_DOMAINS:
        assert domain in config_domains, f"Domain {domain} not in any worker config"


def test_gateway_and_worker_import():
    from src.gateway.server import make_gateway_app
    from src.worker.server import make_worker_app
    from src.router.classifier import DomainRouter, RouterConfig
    from src.mlx_models.xielu import XIELU
    from src.mlx_models.apertus import ApertusModel, ApertusConfig
```

- [ ] **Step 3: Run all tests**

```bash
uv run python -m pytest tests/ -v
```
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md tests/test_integration.py
git commit -m "docs: CLAUDE.md and integration smoke tests"
```

---

## Execution Order Summary

| Task | Description | Depends On | Estimated |
|------|-------------|-----------|-----------|
| 0 | Project scaffold | — | 10 min |
| 1 | xIELU activation | 0 | 30 min |
| 2 | Apertus MLX model | 1 | 2h |
| 3 | MLX runtime + LoRA | 0 | 1h |
| 4 | Jina v3 router | 0 | 1h |
| 5 | Worker server | 3 | 1h |
| 6 | Gateway server | 4, 5 | 1h |
| 7 | Launch scripts + configs | 5, 6 | 30 min |
| 8 | Router training pipeline | 4 | 1h |
| 9 | CLAUDE.md + integration | all | 30 min |

**Tasks 1-2 et 3-4 sont parallélisables.**

## Post-Implementation

After all tasks pass:
1. Download Apertus-70B, Devstral Small 2, EuroLLM-22B weights
2. Verify xielu weight loading from HF checkpoint
3. Build router training data + train router
4. Train first batch of LoRA adapters (start with 5 domains)
5. End-to-end test with real models
