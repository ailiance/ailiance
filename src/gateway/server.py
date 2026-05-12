# src/gateway/server.py
"""Gateway server — routes requests to the correct worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import base64

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.gateway.file_extract import (
    ExtractError,
    ExtractResult,
    MAX_BYTES as FILE_MAX_BYTES,
    extract as extract_file,
)
from src.gateway.observability import track_chat
from src.orchestrator.chain_orchestrator import ChainOrchestrator
from src.orchestrator.chain_policy import ChainPolicy
from src.orchestrator.validators import StubValidator, make_validator
from src.router.domain_map import ALL_DOMAINS, get_worker_for_domain
from src.worker.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

_DEFAULT_WORKER_URLS = {
    9301: "http://localhost:9301",
    8502: "http://localhost:8502",  # ailiance / ailiance worker on macm1 (Gemma 4 E4B + LoRA)
    9303: "http://localhost:9303",
    9304: "http://localhost:9304",
    # Qwen3.6-35B-A3B MLX BF16 on Studio (mlx_lm.server :9305)
    9305: "http://localhost:9305",
    # Qwen3-Next 80B-A3B MoE on kxkm-ai (llama-server, alias 'qwen-32b-awq')
    # reached via the autossh tunnel listening on 0.0.0.0:8002.
    8002: "http://localhost:8002",
    # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai (llama-server :18889)
    # via autossh tunnel electron-server:8003.
    8003: "http://localhost:8003",
    # Tower Ollama :11434 (11 domain-specialized mascarade fine-tunes
    # + bge-m3 embed surface + qwen2.5-coder:3b) via autossh tunnel
    # electron-server:8004 → tower:11434. Set up the tunnel with:
    #   autossh -M 0 -N -L 0.0.0.0:8004:localhost:11434 \
    #       clems@100.78.6.122
    8004: "http://localhost:8004",
}


def _load_worker_urls() -> dict[int, str]:
    """Allow distributed deployments to override WORKER_URLS via env var.

    Set ``AILIANCE_WORKERS_JSON='{"9301":"http://studio:9301", ...}'`` to point
    each worker at a Tailscale/LAN address. Defaults stay localhost so a
    single-host setup just works.

    If the JSON is missing ports that are in _DEFAULT_WORKER_URLS, those
    ports are NOT probed nor reachable — silently breaking the matching
    ailiance-* aliases (they fall back to HEALTH_FALLBACK_PORT). When that
    drift is detected we log a warning so operators see it on startup
    instead of having to grep `model` field on every response. 2026-05-11
    incident: the systemd drop-in for electron-server omitted :8004 after
    the Tower Ollama wire-up shipped, silently routing all 13 mascarade
    aliases to Gemma.
    """
    raw = os.environ.get("AILIANCE_WORKERS_JSON")
    if not raw:
        return dict(_DEFAULT_WORKER_URLS)
    try:
        resolved = {int(k): str(v) for k, v in json.loads(raw).items()}
    except Exception as exc:
        log.warning(
            "failed to parse AILIANCE_WORKERS_JSON (%s); using defaults", exc,
        )
        return dict(_DEFAULT_WORKER_URLS)

    missing = set(_DEFAULT_WORKER_URLS) - set(resolved)
    if missing:
        # Don't reference HEALTH_FALLBACK_PORT here — that constant
        # is defined below `_load_worker_urls()` and this fn runs at
        # module load time, so the symbol isn't bound yet.
        log.warning(
            "AILIANCE_WORKERS_JSON omits %d default port(s): %s — "
            "aliases routed to those ports will fall back to the "
            "health-probe default (Gemma 9304). Update the systemd "
            "drop-in or unset the env var to use defaults.",
            len(missing),
            sorted(missing),
        )
    return resolved


WORKER_URLS = _load_worker_urls()


# Per-worker FIFO concurrency control — serialize requests to the same MLX
# worker to bound KV cache memory budget per worker. Studio M3 Ultra workers
# (Mistral-Medium 128B, Mixtral 8x22B, Qwen3-235B MoE, etc.) carry KV cache
# costs of tens of GB at 32k context; concurrent requests to a single worker
# can OOM the box. Toggle via GATEWAY_FIFO_ENABLED=false to disable.
_worker_locks: dict[str, asyncio.Lock] = {}
_worker_locks_meta_lock: asyncio.Lock | None = None


def _fifo_enabled() -> bool:
    return os.environ.get("GATEWAY_FIFO_ENABLED", "true").lower() not in (
        "0", "false", "no", "off",
    )


async def _acquire_worker_lock(worker_url: str) -> asyncio.Lock:
    """Return (creating if needed) the per-worker-URL asyncio.Lock.

    The meta-lock guards concurrent first-time creation. Subsequent lookups
    are dict reads (atomic in CPython) so the meta-lock is only contended on
    cold paths.
    """
    global _worker_locks_meta_lock
    if _worker_locks_meta_lock is None:
        _worker_locks_meta_lock = asyncio.Lock()
    async with _worker_locks_meta_lock:
        lock = _worker_locks.get(worker_url)
        if lock is None:
            lock = asyncio.Lock()
            _worker_locks[worker_url] = lock
        return lock


@asynccontextmanager
async def _worker_fifo(worker_url: str):
    """Async context manager that serializes requests per worker URL.

    No-op when GATEWAY_FIFO_ENABLED is false. Held for the full duration
    of the forwarded request, including streaming relay, so a streaming
    client cannot starve a follow-up request on the same worker.
    """
    if not _fifo_enabled():
        yield
        return
    lock = await _acquire_worker_lock(worker_url)
    async with lock:
        yield


# ---------------------------------------------------------------------------
# Cascade complexity-based selection (v0.4)
# ---------------------------------------------------------------------------
#
# Heuristic: when the auto-router classifies a prompt, optionally rewrite the
# target alias based on prompt complexity. Short / non-reasoning prompts get
# a fast small model; long / reasoning-heavy prompts get a flagship cascade.
# Disabled by default (env AILIANCE_CASCADE_ENABLED=1 to opt in) so legacy
# behaviour is preserved for production until the bench validates the
# heuristic on real traffic. Forced aliases (ailiance-mistral, …) are NEVER
# cascaded — they remain caller-chosen.

_REASONING_MARKERS = re.compile(
    r"\b(why|step by step|explain|prove|reason|analyze|compare|derive|"
    r"pourquoi|raison|étape|expliquer|démontrer|analyser|comparer)\b",
    re.IGNORECASE,
)
_CODE_MARKERS = re.compile(
    r"```|def\s+\w|class\s+\w|function\s+\w|import\s+\w",
)


def _complexity_estimate(prompt: str) -> str:
    """Return one of ``simple`` / ``medium`` / ``complex`` for a prompt.

    Pure heuristic — word count + lexical markers. Keep cheap so it runs
    on every request without measurable latency.
    """
    if not prompt:
        return "simple"
    n_words = len(prompt.split())
    has_reasoning = bool(_REASONING_MARKERS.search(prompt))
    has_code = bool(_CODE_MARKERS.search(prompt))
    if n_words < 15 and not has_reasoning and not has_code:
        return "simple"
    if n_words > 100 or has_reasoning:
        return "complex"
    return "medium"


# Cascade table: (complexity, domain) → alias override.
# A ``None`` (or absent key) means "no override". Domain ``_default`` applies
# when the classifier domain is not explicitly listed.
_CASCADE_OVERRIDES: dict[str, dict[str, str]] = {
    "simple": {
        # Tiny prompts → cheapest reachable worker: Tower Gemma 3 4B.
        "_default": "ailiance-gemma",
        "chat-fr": "ailiance-gemma",
        "math-gsm8k": "ailiance-gemma",
    },
    "complex": {
        # Reasoning-heavy → flagship cascade.
        "_default": "ailiance-mistral-medium",
        "chat-fr": "ailiance-mistral-medium",
        "math-reasoning": "ailiance-reasoning-r1",
        "llm-orch": "ailiance-mistral-medium",
    },
    # ``medium`` falls through to the router's pick.
}


def _cascade_pick(domain: str, prompt: str) -> str | None:
    """Return an alias override based on complexity, or None to keep default."""
    if os.environ.get("AILIANCE_CASCADE_ENABLED", "0") != "1":
        return None
    bucket = _complexity_estimate(prompt)
    table = _CASCADE_OVERRIDES.get(bucket) or {}
    return table.get(domain) or table.get("_default")


# Workers whose backend supports OpenAI native function calling (tool_calls
# JSON array in responses). Today only the kxkm-ai vLLM Qwen 32B native-FC
# worker on port 8002 qualifies; all MLX backends (Mistral-Medium-128B :9301,
# EuroLLM :9303, Gemma macm1 :8502, Studio MLX :9305/9323-9327) and llama.cpp
# servers (Gemma :9304, Granite :8003, Qwen-80B via :8002 tunnel when running
# llama-server rather than vLLM) either lack FC support entirely or emit
# hallucinated XML shapes that downstream parsers cannot dispatch. When a
# request carries tools[], FC_FORCE_ROUTE_PORT below pins the dispatch to
# this port regardless of which alias the caller picked. This override
# composes with the cascade above: if cascade picks a non-FC alias and
# tools[] is present, the force-route still wins.
FC_CAPABLE_PORTS: frozenset[int] = frozenset({8002})
FC_FORCE_ROUTE_PORT: int = 8002

# Effective context window each worker actually accepts. Source of truth: the
# launch flags of the worker process (llama.cpp --ctx-size, mlx_lm.server
# --max-tokens, Ollama Modelfile parameter num_ctx). Upstream Dirac defaults
# OpenAI-compatible clients to a 128k contextWindow when the model is not in
# its known-model table, which under-counts every backend in the parc and
# triggers the auto-condense / truncate path well before the real ceiling.
# CLI clients read this header off the response and override info.contextWindow
# accordingly. Numbers verified 2026-05-12 from live ps + worker launch args.
WORKER_CONTEXT_WINDOWS: dict[int, int] = {
    8002: 196608,   # llama-server Qwen3-Next-80B-A3B Q4_K_M (--ctx-size 196608)
    8003: 131072,   # llama-server Granite-4.1-30B Q4_K_M (n_ctx_train 131072)
    8004: 32768,    # Tower Ollama: Qwen3 4B Q4 mascarade fine-tunes (32k default)
    8502: 32768,    # macm1 mlx_lm.server multi-model (Ministral/Gemma/Qwen 32k)
    9301: 256000,   # Studio Mistral-Medium-3.5-128B-MLX-Q8
    9303: 131072,   # Studio EuroLLM-22B-Instruct-2512
    9304: 131072,   # Tower llama.cpp Gemma 3 4B IT
    9305: 131072,   # Studio Qwen3.6-35B-A3B-MLX-BF16
    9323: 131072,   # Studio DeepSeek-R1-Distill-Qwen-32B-MLX-4bit
    9324: 131072,   # Studio Llama-3.3-70B-Instruct-MLX-4bit
    9325: 131072,   # Studio Pixtral-12B-MLX-4bit (multimodal)
    9326: 32768,    # Studio Mistral-Small-3.1-24B-Instruct-MLX-4bit
    9327: 262144,   # Studio Qwen3-Coder-30B-A3B-Instruct-MLX-4bit (long ctx)
    9328: 131072,   # Studio Qwen3-235B-A22B-MLX-4bit (when running)
    9329: 65536,    # Studio Mixtral-8x22B-Instruct-MLX-4bit
    9330: 131072,   # Studio Devstral multi-LoRA base
}


def _worker_headers(
    worker_port: int,
    domain: str,
    response_body: dict | None = None,
    chain_policy: str | None = None,
) -> dict[str, str]:
    """Build the X-Ailiance-* headers exposing routing decisions.

    Lets CLI / cockpit / debugging tooling see which worker actually
    served the request without parsing the OpenAI-compatible body.

    - X-Ailiance-Worker-Port: the port the request landed on (after
      cascade override + FC force-route + chain dispatch).
    - X-Ailiance-Domain:      classifier top-1 domain, empty when the
      caller picked a forced alias (no router pass).
    - X-Ailiance-Backend:     upstream `system_fingerprint` from the
      worker (llama.cpp build hash, mlx version, fp_ollama, etc.).
      Only populated on non-streaming responses (streaming body is
      consumed lazily; emitting a header before reading it would
      either block or guess).
    - X-Ailiance-Upstream-Model: the model_id the worker reports
      (e.g. mascarade-kicad:latest, eu-kiki-gemma, /Users/...MLX-Q8).
      Distinct from the alias the caller asked for.
    - X-Ailiance-Chain:       chain policy that was engaged
      (direct / mixture / sequential / deliberate / validate), or
      empty if no chain orchestrator ran.
    """
    headers = {
        "X-Ailiance-Worker-Port": str(worker_port),
        "X-Ailiance-Domain": domain or "",
        "X-Ailiance-Chain": chain_policy or "",
    }
    ctx_window = WORKER_CONTEXT_WINDOWS.get(worker_port)
    if ctx_window:
        # Lets the CLI override its default modelInfo.contextWindow (128k
        # for unknown OpenAI-compatible backends) with the real ceiling
        # of the worker that served the response. Affects when the
        # auto-condense path triggers in subsequent turns of the same
        # task.
        headers["X-Ailiance-Context-Window"] = str(ctx_window)
    if response_body:
        fp = response_body.get("system_fingerprint")
        if fp:
            headers["X-Ailiance-Backend"] = str(fp)
        upstream = response_body.get("model")
        if upstream:
            headers["X-Ailiance-Upstream-Model"] = str(upstream)
    return headers


def _fc_force_route_enabled() -> bool:
    """Toggle for the tools[] -> qwen-32b-awq force-route.

    Default ON. Set GATEWAY_FC_FORCE_ROUTE=false to allow tools[] requests to
    follow normal routing (useful for testing FC support on new workers).
    """
    return os.environ.get("GATEWAY_FC_FORCE_ROUTE", "true").lower() not in (
        "0", "false", "no", "off",
    )


MODEL_FORCE_MAP = {
    "ailiance-mistral-medium": 9301,  # Mistral Medium 3.5 128B Q8 (studio:9301, renamed from ailiance-apertus 2026-05-11)
    "ailiance-mistral": 9301,  # alias for ailiance-mistral-medium (same backend)
    "ailiance-apertus": 9301,  # legacy alias preserved for backwards compatibility — routes to Mistral-Medium
    "ailiance-devstral": 8502,  # legacy alias — macm1 worker now serves Gemma 4
    "ailiance-gemma4": 8502,  # Gemma 4 E4B + ailiance curriculum LoRA (macm1)
    "ailiance-gemma": 9304,  # Gemma 3 4B IT on tower
    "ailiance-qwen": 8002,  # llama-server on kxkm-ai (RTX 4090) via autossh tunnel
    "ailiance-granite": 8003,  # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai
    "ailiance-qwen36": 9305,  # Qwen3.6-35B-A3B-MLX-BF16 on Studio (deeper specialist vs ailiance-qwen Q4)
    "ailiance-ministral": 8502,  # Ministral-3-14B-Instruct MLX 4-bit on macM1
    "ailiance-ministral-reasoning": 8502,  # Ministral-3-14B-Reasoning MLX 4-bit on macM1
    "ailiance-gemma2": 8502,  # Gemma-4-E2B-it MLX 4-bit on macM1 (lighter than E4B)
    # Devstral-Small-2-24B-MLX-4bit on Studio (:9316 base, :9317-9321 LoRA variants).
    "ailiance-devstral-base": 9316,
    "ailiance-python": 9330,
    "ailiance-cpp": 9330,
    "ailiance-rust-emb": 9330,
    "ailiance-html": 9330,
    "ailiance-ml-training": 9330,
    # Tower Ollama :11434 via tunnel :8004 — 11 domain-specialized
    # mascarade fine-tunes (Qwen3 4B Q4_K_M base, compiled as Ollama
    # Modelfile from KXKM-AI .safetensors LoRAs since 2026-04-12).
    "ailiance-kicad": 8004,
    "ailiance-spice": 8004,
    "ailiance-stm32": 8004,
    "ailiance-emc": 8004,
    "ailiance-embedded": 8004,
    "ailiance-platformio": 8004,
    "ailiance-freecad": 8004,
    "ailiance-dsp": 8004,
    "ailiance-iot": 8004,
    "ailiance-power": 8004,
    "ailiance-components-review": 8004,
    "ailiance-coder": 8004,  # mascarade-coder-v2 (Qwen2 1.5B Q4)
    "ailiance-embed": 8004,  # bge-m3 F16 — multilingual embedding
    # Studio multi-LoRA Apertus 70B custom server :9322 — one base model
    # in VRAM with adapters hot-swapped per request via load_adapters.
    "ailiance-apertus-real": 9322,
    "ailiance-apertus-electronics-hw": 9322,
    "ailiance-apertus-math-reasoning": 9322,
    "ailiance-apertus-math-gsm8k": 9322,
    "ailiance-apertus-math": 9322,
    "ailiance-apertus-security-fenrir": 9322,
    "ailiance-apertus-spice-sim": 9322,
    "ailiance-apertus-emc-dsp-power": 9322,
    "ailiance-apertus-embedded": 9322,
    # Studio flagship 2026-05-12 — Qwen3-235B-A22B-Instruct MoE 4-bit.
    "ailiance-flagship": 9328,
    "ailiance-qwen-235b": 9328,
    # Studio S3 additions 2026-05-12 — 5 MLX 4-bit workers on dedicated ports.
    "ailiance-reasoning-r1": 9323,  # DeepSeek-R1-Distill-Qwen-32B 4-bit
    "ailiance-llama": 9324,  # Llama-3.3-70B-Instruct 4-bit
    "ailiance-pixtral": 9325,  # Pixtral-12B 4-bit (vision-language)
    "ailiance-mistral-small": 9326,  # Mistral-Small-3.1-24B-Instruct 4-bit
    "ailiance-coder-pro": 9327,  # Qwen3-Coder-30B-A3B-Instruct 4-bit
    # Mixtral-8x22B-Instruct-v0.1 MLX 4-bit on Studio (~80GB). Worker
    # not yet running 2026-05-12: DL terminé, à relancer manuellement
    # une fois Qwen3-235B settled.
    "ailiance-mixtral": 9329,
}

# Per-port forward overrides for non-ailiance backends. The gateway rewrites
# the request body's `model` field and injects an Authorization header before
# proxying. Both pieces are sourced from env so secrets never land in source.


# Per-alias overrides keyed by the inbound `req.model`. Takes precedence over
# the per-port WORKER_FORWARD_OVERRIDES below. Lets multiple ailiance-* aliases
# share a single backend port (e.g. macM1 :8502 hosts Gemma E4B + Granite 30B
# + 2 Ministral 14B + Gemma E2B simultaneously, each selected by `model` body).
ALIAS_MODEL_REWRITES: dict[str, dict[str, str]] = {
    # macM1 mlx_lm.server :8502 - rewrite to actual HF model id loaded.
    "ailiance-gemma4": {"model": "lmstudio-community/gemma-4-E4B-it-MLX-4bit"},
    "ailiance-gemma2": {"model": "lmstudio-community/gemma-4-E2B-it-MLX-4bit"},
    "ailiance-ministral": {"model": "mlx-community/Ministral-3-14B-Instruct-2512-4bit"},
    "ailiance-ministral-reasoning": {"model": "mlx-community/Ministral-3-14B-Reasoning-2512-4bit"},
    # kxkm-ai llama-server :8003 (via tunnel) - alias is granite-30b, bearer key.
    "ailiance-granite": {"model": "granite-30b", "auth_env": "AILIANCE_QWEN_KEY"},
    # studio mlx_lm.server :9305 - rewrite to on-disk path the server has loaded.
    "ailiance-qwen36": {"model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-MLX-BF16"},
    # studio mlx_lm.server :9301 - rewrite to on-disk path the server has loaded
    # (mlx_lm.server resolves an unknown model field as an HF repo id, causing 404 + 60s timeout).
    "ailiance-mistral-medium": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8"},
    "ailiance-mistral": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8"},
    "ailiance-apertus": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8"},  # legacy alias
    # Tower Ollama :11434 via tunnel :8004 - Ollama needs the exact tag.
    "ailiance-kicad": {"model": "mascarade-kicad:latest"},
    "ailiance-spice": {"model": "mascarade-spice:latest"},
    "ailiance-stm32": {"model": "mascarade-stm32:latest"},
    "ailiance-emc": {"model": "mascarade-emc:latest"},
    "ailiance-embedded": {"model": "mascarade-embedded:latest"},
    "ailiance-platformio": {"model": "mascarade-platformio:latest"},
    "ailiance-freecad": {"model": "mascarade-freecad:latest"},
    "ailiance-dsp": {"model": "mascarade-dsp:latest"},
    "ailiance-iot": {"model": "mascarade-iot:latest"},
    "ailiance-power": {"model": "mascarade-power:latest"},
    "ailiance-components-review": {"model": "mascarade-components-review:latest"},
    "ailiance-coder": {"model": "mascarade-coder-v2:latest"},
    "ailiance-embed": {"model": "bge-m3:latest"},
    # Devstral-Small-2-24B MLX 4-bit on Studio. Server resolves model field
    # as on-disk path or HF repo id; pass the path the server has loaded.
    "ailiance-devstral-base": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-python": {"model": "devstral-python"},
    "ailiance-cpp": {"model": "devstral-cpp"},
    "ailiance-rust-emb": {"model": "devstral-rust-embedded"},
    "ailiance-html": {"model": "devstral-html-css"},
    "ailiance-ml-training": {"model": "devstral-ml-training"},
    # Studio multi-LoRA Apertus 70B custom server :9322. One base model in
    # VRAM, adapters swap per request via mlx_lm.tuner.utils.load_adapters
    # under an asyncio.Lock. Each alias rewrites the `model` body field to
    # the adapter_name the custom server expects.
    "ailiance-apertus-real": {"model": "apertus"},
    "ailiance-apertus-electronics-hw": {"model": "apertus-electronics-hw"},
    "ailiance-apertus-math-reasoning": {"model": "apertus-math-reasoning"},
    "ailiance-apertus-math-gsm8k": {"model": "apertus-math-gsm8k"},
    "ailiance-apertus-math": {"model": "apertus-math"},
    "ailiance-apertus-security-fenrir": {"model": "apertus-security-fenrir-curriculum"},
    "ailiance-apertus-spice-sim": {"model": "apertus-spice-sim"},
    "ailiance-apertus-emc-dsp-power": {"model": "apertus-emc-dsp-power-curriculum"},
    "ailiance-apertus-embedded": {"model": "apertus-embedded"},
    # Studio flagship 2026-05-12 — Qwen3-235B-A22B-Instruct MoE 4-bit (~120GB VRAM).
    "ailiance-flagship": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-235B-A22B-Instruct-MLX-4bit",
    },
    "ailiance-qwen-235b": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-235B-A22B-Instruct-MLX-4bit",
    },
    # Studio S3 additions 2026-05-12 — mlx_lm.server expects on-disk path.
    "ailiance-reasoning-r1": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/DeepSeek-R1-Distill-Qwen-32B-MLX-4bit",
    },
    "ailiance-llama": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Llama-3.3-70B-Instruct-MLX-4bit",
    },
    "ailiance-pixtral": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Pixtral-12B-MLX-4bit",
    },
    "ailiance-mistral-small": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Small-3.1-24B-Instruct-MLX-4bit",
    },
    "ailiance-coder-pro": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    },
    # Mixtral-8x22B 4-bit MLX on Studio :9329. Worker not yet running.
    "ailiance-mixtral": {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Mixtral-8x22B-Instruct-MLX-4bit",
    },
}


WORKER_FORWARD_OVERRIDES: dict[int, dict[str, str]] = {
    8002: {
        "model": "qwen-32b-awq",  # the alias llama-server expects
        "auth_env": "AILIANCE_QWEN_KEY",
    },
    # kxkm-ai llama.cpp :18889 served via tunnel :8003.
    8003: {
        "model": "granite-30b",
    },
    # mlx_lm.server resolves an unknown `model` field as a HF repo and tries to
    # download it; rewrite to the on-disk path the server already has loaded.
    9301: {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8",
    },
    # Tower llama.cpp :9304 served via Tailscale, model loaded with --alias eu-kiki-gemma.
    9304: {
        "model": "eu-kiki-gemma",
    },
    8502: {
        "model": "lmstudio-community/gemma-4-E4B-it-MLX-4bit",  # base model id loaded with curriculum LoRA adapter
    },
    # Tower Ollama :11434 via tunnel :8004 — port-level default model
    # used when no per-alias rewrite applies (should never happen in
    # practice since all 8004 aliases have ALIAS_MODEL_REWRITES).
    8004: {
        "model": "mascarade-generic:latest",  # Qwen2 3.1B Q4 fallback
    },
}



# Per-alias HA worker URL lists. When `req.model` matches a key here, the
# gateway picks a healthy URL (random.choice over healthy entries) instead
# of using the single WORKER_URLS[port] mapping. Backward compatible:
# aliases without an entry route via MODEL_FORCE_MAP + WORKER_URLS as
# before. Health: an entry is considered healthy when its parsed port is
# in `_healthy_ports`. If none healthy, the first URL is used as fallback.
ALIAS_WORKER_URLS: dict[str, list[str]] = {
    "ailiance-gemma4": [
        # macm1 mlx_lm.server :8502 via autossh tunnel (primary when up).
        "http://localhost:8502",
        # Studio mlx_lm.server :9334 - gemma-4-E4B-it + gemma4-eukiki LoRA
        # (HA mirror, also serves when macm1 is offline).
        "http://100.116.92.12:9334",
    ],
}


def _url_port(url: str) -> int | None:
    """Extract the TCP port from an http://host:port[/...] URL."""
    try:
        rest = url.split("://", 1)[1]
        host_port = rest.split("/", 1)[0]
        if ":" in host_port:
            return int(host_port.rsplit(":", 1)[1])
    except Exception:
        return None
    return None


def _pick_ha_url(alias: str) -> str | None:
    """Pick an HA worker URL for `alias`, preferring healthy entries.

    Returns None if alias has no ALIAS_WORKER_URLS entry (callers then
    fall back to the legacy WORKER_URLS[worker_port] lookup).
    """
    import random as _random
    urls = ALIAS_WORKER_URLS.get(alias)
    if not urls:
        return None
    healthy = [u for u in urls if (_url_port(u) or -1) in _healthy_ports]
    if healthy:
        return _random.choice(healthy)
    return urls[0]


# Liveness gating — populated by a background probe task started by the
# FastAPI lifespan. When a worker fails its /v1/models probe, requests
# routed to it fall back to the Gemma worker (:9304) before dispatch,
# avoiding the 33% Russian-roulette pattern where the router classifies
# code/ML prompts to a backend that's currently down.
_healthy_ports: set[int] = set(WORKER_URLS.keys())  # cold-start optimistic
_health_probe_task = None
HEALTH_PROBE_INTERVAL_S = 30.0
HEALTH_PROBE_TIMEOUT_S = 2.0
HEALTH_FALLBACK_PORT = 9304  # Gemma 3 4B on Tower — fast + reachable


async def _probe_workers(client: "httpx.AsyncClient") -> None:
    """Single round: probe every unique worker URL, update _healthy_ports."""
    import asyncio as _asyncio
    new_healthy: set[int] = set()
    async def probe(port: int, url: str) -> None:
        try:
            r = await client.get(f"{url}/v1/models", timeout=HEALTH_PROBE_TIMEOUT_S)
            if r.status_code < 500:
                new_healthy.add(port)
        except Exception:
            pass
    await _asyncio.gather(
        *(probe(port, url) for port, url in WORKER_URLS.items()),
        return_exceptions=True,
    )
    # Atomic swap.
    global _healthy_ports
    _healthy_ports = new_healthy or set(WORKER_URLS.keys())  # never empty


async def _health_probe_loop(client: "httpx.AsyncClient") -> None:
    import asyncio as _asyncio
    while True:
        try:
            await _probe_workers(client)
        except Exception as exc:
            log.warning("health probe round failed: %s", exc)
        await _asyncio.sleep(HEALTH_PROBE_INTERVAL_S)


def _gate_port(classified_port: int | None) -> int:
    """Return classified port if healthy, else fallback to Gemma (or any
    other healthy port if Gemma itself is down)."""
    if classified_port is not None and classified_port in _healthy_ports:
        return classified_port
    if HEALTH_FALLBACK_PORT in _healthy_ports:
        return HEALTH_FALLBACK_PORT
    if _healthy_ports:
        return next(iter(_healthy_ports))
    return classified_port or HEALTH_FALLBACK_PORT  # all dead — let it fail loud


# Strip the *whole* reasoning block. The chain-of-thought is internal and
# shouldn't appear in the final user-visible answer. Two passes:
#   1) Greedy-but-non-overlapping removal of complete blocks.
#   2) Cleanup of orphan opening/closing tags from truncated streams.
_THINK_BLOCK_RE = re.compile(
    r"\[THINK\].*?\[/THINK\]|<think>.*?</think>|<reasoning>.*?</reasoning>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_ORPHAN_RE = re.compile(
    r"\[/?THINK\]|</?think>|</?reasoning>", re.IGNORECASE
)

# Aliases registered in routing tables but NOT chat-capable. They are kept
# for /v1/models/details (introspection) but rejected at /v1/chat/completions
# with a 400 instead of leaking the underlying worker's "does not support
# chat" error body.
_BLOCKED_CHAT_ALIASES: frozenset[str] = frozenset({
    "ailiance-embed",  # bge-m3 — embedding model, no /v1/embeddings endpoint yet
})


# Per-alias default stop sequences for workers whose loaded chat template
# leaks the multi-turn prompt format (Vicuna ``USER:``/``ASSISTANT:``,
# Mistral inst tokens) past the assistant's reply. The worker keeps
# generating past the natural end-of-turn until max_tokens hits, with the
# extra tokens being a fabricated user turn that pollutes the UI.
#
# These defaults are merged with the caller's own ``stop`` (if any); user
# values are preserved and our defaults are appended. Empty list / no key
# = no injection.
_STOP_TOKEN_DEFAULTS: dict[str, tuple[str, ...]] = {
    # Pixtral 12B MLX 4-bit on Mac Studio :9325 — observed leaking
    # ``\nUSER:`` / ``USER:`` in prod chat completions.
    "ailiance-pixtral": ("\nUSER:", "USER:", "</s>", "[INST]"),
}


# Aliases backed by a vision-capable worker. Used by the multimodal
# auto-route override: when the request body carries any ``image_url``
# content and the caller is on the auto-router (``model == "ailiance"``),
# we transparently re-route to the canonical vision alias. Aliases the
# caller explicitly set are honoured verbatim — we never override an
# explicit non-vision choice.
_VISION_ALIASES: frozenset[str] = frozenset({"ailiance-pixtral"})
_CANONICAL_VISION_ALIAS = "ailiance-pixtral"


def _request_has_images(req) -> bool:
    """Return True iff any message in ``req`` carries an image part.

    The OpenAI multimodal spec sends mixed content as
    ``messages[].content = [{"type":"text", ...}, {"type":"image_url", ...}, ...]``.
    Plain text requests use a string for ``content`` and we shortcut.
    """
    for msg in getattr(req, "messages", []) or []:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                t = part.get("type") if isinstance(part, dict) else None
                if t in ("image_url", "image", "input_image"):
                    return True
    return False


def _inject_stop_tokens(body: dict, alias: str) -> None:
    """Merge :data:`_STOP_TOKEN_DEFAULTS` for ``alias`` into ``body['stop']``.

    Preserves the caller's explicit stop tokens (their values come first)
    and appends our defaults that aren't already present. Mutates the
    body in place. No-ops when no defaults are registered.
    """
    defaults = _STOP_TOKEN_DEFAULTS.get(alias)
    if not defaults:
        return
    user_stop = body.get("stop")
    # OpenAI spec: ``stop`` is either a string or a list of strings.
    if isinstance(user_stop, str):
        merged = [user_stop]
    elif isinstance(user_stop, list):
        merged = [s for s in user_stop if isinstance(s, str)]
    else:
        merged = []
    for tok in defaults:
        if tok not in merged:
            merged.append(tok)
    body["stop"] = merged


def _normalize_message_dict(msg: dict) -> None:
    """Apply reasoning→content promotion + tag stripping to one message dict.

    Mutates ``msg`` in place. Shared by the non-streaming response
    normalizer and the SSE stream normalizer (which operates on each
    chunk's ``delta``).
    """
    content = msg.get("content")
    reasoning = msg.get("reasoning")
    if (not content) and isinstance(reasoning, str) and reasoning.strip():
        msg["content"] = reasoning
        content = reasoning
    if isinstance(content, str) and ("THINK" in content.upper() or "<think" in content.lower()):
        stripped = _THINK_BLOCK_RE.sub("", content)
        stripped = _THINK_ORPHAN_RE.sub("", stripped)
        msg["content"] = stripped.strip()


async def _normalize_sse_stream(raw_stream):
    """Pipe and normalize an OpenAI-compatible SSE chat-completion stream.

    Buffers the byte stream until each ``data: …\\n\\n`` event is
    complete, then parses the JSON payload, applies
    :func:`_normalize_message_dict` to each ``choice.delta``, and yields
    the rewritten event back as ``bytes``. Non-JSON events (``data:
    [DONE]``, comments, keep-alives) and parse failures are passed
    through unchanged so we never break a working client over a bad
    rewrite.

    Limitations
    -----------
    Cross-event stripping of ``[THINK]…[/THINK]`` blocks is *not*
    performed — only blocks that fit inside a single SSE event are
    removed. Most workers emit the whole opening tag in one chunk and
    the closing tag arrives a few chunks later; in that intermediate
    window, tokens between the two tags will reach the client. The
    follow-up sweep to add a per-stream ``in_think`` state machine is
    tracked separately.
    """
    buf = ""
    async for raw in raw_stream:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Non-UTF8 bytes (e.g. binary): drop normalization,
                # passthrough so the client gets *something*.
                yield raw if isinstance(raw, bytes) else raw.encode("utf-8")
                continue
        buf += raw
        while "\n\n" in buf:
            event, _, buf = buf.partition("\n\n")
            yield _rewrite_sse_event(event) + b"\n\n"
    # Flush any trailing fragment (no terminating \n\n) untouched.
    if buf:
        yield buf.encode("utf-8")


def _rewrite_sse_event(event: str) -> bytes:
    """Rewrite a single SSE event, normalizing ``delta`` if it's chat JSON.

    Falls back to a byte-for-byte passthrough on any parse failure.
    """
    # SSE events may contain comment lines (':...') and multiple field
    # lines. We rewrite only ``data: <json>`` payloads.
    lines = event.split("\n")
    rewrote_any = False
    for i, line in enumerate(lines):
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        choices = obj.get("choices") if isinstance(obj, dict) else None
        if not isinstance(choices, list):
            continue
        mutated = False
        for choice in choices:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                before = (delta.get("content"), delta.get("reasoning"))
                _normalize_message_dict(delta)
                if (delta.get("content"), delta.get("reasoning")) != before:
                    mutated = True
            # Some workers emit a terminal frame with `message` instead
            # of `delta`. Normalize that too.
            msg = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(msg, dict):
                before = (msg.get("content"), msg.get("reasoning"))
                _normalize_message_dict(msg)
                if (msg.get("content"), msg.get("reasoning")) != before:
                    mutated = True
        if mutated:
            lines[i] = "data: " + json.dumps(obj, ensure_ascii=False)
            rewrote_any = True
    if rewrote_any:
        return "\n".join(lines).encode("utf-8")
    return event.encode("utf-8")


def _normalize_response_body(body: dict) -> dict:
    """Normalize worker chat-completion responses for OpenAI-spec clients.

    Two worker quirks are smoothed over here:

    1. MLX ``mlx_lm.server`` (>= 0.31.3) splits reasoning-model output into
       ``message.reasoning`` and leaves ``message.content`` empty. The OpenAI
       client in the Playground reads ``content`` only, so the user sees a
       blank reply. We copy ``reasoning`` into ``content`` when content is
       missing or empty.
    2. Some reasoning models (Ministral-3 Reasoning, R1 distills) emit
       ``[THINK]…[/THINK]`` or ``<think>…</think>`` literal tags inside
       ``content``. We strip the bare tags (keeping the inner text) so the
       UI doesn't render markup.
    """
    if not isinstance(body, dict):
        return body
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    for choice in choices:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(msg, dict):
            _normalize_message_dict(msg)
    return body


def make_gateway_app(skip_router_load: bool = False) -> FastAPI:
    app = FastAPI(title="ailiance-gateway")
    reg = CollectorRegistry()
    requests_total = Counter(
        "ailiance_gw_requests_total",
        "Gateway requests",
        # path: proxy (1-shot), chain (orchestrator), stream (SSE).
        # auto: 1 when chain engaged via auto-router (model=ailiance +
        # YAML deliberate), 0 for explicit opt-in or proxy/stream.
        ["model", "status", "path", "auto"],
        registry=reg,
    )
    route_latency = Histogram(
        "ailiance_gw_route_seconds",
        "Router latency",
        registry=reg,
    )

    router = None
    if not skip_router_load:
        import yaml
        from src.router.classifier import DomainRouter, RouterConfig

        cfg_path = Path("configs/gateway.yaml")
        raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        rcfg_dict = raw.get("router", {})
        rcfg = RouterConfig(
            **{k: v for k, v in rcfg_dict.items() if k in RouterConfig.__dataclass_fields__}
        )
        router = DomainRouter(rcfg, Path(rcfg_dict.get("weights_dir", "output/router")))

    # Expose on app.state so tests can inject a fake classifier without
    # touching the closure. Production handler reads from app.state.
    app.state.router = router

    start_time = time.time()
    http_client = httpx.AsyncClient(timeout=600.0)

    # v0.3 chain orchestrator — built lazily on first opt-in request so
    # the gateway boots even when configs are missing. Validator kind
    # is selected via AILIANCE_VALIDATOR (auto|iact_bench|stub):
    # "auto" prefers the real iact-bench Docker runner and falls back
    # to StubValidator if the submodule is missing or unimportable.
    app.state.orchestrator = None
    _validator_kind = os.environ.get("AILIANCE_VALIDATOR", "auto")
    app.state.orchestrator_validator = make_validator(_validator_kind)
    log.info(
        "gateway: validator backend = %s (kind=%s)",
        type(app.state.orchestrator_validator).__name__,
        _validator_kind,
    )

    def _build_orchestrator() -> ChainOrchestrator | None:
        if app.state.orchestrator is not None:
            return app.state.orchestrator
        policies = Path("configs/chain_policies.yaml")
        reflector = Path("configs/reflector_prompts.yaml")
        if not policies.exists() or not reflector.exists():
            log.warning(
                "chain orchestrator configs missing (policies=%s "
                "reflector=%s); chain_policy opt-in disabled",
                policies.exists(),
                reflector.exists(),
            )
            return None
        audit_dir = Path(os.environ.get("AILIANCE_AUDIT_DIR", "audit"))

        async def llm_call(messages, model: str) -> str:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": False,
            }
            # Reuse the same forward-rewrite logic so orchestrator-issued
            # calls hit the same upstream model id as direct calls.
            forced = MODEL_FORCE_MAP.get(model, HEALTH_FALLBACK_PORT)
            url = WORKER_URLS[_gate_port(forced)]
            override = ALIAS_MODEL_REWRITES.get(model) or (
                WORKER_FORWARD_OVERRIDES.get(forced)
            )
            hdrs = {}
            if override:
                if "model" in override:
                    payload["model"] = override["model"]
                auth_env = override.get("auth_env")
                if auth_env:
                    key = os.environ.get(auth_env, "")
                    if key:
                        hdrs["Authorization"] = f"Bearer {key}"
            r = await http_client.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers=hdrs,
            )
            data = r.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        app.state.orchestrator = ChainOrchestrator(
            policies_path=policies,
            reflector_path=reflector,
            validator=app.state.orchestrator_validator,
            llm_call=llm_call,
            audit_dir=audit_dir,
        )
        return app.state.orchestrator

    @app.on_event("startup")
    async def _start_health_probe() -> None:
        import asyncio as _asyncio
        global _health_probe_task
        # Run one immediate probe so first-request decisions aren't blind.
        try:
            await _probe_workers(http_client)
        except Exception:
            pass
        _health_probe_task = _asyncio.create_task(_health_probe_loop(http_client))
        log.info("health probe started (interval=%ss, healthy=%s)",
                 HEALTH_PROBE_INTERVAL_S, sorted(_healthy_ports))

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
                {"id": "ailiance-mistral-medium", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-mistral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma4", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-qwen", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-granite", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-qwen36", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-ministral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-ministral-reasoning", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma2", "object": "model", "owned_by": "ailiance"},
                # Tower Ollama :11434 via tunnel :8004 — mascarade fine-tunes
                {"id": "ailiance-kicad", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-spice", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-stm32", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-emc", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-embedded", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-platformio", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-freecad", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-dsp", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-iot", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-power", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-components-review", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-coder", "object": "model", "owned_by": "ailiance"},
                # ailiance-embed (bge-m3) is an embedding model, not chat —
                # intentionally omitted from /v1/models (no /v1/embeddings
                # endpoint yet). See _BLOCKED_CHAT_ALIASES below.
                # Devstral 24B 4-bit MLX + 5 LoRAs on Studio :9316-9321
                {"id": "ailiance-devstral-base", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-python", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-cpp", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-rust-emb", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-html", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-ml-training", "object": "model", "owned_by": "ailiance"},
                # Apertus 70B 4-bit MLX multi-LoRA on Studio :9322
                {"id": "ailiance-apertus-real", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-electronics-hw", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-math-reasoning", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-math-gsm8k", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-math", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-security-fenrir", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-spice-sim", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-emc-dsp-power", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-apertus-embedded", "object": "model", "owned_by": "ailiance"},
                # Studio flagship 2026-05-12 — Qwen3-235B-A22B MoE 4-bit
                {"id": "ailiance-flagship", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-qwen-235b", "object": "model", "owned_by": "ailiance"},
                # Studio S3 additions 2026-05-12 — DeepSeek + Llama + Pixtral + Mistral-Small + Qwen3-Coder
                {"id": "ailiance-reasoning-r1", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-llama", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-pixtral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-mistral-small", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-coder-pro", "object": "model", "owned_by": "ailiance"},
                # Studio Mixtral-8x22B 4-bit MLX :9329 (worker offline 2026-05-12)
                {"id": "ailiance-mixtral", "object": "model", "owned_by": "ailiance"},
            ],
        }

    @app.get("/v1/models/details")
    def list_models_details() -> dict:
        """Enriched model listing with display metadata.

        Reads `configs/models-display.yaml` on each call so descriptions
        can be edited without a gateway restart. The minimal /v1/models
        endpoint stays OpenAI-standard for plain clients.
        """
        import yaml as _yaml

        path = Path("configs/models-display.yaml")
        try:
            raw = _yaml.safe_load(path.read_text()) if path.exists() else {}
        except Exception as exc:
            log.warning("models-display.yaml parse failed: %s", exc)
            raw = {}
        models = raw.get("models", {}) if isinstance(raw, dict) else {}
        # Enumerate the same id list as /v1/models so they stay aligned.
        ids = [
            "ailiance",
            "ailiance-mistral-medium",
            "ailiance-mistral",
            "ailiance-gemma4",
            "ailiance-gemma",
            "ailiance-qwen",
            "ailiance-granite",
            "ailiance-qwen36",
            "ailiance-ministral",
            "ailiance-ministral-reasoning",
            "ailiance-gemma2",
            "ailiance-kicad",
            "ailiance-spice",
            "ailiance-stm32",
            "ailiance-emc",
            "ailiance-embedded",
            "ailiance-platformio",
            "ailiance-freecad",
            "ailiance-dsp",
            "ailiance-iot",
            "ailiance-power",
            "ailiance-components-review",
            "ailiance-coder",
            "ailiance-embed",
        ]
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "owned_by": "ailiance",
                    **(models.get(mid) or {}),
                }
                for mid in ids
            ],
        }

    @app.post("/v1/files/extract")
    async def extract_endpoint(
        file: UploadFile | None = File(default=None),
        filename: str | None = Form(default=None),
        mime: str | None = Form(default=None),
    ) -> dict:
        """Extract markdown text from a PDF / docx / xlsx / pptx / html / txt file.

        Two upload modes:

        * **multipart/form-data** with a ``file`` field (preferred for
          browser uploads). ``filename`` and ``mime`` are taken from the
          ``UploadFile`` itself but can be overridden via the optional
          form fields when the browser fails to set them.
        * No body is accepted other than multipart in this version; a
          JSON-with-base64 mode can be added later if a client needs it.

        Returns ``{"markdown": "...", "format": "pdf", "metadata": {...}}``.
        Errors translate to HTTP 400 with a structured detail.
        """
        if file is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "invalid_request_error",
                    "code": "missing_file",
                    "message": "Expected multipart upload with a 'file' field.",
                },
            )
        data = await file.read()
        await file.close()
        eff_filename = filename or file.filename
        eff_mime = mime or file.content_type
        try:
            result: ExtractResult = extract_file(
                data, filename=eff_filename, mime=eff_mime
            )
        except ExtractError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "invalid_request_error",
                    "code": exc.code,
                    "message": exc.message,
                },
            ) from exc
        return {
            "markdown": result.markdown,
            "format": result.format,
            "metadata": result.metadata,
            "filename": eff_filename,
            "max_bytes": FILE_MAX_BYTES,
        }

    @app.post("/v1/route")
    def route_only(payload: dict) -> dict:
        """Read-only routing decision for a prompt — no chat side-effect.

        Body: {"prompt": "..."}
        Returns: {"router_loaded", "selections": [{"domain","score"}], "chosen_domain", "chosen_port"}
        """
        prompt = (payload or {}).get("prompt", "")
        if router is None:
            return {"router_loaded": False, "selections": [], "chosen_port": 9304}
        selections = router.route(prompt) if prompt else []
        chosen_domain = selections[0][0] if selections else None
        chosen_port = (
            get_worker_for_domain(chosen_domain) if chosen_domain else None
        ) or 9304
        return {
            "router_loaded": True,
            "selections": [
                {"domain": d, "score": float(s)} for d, s in selections[:5]
            ],
            "chosen_domain": chosen_domain,
            "chosen_port": chosen_port,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        _trace_started_at = time.perf_counter()
        _trace_req_body = req.model_dump(exclude_none=True)
        if req.model in _BLOCKED_CHAT_ALIASES:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "invalid_request_error",
                    "message": (
                        f"Model '{req.model}' is an embedding model and does "
                        "not support chat completions. Use a chat-capable "
                        "alias (e.g. 'ailiance' for auto-routing)."
                    ),
                },
            )

        # Multimodal auto-route: when the caller is on the auto-router
        # alias and the request body carries an image (or any non-text
        # block), transparently redirect to the canonical vision alias.
        # An explicit non-vision choice from the caller is respected —
        # they'll get whatever error the worker raises, never a silent
        # rewrite that hides their intent.
        _multimodal_routed = False
        if req.model == "ailiance" and _request_has_images(req):
            req.model = _CANONICAL_VISION_ALIAS
            _multimodal_routed = True

        forced_port = MODEL_FORCE_MAP.get(req.model)
        active_router = app.state.router
        if forced_port:
            domain = ""
            worker_port = _gate_port(forced_port)
        elif active_router is not None:
            user_msg = next(
                (m.content for m in reversed(req.messages) if m.role == "user"), ""
            )
            t0 = time.perf_counter()
            selections = active_router.route(user_msg)
            route_latency.observe(time.perf_counter() - t0)
            domain = selections[0][0] if selections else "python"
            # Fallback to Gemma (9304): reachable + fast for unmapped domains.
            worker_port = get_worker_for_domain(domain) or 9304
            # Health gate: if the classified worker is currently down, fall
            # back to a healthy worker so prompts don't 500 just because the
            # router happened to classify them to a temporarily dead backend.
            worker_port = _gate_port(worker_port)
        else:
            # No router loaded → default to Gemma (fast, reachable).
            domain = "general"
            worker_port = _gate_port(9304)

        # ------------------------------------------------------------------
        # Cascade complexity-based override (v0.4).
        # Only applies on the auto-router path (req.model == "ailiance" /
        # no forced port). When AILIANCE_CASCADE_ENABLED=1, short prompts
        # are rerouted to a fast small model and reasoning-heavy prompts
        # are escalated to a flagship. Forced aliases are NEVER cascaded.
        # ------------------------------------------------------------------
        cascade_alias: str | None = None
        if not forced_port and domain:
            user_msg_for_cascade = next(
                (m.content for m in reversed(req.messages) if m.role == "user"),
                "",
            ) or ""
            cascade_alias = _cascade_pick(domain, user_msg_for_cascade)
            if cascade_alias:
                cascade_port = MODEL_FORCE_MAP.get(cascade_alias)
                if cascade_port is not None:
                    worker_port = _gate_port(cascade_port)
                    log.info(
                        "cascade: domain=%s complexity-override → %s (port %d)",
                        domain, cascade_alias, worker_port,
                    )

        # ------------------------------------------------------------------
        # Native function-calling force-route. Composes with the cascade
        # above: if cascade picked a non-FC alias and the caller shipped
        # tools[], this still wins. The kxkm-ai vLLM Qwen 32B worker on
        # 8002 is the only backend in the parc that serves tool_calls as a
        # structured JSON array; MLX and llama.cpp backends either lack FC
        # or hallucinate XML shapes downstream parsers cannot dispatch
        # (observed 2026-05-12: ailiance-agent CLI v0.6.0-beta hit a
        # 5-retry storm because Mistral-Medium-128B received tools[] via
        # the auto-router and emitted <function=NAME>...</function> blocks
        # the client could not dispatch). Set GATEWAY_FC_FORCE_ROUTE=false
        # to disable when testing FC support on a new worker.
        # ------------------------------------------------------------------
        if (
            req.tools
            and _fc_force_route_enabled()
            and worker_port not in FC_CAPABLE_PORTS
        ):
            log.info(
                "fc-force-route: model=%s tools=%d redirect %d -> %d",
                req.model, len(req.tools), worker_port, FC_FORCE_ROUTE_PORT,
            )
            forced_port = FC_FORCE_ROUTE_PORT
            domain = ""
            worker_port = _gate_port(FC_FORCE_ROUTE_PORT)

        # Router v0.3 dispatch resolution. Two ways to engage the chain:
        #   (1) explicit  — extra_body.chain_policy from any request,
        #   (2) automatic — req.model == "ailiance" (router-driven, no
        #       MODEL_FORCE_MAP entry), policy taken from the YAML map
        #       per the classified domain. Forced aliases stay DIRECT
        #       unless extra_body.chain_policy is set, so a caller
        #       picking ailiance-mistral keeps the OpenAI-style "I know
        #       what I want" 1-shot semantics.
        # Streaming + non-direct is unsupported in v0.3.0: explicit
        # opt-in returns 400 (the user asked for the impossible);
        # auto-engagement silently degrades to DIRECT (the user did
        # not opt in, so we MUST NOT break their stream).
        # Note: extra_body.chain_policy="direct" intentionally falls
        # through both branches below. Explicit DIRECT is a caller
        # saying "no chain even on the auto-router alias" — it must
        # bypass the orchestrator and reach the legacy proxy.
        extra = req.extra_body or {}
        chain_policy_raw = extra.get("chain_policy")
        policy: ChainPolicy | None = None
        auto_engaged = False
        explicit_override: ChainPolicy | None = None
        cached_orch: ChainOrchestrator | None = None

        if chain_policy_raw and chain_policy_raw != ChainPolicy.DIRECT.value:
            try:
                explicit_override = ChainPolicy(chain_policy_raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "type": "invalid_request",
                        "message": (
                            f"unknown chain_policy {chain_policy_raw!r}; "
                            f"valid: {[p.value for p in ChainPolicy]}"
                        ),
                    },
                ) from exc
            if req.stream:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "type": "invalid_request",
                        "message": (
                            "stream=true is not supported with "
                            "chain_policy != direct in v0.3.0"
                        ),
                    },
                )
            policy = explicit_override
        elif (
            not forced_port
            and not req.stream
            and chain_policy_raw is None
            and domain
        ):
            # Auto-router: look up the YAML default for the classified
            # domain. Engage only if it is non-DIRECT. `domain` is the
            # classifier's top-1 string; an empty/falsy value (rare but
            # possible if the classifier returns no selections) skips
            # auto-engagement and falls through to the legacy proxy.
            cached_orch = _build_orchestrator()
            if cached_orch is not None:
                yaml_policy, _ = cached_orch.policy_for_domain(domain)
                if yaml_policy != ChainPolicy.DIRECT:
                    policy = yaml_policy
                    auto_engaged = True

        if policy is not None:
            orch = cached_orch or _build_orchestrator()
            if orch is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "type": "orchestrator_unavailable",
                        "message": "chain configs missing on server",
                    },
                )

            user_msg = next(
                (m.content for m in reversed(req.messages) if m.role == "user"),
                "",
            ) or ""
            include_audit = bool(extra.get("include_audit", False))
            # extra_body.max_retries is documented in the API contract;
            # forward it through so callers can override the per-domain
            # policy default for DELIBERATE chains. None = honour YAML.
            raw_retries = extra.get("max_retries")
            try:
                max_retries_override: int | None = (
                    int(raw_retries) if raw_retries is not None else None
                )
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "type": "invalid_request",
                        "message": (
                            "extra_body.max_retries must be an integer"
                        ),
                    },
                ) from None
            # Auto-engaged path: pass override_policy=None so the YAML
            # entry stays the source of truth (max_retries from YAML
            # too). Explicit opt-in path: pass the override.
            chain_result = await orch.execute(
                user_msg,
                domain=domain or "_default",
                model=req.model,
                override_policy=explicit_override,
                max_retries=max_retries_override,
            )
            requests_total.labels(
                model=req.model,
                status="200",
                path="chain",
                auto="1" if auto_engaged else "0",
            ).inc()
            response: dict = {
                "id": f"chatcmpl-{chain_result.chain_id[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": chain_result.final_output,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "ailiance_chain": {
                    "chain_id": chain_result.chain_id,
                    "policy": chain_result.policy.value,
                    "auto_engaged": auto_engaged,
                    "status": chain_result.status,
                    "domain": chain_result.domain,
                },
            }
            if include_audit:
                response["audit_trace"] = [
                    {
                        "kind": s.kind,
                        "attempt": s.attempt,
                        "success": s.success,
                        "exit_code": s.payload.get("exit_code"),
                        "duration_s": s.duration_s,
                    }
                    for s in chain_result.steps
                ]
            track_chat(
                model_alias=req.model,
                domain=domain,
                kind="chain",
                request_body=_trace_req_body,
                response_body=response,
                started_at=_trace_started_at,
                chain_id=chain_result.chain_id,
            )
            return JSONResponse(
                content=response,
                headers=_worker_headers(
                    worker_port,
                    domain,
                    response,
                    chain_policy=chain_result.policy.value,
                ),
            )

        # ALIAS_WORKER_URLS: if alias has HA list, pick healthy URL
        # instead of using the single WORKER_URLS[worker_port] entry.
        ha_url = _pick_ha_url(req.model)
        worker_url = ha_url if ha_url is not None else WORKER_URLS[worker_port]
        headers = {"X-Lora-Domain": domain}
        # exclude_none so optional ChatMessage fields (tool_calls, name,
        # tool_call_id) don't reach llama.cpp workers as `null` and trip
        # their JSON schema validation.
        body = req.model_dump(exclude_none=True)
        # extra_body is gateway-only metadata — strip before forwarding.
        body.pop("extra_body", None)
        # Inject per-alias stop tokens for workers whose chat template
        # leaks past end-of-turn (e.g. Pixtral fabricating ``USER:`` turns).
        _inject_stop_tokens(body, req.model)

        # Forward rewrites: per-alias takes precedence over per-port. Lets a
        # single backend port host multiple ailiance-* aliases each rewritten
        # to a distinct upstream model id.
        # Cascade: when the v0.4 complexity heuristic remapped this request,
        # honour the cascade alias's rewrite (e.g. mlx_lm.server :8502 needs
        # the on-disk path the cascade target loaded).
        rewrite_key = cascade_alias if cascade_alias else req.model
        override = ALIAS_MODEL_REWRITES.get(rewrite_key) or WORKER_FORWARD_OVERRIDES.get(worker_port)
        if override:
            if "model" in override:
                body["model"] = override["model"]
            auth_env = override.get("auth_env")
            if auth_env:
                key = os.environ.get(auth_env, "")
                if key:
                    headers["Authorization"] = f"Bearer {key}"
        elif req.model == "ailiance":
            # Auto-router fallback: when no override matched the chosen worker,
            # the workers (mlx_lm.server, llama.cpp) don't know what "ailiance"
            # means and treat it as a HF repo / on-disk path → 404 with
            # "ailiance/config.json: No such file or directory". Strip the
            # model field so the worker uses its loaded default model.
            body.pop("model", None)

        # Per-worker FIFO: serialize requests to this worker URL to bound
        # KV cache pressure. Lock is held for the entire forward including
        # streaming relay (see _worker_fifo docstring).
        fifo_cm = _worker_fifo(worker_url)
        await fifo_cm.__aenter__()
        try:
            # Streaming path: pipe SSE chunks back to the client without buffering.
            if body.get("stream"):
                req_stream = http_client.build_request(
                    "POST",
                    f"{worker_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                worker_resp = await http_client.send(req_stream, stream=True)
                requests_total.labels(
                    model=req.model,
                    status=str(worker_resp.status_code),
                    path="stream",
                    auto="0",
                ).inc()

                # Capture for closure so the lock is released once the
                # stream is fully consumed (or the client disconnects).
                _release_cm = fifo_cm
                fifo_cm = None  # prevent finally-block double-release

                async def relay() -> "object":
                    try:
                        async for chunk in _normalize_sse_stream(
                            worker_resp.aiter_text()
                        ):
                            yield chunk
                    finally:
                        await worker_resp.aclose()
                        await _release_cm.__aexit__(None, None, None)

                return StreamingResponse(
                    relay(),
                    status_code=worker_resp.status_code,
                    media_type=worker_resp.headers.get("content-type", "text/event-stream"),
                    headers=_worker_headers(worker_port, domain),
                )

            resp = await http_client.post(
                f"{worker_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
        finally:
            if fifo_cm is not None:
                await fifo_cm.__aexit__(None, None, None)

        requests_total.labels(
            model=req.model,
            status=str(resp.status_code),
            path="proxy",
            auto="0",
        ).inc()
        try:
            response_body = resp.json()
        except ValueError:
            log.exception(
                "Worker %d returned non-JSON body (status=%d, len=%d)",
                worker_port, resp.status_code, len(resp.content),
            )
            track_chat(
                model_alias=req.model,
                domain=domain,
                kind="direct",
                request_body=_trace_req_body,
                response_body={},
                started_at=_trace_started_at,
                error=f"bad_gateway worker={worker_port} status={resp.status_code}",
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "type": "bad_gateway",
                    "worker_port": worker_port,
                    "worker_status": resp.status_code,
                    "message": "Worker returned an empty or invalid response",
                },
            ) from None
        response_body = _normalize_response_body(response_body)
        track_chat(
            model_alias=req.model,
            domain=domain,
            kind="direct",
            request_body=_trace_req_body,
            response_body=response_body,
            started_at=_trace_started_at,
            upstream_model=response_body.get("model"),
        )
        return JSONResponse(
            content=response_body,
            headers=_worker_headers(worker_port, domain, response_body),
        )

    return app
