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
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.gateway.alias_inventory import (
    inventory_or_unknown,
    resolve_effective_alias,
    to_dict as inventory_to_dict,
    to_headers as inventory_to_headers,
)
from src.gateway.file_extract import (
    ExtractError,
    ExtractResult,
    MAX_BYTES as FILE_MAX_BYTES,
    extract as extract_file,
)
from src.gateway.tenant_isolation import (
    derive_tenant_id,
    inject_tenant_prefix,
    isolation_enabled,
)
from src.gateway.inference_defaults import (
    apply_inference_defaults,
    default_system_prompt,
    messages_already_have_system,
)
from src.gateway.inline_files import (
    image_store,
    rewrite_image_urls,
    rewrite_input_files,
)
from src.gateway.observability import track_chat
from src.gateway.training.admin import make_training_router
from src.gateway.training.orchestrator import TrainingOrchestrator, build_training_503
from src.gateway.training.studio_ops import (
    MINIMAL_ROUTABLE_PORTS,
    StudioOps,
)
from src.orchestrator.chain_orchestrator import ChainOrchestrator
from src.orchestrator.chain_policy import ChainPolicy
from src.orchestrator.validators import StubValidator, make_validator
from src.router.domain_map import ALL_DOMAINS, DOMAIN_TO_OMLX_MODEL, DOMAIN_TO_QWEN36, OMLX_PORT, QWEN36_PORT, QWEN36_PORT_B, get_worker_for_domain
from src.worker.schemas import ChatCompletionRequest, ChatMessage
from src.gateway.gaia_x.serving import mount_well_known

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
    # Studio MLX :9340 — 10 qwen3-4b-mascarade hardware experts, each LoRA
    # merged into Qwen3-4B-Instruct-2507 and served as MLX bf16 (no Q4
    # quantization loss vs the Tower Ollama path). via autossh tunnel
    # electron-server:9340 → studio:9340. Set up the tunnel with:
    #   autossh -M 0 -N -L 0.0.0.0:9340:localhost:9340 \
    #       clems@100.116.92.12
    9340: "http://localhost:9340",
    # Studio swap server :9350 — one mlx_lm.server, no fixed model; loads
    # the requested base model on demand. autossh tunnel :9350 → studio.
    9350: "http://localhost:9350",
    # Studio M3 Ultra (512 GB) — MLX backends. All bind 127.0.0.1 on the
    # Studio host and are reached from electron-server via autossh tunnels
    # (one tunnel per port, see systemd `*-tunnel.service`). Defaults stay
    # localhost so a single-host setup just works; multi-host deployments
    # override the full map via AILIANCE_WORKERS_JSON to point each port
    # at the studio Tailscale address.
    9316: "http://localhost:9316",  # Devstral-Small-2-24B base
    9322: "http://localhost:9322",  # Apertus-70B multi-LoRA custom server
    9323: "http://localhost:9323",  # DeepSeek-R1-Distill-Qwen-32B 4-bit
    9324: "http://localhost:9324",  # Llama-3.3-70B-Instruct 4-bit
    9325: "http://localhost:9325",  # Pixtral-12B 4-bit (vision)
    9326: "http://localhost:9326",  # Mistral-Small-3.1-24B-Instruct 4-bit
    9327: "http://localhost:9327",  # Qwen3-Coder-30B-A3B-Instruct 4-bit
    9328: "http://localhost:9328",  # Qwen3-235B-A22B-Instruct MoE 4-bit
    9329: "http://localhost:9329",  # Mixtral-8x22B-Instruct 4-bit
    9330: "http://localhost:9330",  # Devstral multi-LoRA hot-swap server
    9335: "http://localhost:9335",  # Gemma-4-E4B multi-LoRA custom server
    8500: "http://100.116.92.12:8500",  # omlx consolidated multi-model (Tailscale)
    9360: "http://100.116.92.12:9360",  # qwen36 multi-LoRA server (Qwen3.6-35B + 30 adapters)
    9361: "http://100.116.92.12:9361",  # qwen36 instance B (load split)
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
    9340: 32768,    # Studio MLX bf16 qwen3-4b-mascarade experts (conservative cap)
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
    8500: 32768,    # omlx consolidated multi-model server
    9360: 262144,   # qwen36 Qwen3.6-35B multi-LoRA server (256k ctx)
    9361: 262144,   # qwen36 instance B (same ctx ceiling)
}


def _worker_headers(
    worker_port: int,
    domain: str,
    response_body: dict | None = None,
    chain_policy: str | None = None,
    effective_alias: str | None = None,
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
    if effective_alias:
        # Logical alias + base_model + LoRA stack — what the user asked
        # for, in terms the catalog uses. Distinct from the filesystem
        # path / model id the worker reports.
        inv = inventory_or_unknown(effective_alias)
        headers.update(inventory_to_headers(inv))
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
    "ailiance-eurollm": 9303,  # EuroLLM -> omlx via port 9303
    "ailiance-mistral-medium": 9301,  # Mistral Medium 3.5 128B Q8 (studio:9301, renamed from ailiance-apertus 2026-05-11)
    "ailiance-mistral": 9301,  # alias for ailiance-mistral-medium (same backend)
    "ailiance-apertus": 9301,  # legacy alias preserved for backwards compatibility — routes to Mistral-Medium
    "ailiance-devstral": 8502,  # legacy alias — macm1 worker now serves Gemma 4
    "ailiance-gemma4": 8502,  # Gemma 4 E4B + ailiance curriculum LoRA (macm1)
    "ailiance-gemma": 9304,  # Gemma 3 4B IT on tower
    "ailiance-qwen": 8002,  # llama-server on kxkm-ai (RTX 4090) via autossh tunnel
    "ailiance-granite": 8003,  # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai
    "ailiance-qwen36": 9350,  # Qwen3.6-35B-A3B — swap pool :9350 (on-demand)
    "ailiance-ministral": 8502,  # Ministral-3-14B-Instruct MLX 4-bit on macM1
    "ailiance-ministral-reasoning": 8502,  # Ministral-3-14B-Reasoning MLX 4-bit on macM1
    "ailiance-gemma2": 8502,  # Gemma-4-E2B-it MLX 4-bit on macM1 (lighter than E4B)
    # Devstral-Small-2-24B base — swap pool :9350 (on-demand).
    "ailiance-devstral-base": 9350,
    "ailiance-python": 9330,
    "ailiance-cpp": 9330,
    "ailiance-rust-emb": 9330,
    # devstral-html-css / devstral-ml-training LoRA adapters are
    # degenerate (repetition-loop garbage, verified 2026-05-19 by a
    # direct :9330 call). Route to Gemma until the adapters are
    # retrained — a coherent generic answer beats garbage.
    "ailiance-html": 9304,
    "ailiance-ml-training": 9304,
    # Studio MLX :9340 — 10 qwen3-4b-mascarade hardware experts. Each LoRA
    # (trained 2026-05-11, Qwen3-4B-Instruct-2507 base) was merged and
    # converted to MLX bf16, replacing the Tower Ollama Q4_K_M path so the
    # fine-tuned behaviour is served without quantization loss.
    "ailiance-kicad": 9340,
    # spice is excluded from MASCARADE_DOMAINS (auto-router) by PR #55 on
    # bench grounds, but the explicit alias still migrates to Studio: the
    # 2026-05-11 retrain benches well on spice-sim (Phase 7 composite 0.65).
    "ailiance-spice": 9340,
    "ailiance-stm32": 9340,
    "ailiance-emc": 9340,
    "ailiance-embedded": 9340,
    "ailiance-platformio": 9340,
    "ailiance-freecad": 9340,
    "ailiance-dsp": 9340,
    "ailiance-iot": 9340,
    "ailiance-power": 9340,
    # Still on Tower Ollama :8004 — not part of the Qwen3-4B mascarade set.
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
    # Qwen3-235B-A22B-Instruct MoE — swap pool :9350 (on-demand, ~120 GB).
    "ailiance-flagship": 9350,
    "ailiance-qwen-235b": 9350,
    # Studio S3 additions 2026-05-12 — 5 MLX 4-bit workers on dedicated ports.
    "ailiance-reasoning-r1": 9323,  # DeepSeek-R1-Distill-Qwen-32B 4-bit
    "ailiance-llama": 9350,  # Llama-3.3-70B-Instruct — swap pool :9350
    "ailiance-pixtral": 9325,  # Pixtral-12B 4-bit (vision-language)
    "ailiance-mistral-small": 9350,  # Mistral-Small-3.1-24B — swap pool :9350
    "ailiance-coder-pro": 9327,  # Qwen3-Coder-30B-A3B-Instruct 4-bit
    # Mixtral-8x22B-Instruct — swap pool :9350 (on-demand). `ailiance-mixtral`
    # kept for prod main-line consumers; `-8x22b` is the explicit name.
    "ailiance-mixtral": 9350,
    "ailiance-mixtral-8x22b": 9350,
    # Gemma-4-E4B multi-LoRA custom server on Studio :9335 (mascarade + aggro + kicad9plus variants).
    "ailiance-gemma4-mascarade": 9335,
    "ailiance-gemma4-aggro": 9335,
    "ailiance-gemma4-kicad9plus": 9335,
}


# Aliases derived from MODEL_FORCE_MAP but NOT advertised on the public
# /v1/models surface. Today only the bare auto-router id ``ailiance`` is
# *added* (it isn't in MODEL_FORCE_MAP because it's resolved by the
# classifier rather than a forced port). Use this set to denylist legacy
# or internal-only forced aliases — `ailiance-devstral` is the legacy
# pre-rename alias for `ailiance-gemma4`, kept routable for back-compat
# but intentionally hidden from the catalog.
_INTERNAL_ALIASES: frozenset[str] = frozenset({
    "ailiance-devstral",  # legacy alias preserved for backwards compatibility
})


def _compute_public_aliases() -> list[str]:
    """Return the canonical, ordered list of aliases exposed publicly.

    Single source of truth for ``/v1/models`` and ``/v1/models/details``;
    before 2026-05-18 the two endpoints maintained independent hand-rolled
    lists and drifted (22 aliases on /v1/models had no display metadata
    counterpart). The order is: bare ``ailiance`` auto-router first, then
    ``MODEL_FORCE_MAP`` keys in their declaration order (Python dicts
    preserve insertion order since 3.7), minus the ``_INTERNAL_ALIASES``
    denylist.
    """
    ordered: list[str] = ["ailiance"]
    seen: set[str] = {"ailiance"}
    for alias in MODEL_FORCE_MAP:
        if alias in _INTERNAL_ALIASES or alias in seen:
            continue
        ordered.append(alias)
        seen.add(alias)
    return ordered


ALL_PUBLIC_ALIASES: list[str] = _compute_public_aliases()


def _warn_force_map_worker_drift() -> None:
    """Surface MODEL_FORCE_MAP ports missing from WORKER_URLS at startup.

    Each such port routes silently through ``_gate_port`` to the Gemma
    health fallback, hiding misconfiguration behind a (working but wrong)
    response. The 2026-05-11 incident logged in ``_load_worker_urls``
    already covers the AILIANCE_WORKERS_JSON env var case; this one
    catches the *source-level* drift: a new alias landing in
    MODEL_FORCE_MAP without its port being added to ``_DEFAULT_WORKER_URLS``.
    """
    force_ports = {p for p in MODEL_FORCE_MAP.values() if p is not None}
    missing = force_ports - set(WORKER_URLS)
    if not missing:
        return
    affected = sorted(
        a for a, p in MODEL_FORCE_MAP.items() if p in missing
    )
    log.warning(
        "MODEL_FORCE_MAP references %d port(s) absent from WORKER_URLS: "
        "%s — aliases %s will silently fall back to Gemma 9304. Add the "
        "ports to _DEFAULT_WORKER_URLS or to AILIANCE_WORKERS_JSON.",
        len(missing),
        sorted(missing),
        affected,
    )


_warn_force_map_worker_drift()


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
    "ailiance-granite": {"model": "granite-4.1-30b-mxfp8", "auth_env": "AILIANCE_QWEN_KEY"},
    # studio mlx_lm.server :9305 - rewrite to on-disk path the server has loaded.
    "ailiance-qwen36": {"model": "Qwen3.6-35B-A3B-MLX-BF16"},
    # studio mlx_lm.server :9301 - rewrite to on-disk path the server has loaded
    # (mlx_lm.server resolves an unknown model field as an HF repo id, causing 404 + 60s timeout).
    "ailiance-mistral-medium": {"model": "Mistral-Medium-3.5-128B-MLX-Q8"},
    "ailiance-mistral": {"model": "Mistral-Medium-3.5-128B-MLX-Q8"},
    "ailiance-apertus": {"model": "Mistral-Medium-3.5-128B-MLX-Q8"},  # legacy alias
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
    "ailiance-devstral-base": {"model": "Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-python": {"model": "devstral-python"},
    "ailiance-cpp": {"model": "devstral-cpp"},
    "ailiance-rust-emb": {"model": "devstral-rust-embedded"},
    # ailiance-html / ailiance-ml-training intentionally omitted — they
    # route to Gemma :9304 (degenerate adapters); the :9304 port-level
    # override supplies the eu-kiki-gemma model id.
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
        "model": "Qwen3-235B-A22B-Instruct-MLX-4bit",
    },
    "ailiance-qwen-235b": {
        "model": "Qwen3-235B-A22B-Instruct-MLX-4bit",
    },
    # Studio S3 additions 2026-05-12 — mlx_lm.server expects on-disk path.
    "ailiance-reasoning-r1": {
        "model": "DeepSeek-R1-Distill-Qwen-32B-MLX-4bit",
    },
    "ailiance-llama": {
        "model": "Llama-3.3-70B-Instruct-MLX-4bit",
    },
    "ailiance-pixtral": {
        "model": "Pixtral-12B-MLX-4bit",
    },
    "ailiance-mistral-small": {
        "model": "Mistral-Small-3.1-24B-Instruct-MLX-4bit",
    },
    "ailiance-coder-pro": {
        "model": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    },
    # Mixtral-8x22B-Instruct MLX 4-bit on Studio :9329 — mlx_lm.server
    # expects on-disk path. Both alias names share the same rewrite.
    "ailiance-mixtral": {
        "model": "Mixtral-8x22B-Instruct-MLX-4bit",
    },
    "ailiance-mixtral-8x22b": {
        "model": "Mixtral-8x22B-Instruct-MLX-4bit",
    },
    # Studio multi-LoRA Gemma-4-E4B custom server :9335. Each alias rewrites
    # `model` body field to the adapter_name the custom server expects.
    # ailiance-gemma4 stays on macM1:8502 + studio:9334 HA mirror (single-LoRA eukiki).
    "ailiance-gemma4-mascarade": {"model": "gemma4-mascarade"},
    "ailiance-gemma4-aggro": {"model": "gemma4-aggro"},
    "ailiance-gemma4-kicad9plus": {"model": "gemma4-kicad9plus"},
}


WORKER_FORWARD_OVERRIDES: dict[int, dict[str, str]] = {
    9303: {"model": "EuroLLM-22B-Instruct-2512"},  # eurollm -> omlx
    8002: {
        "model": "qwen-32b-awq",  # the alias llama-server expects
        "auth_env": "AILIANCE_QWEN_KEY",
    },
    # kxkm-ai llama.cpp :18889 served via tunnel :8003.
    8003: {
        "model": "granite-4.1-30b-mxfp8",
    },
    # mlx_lm.server resolves an unknown `model` field as a HF repo and tries to
    # download it; rewrite to the on-disk path the server already has loaded.
    9301: {
        "model": "Mistral-Medium-3.5-128B-MLX-Q8",
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


def _request_has_input_files(req) -> bool:
    """Return True iff any message in ``req`` carries an ``input_file`` part."""
    for msg in getattr(req, "messages", []) or []:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_file":
                    return True
    return False


def _content_to_text(content: Any) -> str:
    """Flatten a ChatMessage.content value to a routing-friendly string.

    After the inline file/image rewrites, ``content`` may legitimately
    be a list of OpenAI blocks (text + image_url + input_file →
    rewritten text). Components downstream that expect a plain string
    (the classifier, complexity heuristics, cascade pick) need the
    concatenated text. We pull ``text`` from text blocks and drop the
    rest — non-text blocks (image_url) don't carry routing signal.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _last_user_text(req) -> str:
    """Return the last user message's text content, flat string only."""
    for msg in reversed(getattr(req, "messages", []) or []):
        if getattr(msg, "role", None) == "user":
            return _content_to_text(getattr(msg, "content", None))
    return ""


def _public_base_url() -> str:
    """Public base URL the gateway is reachable at — for staged image URLs.

    Set via ``AILIANCE_PUBLIC_BASE_URL`` (e.g. ``https://gateway.ailiance.fr``).
    Falls back to ``http://localhost:9300`` for local dev / tests; that
    fallback is only useful when both gateway and worker run on the
    same host.
    """
    return os.environ.get("AILIANCE_PUBLIC_BASE_URL", "http://localhost:9300")


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


_DEFAULT_CORS_ORIGINS = (
    "https://www.ailiance.fr",
    "https://preview.ailiance.fr",
    "https://ailiance.fr",
    # Local development for the cockpit-public vite dev server.
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
)


def _cors_origins() -> list[str]:
    """Resolve the allow-list for cross-origin requests.

    Override via ``AILIANCE_CORS_ORIGINS`` (comma-separated). The
    default list covers the production cockpit (www.ailiance.fr), its
    preview deployment, the apex, and the two vite dev ports.
    """
    raw = os.environ.get("AILIANCE_CORS_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return list(_DEFAULT_CORS_ORIGINS)


def make_gateway_app(skip_router_load: bool = False) -> FastAPI:
    app = FastAPI(title="ailiance-gateway")
    # CORS for the browser-side Playground: cockpit-public (deployed on
    # www.ailiance.fr) calls gateway.ailiance.fr from a different
    # origin. Browsers send a preflight OPTIONS for any multipart POST
    # — without CORSMiddleware that preflight returns 405 and the
    # browser aborts the real request ("Load failed" in DevTools).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
        max_age=86400,
    )
    # Gaia-X: serve did.json + signed VCs under /.well-known (no-op until
    # `python -m src.gateway.gaia_x.cli render` has written var/well-known).
    mount_well_known(app)
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

    app.state.training = TrainingOrchestrator(StudioOps(), Path(os.environ.get("AILIANCE_CAMPAIGN_STATE", "campaign_state.json")))
    app.include_router(make_training_router())

    # v1 OpenAI Realtime-compatible WebSocket — see src/realtime/.
    # Behind the same auth as /v1/chat/completions; one upstream Kyutai
    # STT session per connection (MacStudio :8304).
    from src.realtime.router import router as realtime_router
    app.include_router(realtime_router)

    start_time = time.time()
    http_client = httpx.AsyncClient(timeout=1800.0)

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
        try:
            await app.state.training.reattach()
        except Exception:  # noqa: BLE001
            log.exception("training re-attach failed")

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
    def list_models(request: Request):
        """OpenAI-compatible chat-model catalog.

        Sourced from ``ALL_PUBLIC_ALIASES`` (derived from
        ``MODEL_FORCE_MAP`` minus ``_INTERNAL_ALIASES``) and further
        filtered to drop ``_BLOCKED_CHAT_ALIASES`` (embedding-only
        surfaces have no chat completion semantics). Stays byte-identical
        to ``/v1/models/details`` minus that one documented embed
        exclusion — the two used to drift, see E.1 audit 2026-05-18.
        """
        ids = [a for a in ALL_PUBLIC_ALIASES if a not in _BLOCKED_CHAT_ALIASES]
        # Liveness filter: suppress aliases whose forced port is currently
        # unhealthy. The bare "ailiance" auto-router alias has no fixed port
        # (it's classifier-dispatched) so it is always kept.
        ids = [
            a for a in ids
            if a == "ailiance" or MODEL_FORCE_MAP.get(a) in _healthy_ports
        ]
        training = request.app.state.training
        unloaded = (
            set(training.state.unloaded_ports)
            if training.state.is_active
            else set()
        )
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "owned_by": "ailiance",
                    "status": (
                        "training"
                        if MODEL_FORCE_MAP.get(mid) in unloaded
                        else "ready"
                    ),
                }
                for mid in ids
            ],
        }

    @app.get("/v1/models/details")
    def list_models_details() -> dict:
        """Enriched model listing with display metadata.

        Reads `configs/models-display.yaml` on each call so descriptions
        can be edited without a gateway restart. The minimal /v1/models
        endpoint stays OpenAI-standard for plain clients.

        Shares ``ALL_PUBLIC_ALIASES`` with /v1/models; the only delta is
        that this surface also lists ``_BLOCKED_CHAT_ALIASES`` (the embed
        worker) because the metadata catalog is the canonical introspection
        view, regardless of whether the alias is chat-callable.
        """
        import yaml as _yaml

        path = Path("configs/models-display.yaml")
        try:
            raw = _yaml.safe_load(path.read_text()) if path.exists() else {}
        except Exception as exc:
            log.warning("models-display.yaml parse failed: %s", exc)
            raw = {}
        models = raw.get("models", {}) if isinstance(raw, dict) else {}
        # Enumerate the same id list as /v1/models, then re-add any
        # blocked-chat surfaces (embedding workers) so the metadata
        # catalog is exhaustive even if they're not chat-callable.
        ids: list[str] = list(ALL_PUBLIC_ALIASES)
        for blocked in _BLOCKED_CHAT_ALIASES:
            if blocked not in ids and blocked in MODEL_FORCE_MAP:
                ids.append(blocked)
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

    @app.get("/v1/_staged/{key}")
    def get_staged(key: str) -> Response:
        """Serve a staged image so vision workers can fetch it over HTTP.

        Used by the data-URL rewrite: when a caller embeds
        ``image_url.url = "data:image/png;base64,…"`` the gateway stages
        the bytes here and rewrites the URL to ``…/v1/_staged/<key>``.
        Worker downloads as a normal HTTP image. TTL is enforced inside
        :class:`_ImageStore`; expired keys return 404.
        """
        entry = image_store.get(key)
        if entry is None:
            raise HTTPException(status_code=404, detail="staged image not found or expired")
        return Response(content=entry.data, media_type=entry.mime or "application/octet-stream")

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
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        _trace_started_at = time.perf_counter()
        _trace_req_body = req.model_dump(exclude_none=True)
        # Per-tenant KV-cache isolation: prepend a short session marker
        # so two callers with different identities never share cache
        # entries on the worker. Toggled off via AILIANCE_TENANT_ISOLATION=0.
        if isolation_enabled():
            tenant_id = derive_tenant_id(
                dict(request.headers),
                request.client.host if request.client else None,
            )
            req.messages = inject_tenant_prefix(
                req.messages, tenant_id, req.model
            )
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

        # Inline file extraction: replace any input_file blocks in
        # messages[].content with text blocks carrying the extracted
        # markdown. Lets callers attach PDF/DOCX/etc. directly inside
        # /v1/chat/completions without a separate /v1/files/extract
        # round-trip.
        if _request_has_input_files(req):
            try:
                await rewrite_input_files(req.messages, http_client)
            except ExtractError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "type": "invalid_request_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                ) from exc

        # Image data-URL staging: MLX vision workers (Pixtral on
        # :9325) reject ``data:image/*;base64,…`` URLs silently. We
        # decode any such URL, stash the bytes in a short-lived in-
        # memory store, and rewrite the URL to a public endpoint
        # this gateway serves. Worker fetches the image as a normal
        # HTTP URL and actually sees it.
        if _request_has_images(req):
            rewrite_image_urls(req.messages, _public_base_url())

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

        # Per-alias default system prompt: only prepended if the caller
        # hasn't already supplied a ``system`` message. Used today to
        # suppress Pixtral's short-prompt quirk where it regurgitates a
        # Python dict instead of plain text. Insertion happens after
        # multimodal auto-route so the prompt matches the alias the
        # worker actually serves.
        _sys_default = default_system_prompt(req.model)
        if _sys_default and not messages_already_have_system(req.messages):
            req.messages.insert(0, ChatMessage(role="system", content=_sys_default))

        # Extract the last user message once; reused for routing, cascade
        # complexity scoring, and the orchestrator path below. _last_user_text
        # walks all messages and reflattens content, so caching it avoids
        # O(n) reparses on the hot path.
        user_msg = _last_user_text(req)

        forced_port = MODEL_FORCE_MAP.get(req.model)
        active_router = app.state.router
        if forced_port:
            domain = ""
            # Raw resolved port — gated below, after the training-mode check.
            worker_port = forced_port
        elif active_router is not None:
            t0 = time.perf_counter()
            # Encode runs on CPU (no GPU on the prod host) and holds the GIL
            # during tokenization — run it in a worker thread so concurrent
            # stream relays keep progressing.
            selections = await asyncio.to_thread(active_router.route, user_msg)
            route_latency.observe(time.perf_counter() - t0)
            domain = selections[0][0] if selections else "python"
            # Fallback to Gemma (9304): reachable + fast for unmapped domains.
            worker_port = get_worker_for_domain(domain) or 9304
        else:
            # No router loaded → default to Gemma (fast, reachable).
            domain = "general"
            worker_port = 9304

        # Training-mode interception. Runs on the RAW resolved worker_port,
        # BEFORE _gate_port() rewrites a dead/unloaded port to the 9304
        # fallback — otherwise the unloaded-port match would be hidden and
        # the 503 would never fire. When a medium35 campaign is active and
        # the resolved worker has been unloaded: an explicitly-named model
        # gets a graceful 503; an auto-routed request is re-pointed to a
        # minimal still-loaded worker.
        training = request.app.state.training
        if training.state.is_active and worker_port in training.state.unloaded_ports:
            if MODEL_FORCE_MAP.get(req.model) is not None:
                return JSONResponse(
                    status_code=503,
                    content=build_training_503(training.state, req.model),
                    headers={"Retry-After": "3600"},
                )
            # Auto-routed request: re-point to a minimal still-loaded worker.
            # Note: a later cascade-override may re-resolve the port; if cascade
            # lands on an unloaded worker the request is not re-checked here — it
            # degrades gracefully via _gate_port to the 9304 fallback (no 503).
            worker_port = sorted(MINIMAL_ROUTABLE_PORTS)[0]

        # Health gate: if the resolved worker is currently down, fall back to
        # a healthy worker so prompts don't 500 just because the router
        # happened to classify them to a temporarily dead backend.
        worker_port = _gate_port(worker_port)

        # ------------------------------------------------------------------
        # Cascade complexity-based override (v0.4).
        # Only applies on the auto-router path (req.model == "ailiance" /
        # no forced port). When AILIANCE_CASCADE_ENABLED=1, short prompts
        # are rerouted to a fast small model and reasoning-heavy prompts
        # are escalated to a flagship. Forced aliases are NEVER cascaded.
        # ------------------------------------------------------------------
        cascade_alias: str | None = None
        if not forced_port and domain:
            cascade_alias = _cascade_pick(domain, user_msg)
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
            chain_alias = resolve_effective_alias(
                req.model, cascade_alias=cascade_alias, domain=domain,
            )
            chain_inv = inventory_or_unknown(chain_alias)
            response["ailiance"] = inventory_to_dict(chain_inv)
            return JSONResponse(
                content=response,
                headers=_worker_headers(
                    worker_port,
                    domain,
                    response,
                    chain_policy=chain_result.policy.value,
                    effective_alias=chain_alias,
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
        # Apply per-alias inference defaults: max_tokens for reasoning
        # models, low temp for vision/code workers, enable_thinking=False
        # for Qwen3.5, stop tokens for Pixtral, etc. Caller wins on every
        # field; defaults fill gaps only.
        _defaults_applied = apply_inference_defaults(body, req.model)

        # Forward rewrites: per-alias takes precedence over per-port. Lets a
        # single backend port host multiple ailiance-* aliases each rewritten
        # to a distinct upstream model id.
        # Cascade: when the v0.4 complexity heuristic remapped this request,
        # honour the cascade alias's rewrite (e.g. mlx_lm.server :8502 needs
        # the on-disk path the cascade target loaded).
        rewrite_key = cascade_alias if cascade_alias else req.model
        _qwen36_override = (
            {"model": DOMAIN_TO_QWEN36[domain]}
            if worker_port in (QWEN36_PORT, QWEN36_PORT_B) and domain in DOMAIN_TO_QWEN36
            else None
        )
        _omlx_override = (
            {"model": DOMAIN_TO_OMLX_MODEL[domain]}
            if worker_port == OMLX_PORT and domain in DOMAIN_TO_OMLX_MODEL
            else None
        )
        override = ALIAS_MODEL_REWRITES.get(rewrite_key) or _qwen36_override or _omlx_override or WORKER_FORWARD_OVERRIDES.get(worker_port)
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

                # Streaming asymmetry: once bytes start flushing we cannot
                # change the HTTP status to 502.  Instead we track whether
                # any content/tool_calls delta was observed and log a warning
                # (+ track_chat error) at stream end if none was seen.
                async def relay() -> "object":
                    saw_content = False
                    _mid_stream_drop = False  # set in except; suppresses double-signal in finally
                    try:
                        async for chunk in _normalize_sse_stream(
                            worker_resp.aiter_text()
                        ):
                            # Detect non-empty deltas to identify empty completions.
                            if not saw_content and isinstance(chunk, (bytes, str)):
                                text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
                                if ('"content"' in text or '"tool_calls"' in text or '"function_call"' in text):
                                    # Quick heuristic: if any content/tool key appears in the
                                    # chunk the stream has substance.  False-negatives (key
                                    # present but value empty) are acceptable — this is a
                                    # best-effort warning, not a hard gate.
                                    import re as _re
                                    if _re.search(r'"content"\s*:\s*"[^"\\]', text):
                                        saw_content = True
                                    elif '"tool_calls"' in text or '"function_call"' in text:
                                        saw_content = True
                            yield chunk
                    except httpx.RequestError as exc:
                        # Worker dropped the connection mid-stream (after headers were
                        # sent, so the HTTP 200 is already committed — cannot become a
                        # 5xx).  Surface via log + audit telemetry instead of failing
                        # silently.  The generator ends here; the client stream is
                        # truncated.
                        #
                        # Double-signal policy: set _mid_stream_drop=True so the
                        # finally block skips the empty-completion warning even if
                        # saw_content is False (the drop warning is the right signal;
                        # emitting both would be confusing in the audit trail).
                        _mid_stream_drop = True
                        log.warning(
                            "mid-stream worker drop worker=%s: %s",
                            worker_port,
                            exc,
                        )
                        track_chat(
                            model_alias=req.model,
                            domain=domain,
                            kind="direct",
                            request_body=_trace_req_body,
                            response_body={},
                            started_at=_trace_started_at,
                            error=f"mid_stream_drop worker={worker_port}: {exc}",
                        )
                        # Generator ends; finally still runs for lock/socket cleanup.
                    finally:
                        if not saw_content and not _mid_stream_drop:
                            log.warning(
                                "empty streaming completion worker=%s",
                                worker_port,
                            )
                            track_chat(
                                model_alias=req.model,
                                domain=domain,
                                kind="direct",
                                request_body=_trace_req_body,
                                response_body={},
                                started_at=_trace_started_at,
                                error=f"empty_completion_stream worker={worker_port}",
                            )
                        await worker_resp.aclose()
                        await _release_cm.__aexit__(None, None, None)

                return StreamingResponse(
                    relay(),
                    status_code=worker_resp.status_code,
                    media_type=worker_resp.headers.get("content-type", "text/event-stream"),
                    headers=_worker_headers(
                        worker_port, domain,
                        effective_alias=resolve_effective_alias(
                            req.model, cascade_alias=cascade_alias, domain=domain,
                        ),
                    ),
                )

            resp = await http_client.post(
                f"{worker_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
        except httpx.RequestError as exc:
            log.warning(
                "Worker %d unreachable: %s", worker_port, exc,
            )
            track_chat(
                model_alias=req.model,
                domain=domain,
                kind="direct",
                request_body=_trace_req_body,
                response_body={},
                started_at=_trace_started_at,
                error=f"upstream_unreachable worker={worker_port}",
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "type": "upstream_unreachable",
                    "worker_port": worker_port,
                    "message": f"Worker on port {worker_port} is unreachable",
                },
            ) from exc
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

        # Issue #10 — guard against silent empty completions.
        # A worker that returns HTTP 200 with structurally-valid JSON but
        # empty content and completion_tokens==0 would otherwise be relayed
        # silently.  We surface it as a clean 502 so callers can retry.
        #
        # Conservative policy: when ``usage`` is absent the worker gave no
        # token info, so ``completion_tokens`` is None (not 0) and we relay
        # rather than 502 — we cannot distinguish an empty completion from a
        # tool-call or a future extension that omits usage.
        #
        # This guard runs AFTER _normalize_response_body so that a
        # reasoning-only response whose content was backfilled from
        # ``message.reasoning`` already has non-empty content and passes.
        _g_choice0 = (response_body.get("choices") or [{}])[0]
        _g_msg = _g_choice0.get("message") or {}
        _g_content = _g_msg.get("content")
        _g_usage = response_body.get("usage") or {}
        _g_completion_tokens = _g_usage.get("completion_tokens")  # None when absent

        _g_content_empty = not (isinstance(_g_content, str) and _g_content.strip())
        _g_no_tools = not _g_msg.get("tool_calls") and not _g_msg.get("function_call")
        _g_no_tokens = _g_completion_tokens == 0  # False when None (absent) → conservative relay

        if _g_content_empty and _g_no_tools and _g_no_tokens:
            log.warning(
                "Worker %d returned empty completion (completion_tokens=0, no content)",
                worker_port,
            )
            track_chat(
                model_alias=req.model,
                domain=domain,
                kind="direct",
                request_body=_trace_req_body,
                response_body=response_body,
                started_at=_trace_started_at,
                error=f"empty_completion worker={worker_port}",
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "type": "empty_completion",
                    "worker_port": worker_port,
                    "message": "Worker returned a structurally-valid but empty completion",
                },
            )

        # Resolve the effective alias for observability: cascade overrides
        # win > FC force-route > caller's req.model > 'ailiance' fallback.
        # For req.model == "ailiance" the resolver lifts to the actually
        # served alias using the classifier's domain (e.g. domain="kicad"
        # → ailiance-kicad).
        effective_alias = resolve_effective_alias(
            req.model, cascade_alias=cascade_alias, domain=domain,
        )
        # Stamp the inventory dict on the JSON body so SDKs that hide
        # response headers (most OpenAI clients do) still see the
        # alias / base_model / LoRA stack that served them.
        inv = inventory_or_unknown(effective_alias)
        response_body["ailiance"] = inventory_to_dict(inv)
        track_chat(
            model_alias=effective_alias,
            domain=domain,
            kind="direct",
            request_body=_trace_req_body,
            response_body=response_body,
            started_at=_trace_started_at,
            upstream_model=response_body.get("model"),
        )
        return JSONResponse(
            content=response_body,
            headers=_worker_headers(
                worker_port, domain, response_body,
                effective_alias=effective_alias,
            ),
        )

    return app
