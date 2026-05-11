# src/gateway/server.py
"""Gateway server — routes requests to the correct worker."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

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

MODEL_FORCE_MAP = {
    "ailiance-mistral-medium": 9301,  # Mistral Medium 3.5 128B Q8 (studio:9301, renamed from ailiance-apertus 2026-05-11)
    "ailiance-mistral": 9301,  # alias for ailiance-mistral-medium (same backend)
    "ailiance-apertus": 9301,  # legacy alias preserved for backwards compatibility — routes to Mistral-Medium
    "ailiance-devstral": 8502,  # legacy alias — macm1 worker now serves Gemma 4
    "ailiance-gemma4": 8502,  # Gemma 4 E4B + ailiance curriculum LoRA (macm1)
    "ailiance-eurollm": 9303,
    "ailiance-gemma": 9304,  # Gemma 3 4B IT on tower
    "ailiance-qwen": 8002,  # llama-server on kxkm-ai (RTX 4090) via autossh tunnel
    "ailiance-granite": 8003,  # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai
    "ailiance-qwen36": 9305,  # Qwen3.6-35B-A3B-MLX-BF16 on Studio (deeper specialist vs ailiance-qwen Q4)
    "ailiance-ministral": 8502,  # Ministral-3-14B-Instruct MLX 4-bit on macM1
    "ailiance-ministral-reasoning": 8502,  # Ministral-3-14B-Reasoning MLX 4-bit on macM1
    "ailiance-gemma2": 8502,  # Gemma-4-E2B-it MLX 4-bit on macM1 (lighter than E4B)
    # Devstral-Small-2-24B-MLX-4bit on Studio (:9316 base, :9317-9321 LoRA variants).
    "ailiance-devstral-base": 9316,
    "ailiance-python": 9317,
    "ailiance-cpp": 9318,
    "ailiance-rust-emb": 9319,
    "ailiance-html": 9320,
    "ailiance-ml-training": 9321,
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
    "ailiance-python": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-cpp": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-rust-emb": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-html": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-ml-training": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
}


WORKER_FORWARD_OVERRIDES: dict[int, dict[str, str]] = {
    8002: {
        "model": "qwen-32b-awq",  # the alias llama-server expects
        "auth_env": "AILIANCE_QWEN_KEY",
    },
    # mlx_lm.server resolves an unknown `model` field as a HF repo and tries to
    # download it; rewrite to the on-disk path the server already has loaded.
    9301: {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8",
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
                {"id": "ailiance-eurollm", "object": "model", "owned_by": "ailiance"},
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
                {"id": "ailiance-embed", "object": "model", "owned_by": "ailiance"},
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
            "ailiance-eurollm",
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
            return response

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

        # Forward rewrites: per-alias takes precedence over per-port. Lets a
        # single backend port host multiple ailiance-* aliases each rewritten
        # to a distinct upstream model id.
        override = ALIAS_MODEL_REWRITES.get(req.model) or WORKER_FORWARD_OVERRIDES.get(worker_port)
        if override:
            if "model" in override:
                body["model"] = override["model"]
            auth_env = override.get("auth_env")
            if auth_env:
                key = os.environ.get(auth_env, "")
                if key:
                    headers["Authorization"] = f"Bearer {key}"

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

            async def relay() -> "object":
                try:
                    async for chunk in worker_resp.aiter_raw():
                        yield chunk
                finally:
                    await worker_resp.aclose()

            return StreamingResponse(
                relay(),
                status_code=worker_resp.status_code,
                media_type=worker_resp.headers.get("content-type", "text/event-stream"),
            )

        resp = await http_client.post(
            f"{worker_url}/v1/chat/completions",
            json=body,
            headers=headers,
        )

        requests_total.labels(
            model=req.model,
            status=str(resp.status_code),
            path="proxy",
            auto="0",
        ).inc()
        try:
            return resp.json()
        except ValueError:
            log.exception(
                "Worker %d returned non-JSON body (status=%d, len=%d)",
                worker_port, resp.status_code, len(resp.content),
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

    return app
