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

from src.router.domain_map import ALL_DOMAINS, get_worker_for_domain
from src.worker.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

_DEFAULT_WORKER_URLS = {
    9301: "http://localhost:9301",
    8502: "http://localhost:8502",  # eu-kiki / ailiance worker on macm1 (Gemma 4 E4B + LoRA)
    9303: "http://localhost:9303",
    9304: "http://localhost:9304",
    # Qwen3.5 35B A3B on kxkm-ai (llama-server, alias 'qwen-32b-awq')
    # reached via the autossh tunnel listening on 0.0.0.0:8002.
    8002: "http://localhost:8002",
    # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai (llama-server :18889)
    # via autossh tunnel electron-server:8003.
    8003: "http://localhost:8003",
}


def _load_worker_urls() -> dict[int, str]:
    """Allow distributed deployments to override WORKER_URLS via env var.

    Set ``AILIANCE_WORKERS_JSON='{"9301":"http://studio:9301", ...}'`` to point
    each worker at a Tailscale/LAN address. Defaults stay localhost so a
    single-host setup just works.
    """
    raw = os.environ.get("AILIANCE_WORKERS_JSON")
    if not raw:
        return dict(_DEFAULT_WORKER_URLS)
    try:
        return {int(k): str(v) for k, v in json.loads(raw).items()}
    except Exception as exc:
        log.warning(
            "failed to parse AILIANCE_WORKERS_JSON (%s); using defaults", exc,
        )
        return dict(_DEFAULT_WORKER_URLS)


WORKER_URLS = _load_worker_urls()

MODEL_FORCE_MAP = {
    "ailiance-apertus": 9301,
    "ailiance-mistral": 9301,  # Mistral Medium 3.5 128B Q8 (replaces Apertus on studio:9301)
    "ailiance-devstral": 8502,  # legacy alias — macm1 worker now serves Gemma 4
    "ailiance-gemma4": 8502,  # Gemma 4 E4B + ailiance curriculum LoRA (macm1)
    "ailiance-eurollm": 9303,
    "ailiance-gemma": 9304,  # Gemma 3 4B IT on tower
    "ailiance-qwen": 8002,  # llama-server on kxkm-ai (RTX 4090) via autossh tunnel
    "ailiance-granite": 8003,  # Granite 4.1 30B Q4_K_M GGUF on kxkm-ai
    "ailiance-ministral": 8502,  # Ministral-3-14B-Instruct MLX 4-bit on macM1
    "ailiance-ministral-reasoning": 8502,  # Ministral-3-14B-Reasoning MLX 4-bit on macM1
    "ailiance-gemma2": 8502,  # Gemma-4-E2B-it MLX 4-bit on macM1 (lighter than E4B)
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
    "ailiance-granite": {"model": "granite-30b", "auth_env": "KXKM_QWEN_KEY"},
}


WORKER_FORWARD_OVERRIDES: dict[int, dict[str, str]] = {
    8002: {
        "model": "qwen-32b-awq",  # the alias llama-server expects
        "auth_env": "KXKM_QWEN_KEY",
    },
    # mlx_lm.server resolves an unknown `model` field as a HF repo and tries to
    # download it; rewrite to the on-disk path the server already has loaded.
    9301: {
        "model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8",
    },
    8502: {
        "model": "lmstudio-community/gemma-4-E4B-it-MLX-4bit",  # base model id loaded with curriculum LoRA adapter
    },
}


def make_gateway_app(skip_router_load: bool = False) -> FastAPI:
    app = FastAPI(title="ailiance-gateway")
    reg = CollectorRegistry()
    requests_total = Counter(
        "ailiance_gw_requests_total",
        "Gateway requests",
        ["model", "status"],
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
                {"id": "ailiance-mistral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma4", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-eurollm", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-qwen", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-granite", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-ministral", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-ministral-reasoning", "object": "model", "owned_by": "ailiance"},
                {"id": "ailiance-gemma2", "object": "model", "owned_by": "ailiance"},
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
            "ailiance-apertus",
            "ailiance-mistral",
            "ailiance-gemma4",
            "ailiance-eurollm",
            "ailiance-gemma",
            "ailiance-qwen",
            "ailiance-granite",
            "ailiance-ministral",
            "ailiance-ministral-reasoning",
            "ailiance-gemma2",
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
        if forced_port:
            domain = ""
            worker_port = forced_port
        elif router is not None:
            user_msg = next(
                (m.content for m in reversed(req.messages) if m.role == "user"), ""
            )
            t0 = time.perf_counter()
            selections = router.route(user_msg)
            route_latency.observe(time.perf_counter() - t0)
            domain = selections[0][0] if selections else "python"
            # Fallback to Gemma (9304): reachable + fast for unmapped domains.
            worker_port = get_worker_for_domain(domain) or 9304
        else:
            # No router loaded → default to Gemma (fast, reachable).
            domain = "general"
            worker_port = 9304

        worker_url = WORKER_URLS[worker_port]
        headers = {"X-Lora-Domain": domain}
        # exclude_none so optional ChatMessage fields (tool_calls, name,
        # tool_call_id) don't reach llama.cpp workers as `null` and trip
        # their JSON schema validation.
        body = req.model_dump(exclude_none=True)

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
                model=req.model, status=str(worker_resp.status_code)
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

        requests_total.labels(model=req.model, status=str(resp.status_code)).inc()
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
