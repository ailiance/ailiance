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
    9302: "http://localhost:9302",
    9303: "http://localhost:9303",
    9304: "http://localhost:9304",
    # Qwen3.5 35B A3B on kxkm-ai (llama-server, alias 'qwen-32b-awq')
    # reached via the autossh tunnel listening on 0.0.0.0:8002.
    8002: "http://localhost:8002",
}


def _load_worker_urls() -> dict[int, str]:
    """Allow distributed deployments to override WORKER_URLS via env var.

    Set ``EU_KIKI_WORKERS_JSON='{"9301":"http://studio:9301", ...}'`` to point
    each worker at a Tailscale/LAN address. Defaults stay localhost so a
    single-host setup just works.
    """
    raw = os.environ.get("EU_KIKI_WORKERS_JSON")
    if not raw:
        return dict(_DEFAULT_WORKER_URLS)
    try:
        return {int(k): str(v) for k, v in json.loads(raw).items()}
    except Exception as exc:
        log.warning(
            "failed to parse EU_KIKI_WORKERS_JSON (%s); using defaults", exc,
        )
        return dict(_DEFAULT_WORKER_URLS)


WORKER_URLS = _load_worker_urls()

MODEL_FORCE_MAP = {
    "eu-kiki-apertus": 9301,
    "eu-kiki-devstral": 9302,
    "eu-kiki-eurollm": 9303,
    "eu-kiki-qwen": 8002,  # llama-server on kxkm-ai
}

# Per-port forward overrides for non-eu-kiki backends. The gateway rewrites
# the request body's `model` field and injects an Authorization header before
# proxying. Both pieces are sourced from env so secrets never land in source.
WORKER_FORWARD_OVERRIDES: dict[int, dict[str, str]] = {
    8002: {
        "model": "qwen-32b-awq",  # the alias llama-server expects
        "auth_env": "KXKM_QWEN_KEY",
    },
}


def make_gateway_app(skip_router_load: bool = False) -> FastAPI:
    app = FastAPI(title="eu-kiki-gateway")
    reg = CollectorRegistry()
    requests_total = Counter(
        "eu_kiki_gw_requests_total",
        "Gateway requests",
        ["model", "status"],
        registry=reg,
    )
    route_latency = Histogram(
        "eu_kiki_gw_route_seconds",
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
                {"id": "eu-kiki", "object": "model", "owned_by": "eu-kiki"},
                {"id": "eu-kiki-apertus", "object": "model", "owned_by": "eu-kiki"},
                {"id": "eu-kiki-devstral", "object": "model", "owned_by": "eu-kiki"},
                {"id": "eu-kiki-eurollm", "object": "model", "owned_by": "eu-kiki"},
                {"id": "eu-kiki-qwen", "object": "model", "owned_by": "eu-kiki"},
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

        # Per-port forward rewrites: rename `model` and inject Authorization
        # for backends that aren't part of the eu-kiki worker pool (kxkm-ai
        # llama-server expects an alias + bearer key).
        override = WORKER_FORWARD_OVERRIDES.get(worker_port)
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
