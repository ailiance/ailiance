# src/gateway/server.py
"""Gateway server — routes requests to the correct worker."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.router.domain_map import ALL_DOMAINS, get_worker_for_domain
from src.worker.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

WORKER_URLS = {
    9201: "http://localhost:9201",
    9202: "http://localhost:9202",
    9203: "http://localhost:9203",
}

MODEL_FORCE_MAP = {
    "eu-kiki-apertus": 9201,
    "eu-kiki-devstral": 9202,
    "eu-kiki-eurollm": 9203,
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
            ],
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
