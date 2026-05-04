"""Worker FastAPI server — serves one MLX model with LoRA hot-swap."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.worker.runtime import MLXWorkerRuntime, WorkerConfig
from src.worker.schemas import ChatCompletion, ChatCompletionRequest, ChatMessage, Choice

log = logging.getLogger(__name__)


def make_worker_app(cfg: WorkerConfig, skip_model_load: bool = False) -> FastAPI:
    app = FastAPI(title=f"eu-kiki-worker-{cfg.port}")
    reg = CollectorRegistry()
    requests_total = Counter(
        "eu_kiki_worker_requests_total",
        "Requests",
        ["status"],
        registry=reg,
    )
    inference_latency = Histogram(
        "eu_kiki_worker_inference_seconds",
        "Inference latency",
        registry=reg,
    )
    semaphore = asyncio.Semaphore(1)

    runtime = MLXWorkerRuntime(cfg)
    if not skip_model_load:
        runtime.load_model()
        count = runtime.preload_adapters()
        log.info("Preloaded %d adapters on port %d", count, cfg.port)

    start_time = time.time()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": runtime.is_loaded,
            "port": cfg.port,
            "domains": cfg.domains,
            "uptime_s": int(time.time() - start_time),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> ChatCompletion:
        domain = request.headers.get("X-Lora-Domain", cfg.domains[0] if cfg.domains else "")

        try:
            async with semaphore:
                # MLX Metal requires GPU ops on main thread
                runtime.apply(domain)
                t0 = time.perf_counter()
                messages = [{"role": m.role, "content": m.content} for m in req.messages]
                text, _meta = runtime.generate(messages, req.max_tokens, req.temperature)
                latency = time.perf_counter() - t0
        except Exception as exc:
            log.exception("inference failed (domain=%s)", domain)
            requests_total.labels(status="500").inc()
            raise HTTPException(
                status_code=500,
                detail={
                    "type": "inference_error",
                    "domain": domain,
                    "message": str(exc),
                },
            ) from exc

        inference_latency.observe(latency)
        requests_total.labels(status="200").inc()

        return ChatCompletion(
            model=req.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=text))],
        )

    return app


def _load_config_from_yaml(path: str) -> WorkerConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    return WorkerConfig(**raw)


def make_apertus_app() -> FastAPI:
    return make_worker_app(_load_config_from_yaml("configs/apertus.yaml"))


def make_devstral_app() -> FastAPI:
    return make_worker_app(_load_config_from_yaml("configs/devstral.yaml"))


def make_eurollm_app() -> FastAPI:
    return make_worker_app(_load_config_from_yaml("configs/eurollm.yaml"))
