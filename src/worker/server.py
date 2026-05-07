"""Worker FastAPI server — serves one MLX model with LoRA hot-swap."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.worker.function_calling import (
    inject_tools_into_messages,
    parse_tool_call_from_text,
    sse_chunk_finish,
    sse_chunk_for_content,
    sse_chunk_for_tool_call,
    sse_done,
)
from src.worker.runtime import MLXWorkerRuntime, WorkerConfig
from src.worker.schemas import ChatCompletion, ChatCompletionRequest, ChatMessage, Choice

log = logging.getLogger(__name__)


def make_worker_app(cfg: WorkerConfig, skip_model_load: bool = False) -> FastAPI:
    app = FastAPI(title=f"ailiance-worker-{cfg.port}")
    reg = CollectorRegistry()
    requests_total = Counter(
        "ailiance_worker_requests_total",
        "Requests",
        ["status"],
        registry=reg,
    )
    inference_latency = Histogram(
        "ailiance_worker_inference_seconds",
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
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        domain = request.headers.get("X-Lora-Domain", cfg.domains[0] if cfg.domains else "")

        # Build messages: inject tool spec into system prompt when tools given.
        if req.tools:
            messages = inject_tools_into_messages(req.messages, req.tools)
        else:
            messages = [
                {"role": m.role, "content": m.content or ""} for m in req.messages
            ]

        try:
            async with semaphore:
                # MLX Metal requires GPU ops on main thread
                runtime.apply(domain)
                t0 = time.perf_counter()
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

        # If tools were provided, try to parse a tool call from the output.
        tool_calls = None
        content_text = text
        if req.tools:
            content_text, tool_calls = parse_tool_call_from_text(text)
            if tool_calls is None:
                content_text = text

        finish_reason = "tool_calls" if tool_calls else "stop"

        # Rough token accounting (model-agnostic placeholder).
        prompt_tokens = sum(len((m.get("content") or "").split()) for m in messages)
        completion_tokens = len((text or "").split())

        if req.stream:
            async def event_stream() -> AsyncIterator[str]:
                if tool_calls:
                    yield sse_chunk_for_tool_call(req.model, tool_calls)
                else:
                    yield sse_chunk_for_content(req.model, content_text)
                yield sse_chunk_finish(
                    req.model, finish_reason, prompt_tokens, completion_tokens
                )
                yield sse_done()

            return StreamingResponse(
                event_stream(), media_type="text/event-stream"
            )

        # Non-streaming response.
        message = ChatMessage(
            role="assistant",
            content=None if tool_calls else content_text,
            tool_calls=tool_calls,
        )
        return ChatCompletion(
            model=req.model,
            choices=[Choice(message=message, finish_reason=finish_reason)],
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


def make_gemma4_app() -> FastAPI:
    return make_worker_app(_load_config_from_yaml("configs/gemma4.yaml"))
