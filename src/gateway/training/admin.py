"""Admin endpoints to control the medium35 training campaign."""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse


def _check_token(token: str | None) -> None:
    expected = os.environ.get("AILIANCE_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="admin disabled (no token)")
    if not hmac.compare_digest(token or "", expected):
        raise HTTPException(status_code=401, detail="invalid admin token")


def make_training_router() -> APIRouter:
    router = APIRouter(prefix="/admin/training", tags=["training"])

    @router.get("/status")
    async def status(request: Request,
                     x_admin_token: str | None = Header(default=None)):
        _check_token(x_admin_token)
        return request.app.state.training.status()

    @router.post("/start", status_code=202)
    async def start(request: Request, body: dict | None = None,
                    x_admin_token: str | None = Header(default=None)):
        _check_token(x_admin_token)
        domains = (body or {}).get("domains")
        try:
            await request.app.state.training.start(domains)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"accepted": True}

    @router.post("/abort")
    async def abort(request: Request,
                    x_admin_token: str | None = Header(default=None)):
        _check_token(x_admin_token)
        await request.app.state.training.abort()
        return {"aborted": True}

    @router.get("/log/{domain}", response_class=PlainTextResponse)
    async def domain_log(request: Request, domain: str, tail: int = 100,
                         x_admin_token: str | None = Header(default=None)):
        _check_token(x_admin_token)
        tail = max(1, min(tail, 1000))
        ops = request.app.state.training._ops  # noqa: SLF001 — same package
        text = await ops.read_domain_log(domain)
        if not text:
            return ""
        return "\n".join(text.splitlines()[-tail:])

    return router
