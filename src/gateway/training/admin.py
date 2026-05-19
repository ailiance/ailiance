"""Admin endpoints to control the medium35 training campaign."""
from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException, Request


def _check_token(token: str | None) -> None:
    expected = os.environ.get("AILIANCE_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="admin disabled (no token)")
    if token != expected:
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

    return router
