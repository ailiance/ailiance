"""FastAPI sub-router that exposes /v1/realtime.

Mount in the ailiance gateway main app:

    from src.realtime.router import router as realtime_router
    app.include_router(realtime_router)

One WebSocket connection ⇄ one ``RealtimeSession`` ⇄ one upstream
Kyutai WS. The gateway's existing auth middleware should apply to the
WebSocket handshake; reuse the same bearer-token scheme as
/v1/chat/completions.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from .session import RealtimeSession


log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    await ws.accept()
    session = RealtimeSession(ws)
    try:
        await session.run()
    except Exception:
        log.exception("realtime session crashed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
