"""WebSocket endpoint for real-time translation.

``/api/v1/transcription/ws`` accepts unauthenticated connections (local
development). This is NOT authentication: anyone who can reach the endpoint
can consume API quota. Origin checks, message limits, idle timeouts,
and connection caps are resource protections, not access control. Do not
expose this endpoint publicly without added abuse protections.
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.events import SessionReadyEvent
from app.services.relay import RelaySession
from app.services.stt_factory import STTWorkflowFactory

log = get_logger(__name__)

router = APIRouter()


def build_upstream_client(settings: Settings) -> Any:
    """Build the upstream client for the configured ``STT_PROVIDER``."""
    return STTWorkflowFactory.create_upstream_client(settings)


@router.websocket("/api/v1/transcription/ws")
async def transcription_ws(websocket: WebSocket) -> None:
    settings: Settings = websocket.app.state.settings

    origin = websocket.headers.get("origin")
    if not settings.is_origin_allowed(origin):
        # Must accept before we can send a close frame with a reason.
        await websocket.accept()
        try:
            await websocket.close(code=1008, reason="origin not allowed")
        except Exception:
            pass
        log.warning("rejected connection from disallowed origin")
        return

    await websocket.accept()

    state = websocket.app.state
    async with state.slot_lock:
        if state.active_sessions >= settings.max_concurrent_sessions:
            full = True
        else:
            state.active_sessions += 1
            full = False
    if full:
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "code": "server_busy",
                        "message": "too many active sessions",
                    }
                )
            )
            await websocket.close(code=1013, reason="server busy")
        except Exception:
            pass
        log.warning("rejected connection: session limit reached")
        return

    try:
        await _handle_session(websocket, settings)
    finally:
        async with state.slot_lock:
            state.active_sessions = max(0, state.active_sessions - 1)


async def _handle_session(websocket: WebSocket, settings: Settings) -> None:
    if not settings.upstream_api_key():
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "code": "not_configured",
                        "message": "translation service is not configured",
                    }
                )
            )
            await websocket.close(code=1011, reason="service not configured")
        except Exception:
            pass
        log.warning("rejected connection: translation API key is not set")
        return

    factory = websocket.app.state.upstream_factory
    upstream = factory()
    try:
        await upstream.connect()
    except Exception as exc:
        code = getattr(exc, "code", "upstream_error")
        message = getattr(exc, "message", "translation service unavailable")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "code": code, "message": message})
            )
            await websocket.close(code=1011, reason="upstream unavailable")
        except Exception:
            pass
        log.warning("upstream connect failed code=%s", code)
        try:
            await upstream.close()
        except Exception:
            pass
        return

    try:
        await websocket.send_text(
            SessionReadyEvent(model=settings.active_model())
            .model_dump_json()
        )
    except Exception:
        await upstream.close()
        return

    relay = RelaySession(client=websocket, upstream=upstream, settings=settings)
    try:
        reason = await relay.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("session failed type=%s", type(exc).__name__)
        reason = "internal_error"
    log.info("session ended reason=%s", reason)
