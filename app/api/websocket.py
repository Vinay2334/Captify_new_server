"""WebSocket endpoint for real-time translation.

``/api/v1/transcription/ws`` requires a Supabase access token when the
backend is configured with ``SUPABASE_URL`` (``Authorization: Bearer``
header or ``?token=`` query parameter): every session is then tied to
the verified user and billed against their quota. Without Supabase
configuration it keeps the previous local-development behavior (open,
with a startup warning in ``app.main``). Origin checks, message limits,
idle timeouts, and connection caps are resource protections, not access
control. Do not expose this endpoint publicly without added guards.
"""

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.supabase_auth import AuthError, verify_websocket_user
from app.schemas.events import SessionReadyEvent
from app.services.quota import (
    QuotaDenied,
    SessionQuotaTracker,
    begin_tracked_session,
)
from app.services.relay import RelaySession
from app.services.stt_factory import STTWorkflowFactory

log = get_logger(__name__)

router = APIRouter()


async def _reject(websocket: WebSocket, code: str, message: str, close_code: int) -> None:
    try:
        await websocket.send_text(
            json.dumps({"type": "error", "code": code, "message": message})
        )
    except Exception:
        pass
    try:
        await websocket.close(code=close_code, reason=message)
    except Exception:
        pass


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

    quota: SessionQuotaTracker | None = None
    if settings.auth_enabled():
        try:
            user = verify_websocket_user(websocket, settings)
        except AuthError as exc:
            await _reject(websocket, "auth_expired", exc.message, 4401)
            log.warning("rejected websocket: auth failed")
            async with state.slot_lock:
                state.active_sessions = max(0, state.active_sessions - 1)
            return
        # Optional idempotency key for reconnects (?session=); otherwise
        # each connection bills exactly its own audio under a fresh id.
        client_session_id = (
            websocket.query_params.get("session")
            or websocket.query_params.get("client_session_id")
            or f"ws-{uuid.uuid4().hex}"
        )
        try:
            quota = await begin_tracked_session(
                state.quota_store,
                user.id,
                client_session_id,
                min_start_seconds=settings.quota_min_start_seconds,
            )
        except Exception as exc:
            if isinstance(exc, QuotaDenied):
                await _reject(
                    websocket,
                    "quota_exceeded",
                    "Subtitle quota exhausted. Upgrade your plan to continue.",
                    1008,
                )
                log.info("rejected websocket: quota exhausted user=%s", user.id)
            else:
                await _reject(
                    websocket, "internal_error", "could not start session", 1011
                )
                log.warning(
                    "quota authorize failed type=%s", type(exc).__name__
                )
            async with state.slot_lock:
                state.active_sessions = max(0, state.active_sessions - 1)
            return
        log.info("websocket session authorized user=%s", user.id)

    try:
        await _handle_session(websocket, settings, quota=quota)
    finally:
        async with state.slot_lock:
            state.active_sessions = max(0, state.active_sessions - 1)


async def _handle_session(
    websocket: WebSocket, settings: Settings, quota: SessionQuotaTracker | None = None
) -> None:
    async def _abort(reason: str) -> None:
        # Sessions that never reached the relay still finalize (0 billed,
        # marked failed) so reconnects with the same key resume cleanly.
        if quota is not None:
            try:
                await quota.finish(reason)
            except Exception:
                pass

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
        await _abort("not_configured")
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
        await _abort(f"upstream_error:{code}")
        return

    try:
        await websocket.send_text(
            SessionReadyEvent(model=settings.active_model())
            .model_dump_json()
        )
    except Exception:
        await upstream.close()
        await _abort("client_disconnected")
        return

    relay = RelaySession(client=websocket, upstream=upstream, settings=settings, quota=quota)
    try:
        reason = await relay.run()
    except asyncio.CancelledError:
        # Server shutdown: still commit what was actually generated.
        if quota is not None:
            try:
                await quota.finish("cancelled")
            except Exception:
                pass
        raise
    except Exception as exc:
        log.warning("session failed type=%s", type(exc).__name__)
        reason = "internal_error"
    log.info("session ended reason=%s", reason)
