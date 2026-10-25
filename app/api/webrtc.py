"""WebRTC signaling endpoint for the extensionRTC build.

``POST /api/v1/transcription/webrtc/offer`` exchanges an SDP offer for an
SDP answer (non-trickle ICE: the extension's offer already contains all
candidates, and the answer is returned only after local gathering
completes). Audio then flows as raw Int16 PCM messages over the ``audio``
``RTCDataChannel`` (see ``app.services.webrtc_session``), not over a
WebSocket.

Session accounting mirrors the WebSocket endpoint: unauthenticated local
development, origin checks, one slot per session, one upstream OpenAI
socket per session. Origin checks and connection caps are resource
protections, not access control: anyone who can reach the endpoint may
consume OpenAI API quota. Do not expose it publicly without added abuse
protections.
"""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.openai_realtime import OpenAIRealtimeClient
from app.services.webrtc_session import WebRTCRelaySession
from app.schemas.events import UpstreamError

log = get_logger(__name__)

router = APIRouter()


class OfferRequest(BaseModel):
    sdp: str = Field(min_length=1)
    type: str = "offer"


class _OfferFailure(Exception):
    """Expected offer rejection with an HTTP status and safe error body."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "code": code, "message": message},
    )


async def _wait_ice_gathering_complete(pc: object, timeout: float = 5.0) -> None:
    """Wait for non-trickle ICE gathering, like the extension does.

    The extension waits for its own gathering before POSTing the offer,
    so the answer must likewise carry all server candidates.
    """
    if getattr(pc, "iceGatheringState", "complete") == "complete":
        return
    loop = asyncio.get_running_loop()
    done: asyncio.Future = loop.create_future()

    def _check() -> None:
        if (
            getattr(pc, "iceGatheringState", None) == "complete"
            and not done.done()
        ):
            done.set_result(True)

    pc.on("icegatheringstatechange", _check)  # type: ignore[attr-defined]
    try:
        await asyncio.wait_for(done, timeout)
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("ICE gathering timed out; answering with trickle state")
    finally:
        try:
            pc.remove_listener("icegatheringstatechange", _check)  # type: ignore[attr-defined]
        except Exception:
            pass


@router.post("/api/v1/transcription/webrtc/offer")
async def webrtc_offer(payload: OfferRequest, request: Request):
    settings: Settings = request.app.state.settings

    origin = request.headers.get("origin")
    if not settings.is_origin_allowed(origin):
        log.warning("webrtc offer rejected: disallowed origin")
        return _error(403, "origin_not_allowed", "origin not allowed")

    if not settings.openai_api_key:
        log.warning("webrtc offer rejected: OPENAI_API_KEY is not set")
        return _error(
            503, "not_configured", "translation service is not configured"
        )

    state = request.app.state
    async with state.slot_lock:
        if state.active_sessions >= settings.max_concurrent_sessions:
            full = True
        else:
            state.active_sessions += 1
            full = False
    if full:
        log.warning("webrtc offer rejected: session limit reached")
        return _error(429, "server_busy", "too many active sessions")

    try:
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ImportError:
            raise _OfferFailure(
                503, "webrtc_unavailable", "webrtc support is not installed"
            )

        if payload.type != "offer":
            raise _OfferFailure(
                400, "invalid_offer", "expected an SDP offer"
            )

        # Connect upstream BEFORE answering so auth/outage failures surface
        # as HTTP errors, mirroring the WebSocket endpoint (which connects
        # before sending session.ready).
        upstream = request.app.state.upstream_factory()
        if not isinstance(upstream, OpenAIRealtimeClient):
            # Tests inject fakes through upstream_factory; accept anything
            # with the connect/close interface.
            pass
        try:
            await upstream.connect()
        except UpstreamError as exc:
            raise _OfferFailure(502, exc.code, exc.message)
        except Exception:
            log.warning("webrtc upstream connect failed unexpectedly")
            raise _OfferFailure(
                502, "upstream_error", "translation service unavailable"
            )

        pc = RTCPeerConnection()
        state.webrtc_peers.add(pc)
        channel_holder: dict = {}

        @pc.on("datachannel")
        def _on_datachannel(channel) -> None:
            if channel.label != settings.webrtc_data_channel_label:
                log.warning(
                    "ignoring datachannel with unexpected label=%r",
                    channel.label,
                )
                return
            if "channel" not in channel_holder:
                channel_holder["channel"] = channel

        @pc.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            log.info("webrtc connection state=%s", pc.connectionState)

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=payload.sdp, type="offer")
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await _wait_ice_gathering_complete(pc)
        except _OfferFailure:
            raise
        except Exception as exc:
            log.warning(
                "webrtc SDP negotiation failed type=%s", type(exc).__name__
            )
            try:
                await pc.close()
            except Exception:
                pass
            state.webrtc_peers.discard(pc)
            try:
                await upstream.close()
            except Exception:
                pass
            raise _OfferFailure(
                400, "invalid_offer", "could not negotiate SDP offer"
            )

        local = pc.localDescription
        task = asyncio.create_task(
            _serve_webrtc_session(pc, upstream, channel_holder, settings, state),
            name="webrtc-session",
        )
        request.app.state.background_tasks.add(task)
        task.add_done_callback(request.app.state.background_tasks.discard)
        log.info("webrtc offer answered; session starting")
        return {"sdp": local.sdp, "type": local.type}
    except _OfferFailure as failure:
        async with state.slot_lock:
            state.active_sessions = max(0, state.active_sessions - 1)
        return _error(failure.status, failure.code, failure.message)
    except Exception as exc:
        log.warning("webrtc offer failed type=%s", type(exc).__name__)
        async with state.slot_lock:
            state.active_sessions = max(0, state.active_sessions - 1)
        return _error(500, "internal_error", "could not handle SDP offer")


async def _serve_webrtc_session(pc, upstream, channel_holder, settings, state) -> None:
    """Own one answered session until the channel or upstream ends."""
    reason = "not_started"
    try:
        channel = await _wait_for_channel(channel_holder, settings)
        if channel is None:
            reason = "no_data_channel"
            log.warning("webrtc session ended: datachannel never arrived")
            return
        if not await _wait_for_open(channel, settings):
            reason = "channel_never_opened"
            log.warning("webrtc session ended: datachannel never opened")
            return
        session = WebRTCRelaySession(channel, upstream, settings)

        @channel.on("message")
        def _on_message(message) -> None:
            session.on_channel_message(message)

        @channel.on("close")
        def _on_close() -> None:
            session.on_channel_close()

        reason = await session.run()
    except asyncio.CancelledError:
        reason = "cancelled"
        raise
    except Exception as exc:
        log.warning("webrtc session failed type=%s", type(exc).__name__)
        reason = "internal_error"
    finally:
        try:
            await upstream.close()
        except Exception:
            pass
        try:
            await pc.close()
        except Exception:
            pass
        state.webrtc_peers.discard(pc)
        async with state.slot_lock:
            state.active_sessions = max(0, state.active_sessions - 1)
        log.info("webrtc session ended reason=%s", reason)


async def _wait_for_channel(channel_holder: dict, settings: Settings):
    """The datachannel event fires during setRemoteDescription, usually
    before this runs; still allow a grace period for slow transports."""
    if "channel" in channel_holder:
        return channel_holder["channel"]
    deadline = settings.session_idle_timeout_seconds
    waited = 0.0
    while waited < deadline:
        await asyncio.sleep(0.1)
        waited += 0.1
        if "channel" in channel_holder:
            return channel_holder["channel"]
    return None


async def _wait_for_open(channel, settings: Settings) -> bool:
    if getattr(channel, "readyState", "connecting") == "open":
        return True
    loop = asyncio.get_running_loop()
    opened: asyncio.Future = loop.create_future()

    def _on_open() -> None:
        if not opened.done():
            opened.set_result(True)

    def _on_close() -> None:
        if not opened.done():
            opened.set_result(False)

    channel.on("open", _on_open)
    channel.on("close", _on_close)
    try:
        return await asyncio.wait_for(
            opened, timeout=settings.session_idle_timeout_seconds
        )
    except (asyncio.TimeoutError, TimeoutError):
        return False
    finally:
        for event, cb in (("open", _on_open), ("close", _on_close)):
            try:
                channel.remove_listener(event, cb)
            except Exception:
                pass
