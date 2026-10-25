"""Bridge between a WebRTC DataChannel and one OpenAI Realtime session.

Extension contract (see ``extensionRTC/offscreen.js`` + ``rtc-config.js``):

- One reliable + ordered ``RTCDataChannel`` labelled ``"audio"``.
- Binary messages: raw little-endian Int16 mono PCM at 24 kHz, nominally
  200 ms per chunk (4800 samples = 9600 bytes). The final flush of a
  recording may be shorter. No JSON framing and no base64 on the wire.
- Text messages: optional JSON client events using the same schema as the
  WebSocket endpoint (``audio.commit`` et al), for forward compatibility.
  The extension currently never sends commits.
- The upstream translation session is continuous (no commits, no turn
  lifecycle), so this bridge forwards audio as it arrives: no auto-commit
  timer and no final commit on channel close.
- Translated transcripts flow back as JSON text on the same channel
  (``session.ready`` / ``transcript.delta`` / ``error``), mirroring the
  WebSocket endpoint's downstream schema. Translation emits deltas only,
  so ``transcript.completed`` never fires. The current extension build
  only logs them.
"""

import asyncio
import base64
import binascii
import json
import time
from typing import Any, Callable, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.events import (
    AudioAppendEvent,
    ErrorEvent,
    LanguageConfigEvent,
    SessionReadyEvent,
    TranscriptCompletedEvent,
    TranscriptDeltaEvent,
    UpstreamAudioAppendEvent,
    UpstreamError,
    map_upstream_event,
    parse_client_event,
)
from app.services.openai_realtime import build_session_update
from app.services.relay import (
    UPSTREAM_SEND_TIMEOUT_SECONDS,
    AudioThroughputLimiter,
)

log = get_logger(__name__)

# Audio flow visibility, mirroring relay.py: INFO summaries only, never
# payloads. Only sizes, counts, and rates are logged.
AUDIO_LOG_INTERVAL_SECONDS = 5.0
AUDIO_LOG_INTERVAL_CHUNKS = 25


class DataChannel(Protocol):
    """Structural subset of aiortc's RTCDataChannel used by the session."""

    @property
    def readyState(self) -> str: ...  # noqa: N802  (Web API casing)

    def send(self, message: bytes | str) -> None: ...


def webrtc_audio_to_upstream_payload(raw: bytes) -> str:
    """Validate one raw PCM16 DataChannel message and map it upstream.

    Returns the ``session.input_audio_buffer.append`` JSON payload with the
    audio base64-encoded (upstream only accepts base64). Raises ValueError
    for empty or odd-length payloads, which cannot be Int16 samples.
    """
    if not raw or len(raw) % 2 != 0:
        raise ValueError("audio payload must be non-empty Int16 PCM bytes")
    try:
        audio_b64 = base64.b64encode(raw).decode("ascii")
    except (binascii.Error, ValueError) as exc:
        raise ValueError("could not encode audio payload") from exc
    return json.dumps(
        {"type": "session.input_audio_buffer.append", "audio": audio_b64}
    )


class WebRTCRelaySession:
    """Relay between one audio DataChannel and one upstream session.

    Inbound DataChannel messages arrive via the sync ``on_channel_message``
    callback (invoked by aiortc); ``on_channel_close`` signals teardown.
    ``run`` pumps both directions until either side ends and returns the
    reason. The caller owns the peer connection and closes it afterwards.
    """

    def __init__(
        self,
        channel: DataChannel,
        upstream: Any,
        settings: Settings,
        on_message: Callable[[bytes | str], None] | None = None,
    ) -> None:
        self._channel = channel
        self._upstream = upstream
        self._settings = settings
        self._limiter = AudioThroughputLimiter(settings.max_audio_bytes_per_second)
        self._queue: asyncio.Queue = asyncio.Queue()
        # Test hook: lets tests observe raw channel traffic if needed.
        self._on_message = on_message
        self._audio_chunks = 0
        self._audio_bytes = 0
        self._upstream_events = 0
        # Target language already applied upstream by the initial
        # session.update sent at connect time. Repeat language-config
        # events for the same language are skipped, never re-sent.
        self._applied_target_language = settings.translation_target_language
        self._session_start = time.monotonic()
        self._last_audio_log = self._session_start
        self._last_logged_chunks = 0
        self._last_logged_bytes = 0

    # --- inbound entry points (called synchronously by aiortc) ---

    def on_channel_message(self, message: bytes | str) -> None:
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            log.warning("dropping datachannel message: session queue full")

    def on_channel_close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # --- logging helpers (metadata only, never payloads) ---

    def _log_audio_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        chunks_since = self._audio_chunks - self._last_logged_chunks
        elapsed_since = now - self._last_audio_log
        if not force and (
            chunks_since < AUDIO_LOG_INTERVAL_CHUNKS
            and elapsed_since < AUDIO_LOG_INTERVAL_SECONDS
        ):
            return
        bytes_since = self._audio_bytes - self._last_logged_bytes
        total_elapsed = max(now - self._session_start, 1e-6)
        interval_rate = (
            bytes_since / max(elapsed_since, 1e-6) if elapsed_since > 0 else 0.0
        )
        avg_chunk = (
            self._audio_bytes / self._audio_chunks if self._audio_chunks else 0.0
        )
        log.info(
            "webrtc audio flow chunks=%d bytes=%d "
            "avg_chunk=%.0fB overall=%.0fB/s interval=%.0fB/s elapsed=%.1fs",
            self._audio_chunks,
            self._audio_bytes,
            avg_chunk,
            self._audio_bytes / total_elapsed,
            interval_rate,
            total_elapsed,
        )
        self._last_audio_log = now
        self._last_logged_chunks = self._audio_chunks
        self._last_logged_bytes = self._audio_bytes

    def _log_session_summary(self, reason: str) -> None:
        elapsed = time.monotonic() - self._session_start
        log.info(
            "webrtc audio session ended reason=%s chunks=%d bytes=%d "
            "upstream_events=%d elapsed=%.1fs overall=%.0fB/s",
            reason,
            self._audio_chunks,
            self._audio_bytes,
            self._upstream_events,
            elapsed,
            self._audio_bytes / max(elapsed, 1e-6),
        )

    # --- run ---

    async def run(self) -> str:
        """Run both relay directions until either side ends. Returns reason."""
        log.info("webrtc audio session started")
        try:
            self._channel.send(
                SessionReadyEvent(
                    model=self._settings.openai_realtime_model
                ).model_dump_json()
            )
        except Exception:
            return "channel_closed"
        c2u = asyncio.create_task(
            self._channel_to_upstream(), name="webrtc-channel-to-upstream"
        )
        u2c = asyncio.create_task(
            self._upstream_to_channel(), name="webrtc-upstream-to-channel"
        )
        try:
            done, _ = await asyncio.wait(
                {c2u, u2c}, return_when=asyncio.FIRST_COMPLETED
            )
            reason = "closed"
            for task in done:
                try:
                    result = task.result()
                    if isinstance(result, str):
                        reason = result
                except asyncio.CancelledError:
                    reason = "cancelled"
                except Exception as exc:
                    log.warning(
                        "webrtc relay task failed type=%s", type(exc).__name__
                    )
                    reason = f"relay_error:{type(exc).__name__}"
            self._log_audio_progress(force=True)
            self._log_session_summary(reason)
            return reason
        finally:
            for task in (c2u, u2c):
                if not task.done():
                    task.cancel()
            await asyncio.gather(c2u, u2c, return_exceptions=True)
            try:
                await self._upstream.close()
            except Exception:
                pass

    def _send_channel_error(self, code: str, message: str) -> None:
        try:
            self._channel.send(
                ErrorEvent(code=code, message=message).model_dump_json()
            )
        except Exception:
            pass

    async def _apply_language_config(
        self, event: LanguageConfigEvent
    ) -> str | None:
        """Forward a target-language change upstream. Returns stop reason."""
        if event.target_lang == self._applied_target_language:
            log.debug("language unchanged target=%s", event.target_lang)
            return None
        payload = build_session_update(
            event.target_lang,
            self._settings.translation_transcription_model,
        )
        ok = await self._forward_upstream(payload)
        if not ok:
            return "upstream_send_failed"
        self._applied_target_language = event.target_lang
        log.info(
            "webrtc translation target language updated target=%s",
            event.target_lang,
        )
        return None

    async def _forward_upstream(self, payload: str) -> bool:
        """Send one payload upstream. Returns False when upstream is gone."""
        try:
            await asyncio.wait_for(
                self._upstream.send_text(payload),
                timeout=UPSTREAM_SEND_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            log.warning(
                "webrtc upstream send failed type=%s", type(exc).__name__
            )
            self._send_channel_error(
                "upstream_error", "translation service unavailable"
            )
            return False
        return True

    def _count_audio_bytes(self, size: int, *, source: str) -> bool:
        """Apply the throughput limiter and count one chunk. True if kept."""
        if not self._limiter.allow(size):
            log.warning(
                "webrtc audio chunk rate-limited size=%dB budget=%dB/s "
                "chunks=%d total_bytes=%d",
                size,
                self._settings.max_audio_bytes_per_second,
                self._audio_chunks,
                self._audio_bytes,
            )
            self._send_channel_error(
                "rate_limited", "audio throughput limit exceeded"
            )
            return False
        self._audio_chunks += 1
        self._audio_bytes += size
        if self._audio_chunks == 1:
            log.info(
                "webrtc audio stream started first_chunk=%dB source=%s",
                size,
                source,
            )
        self._log_audio_progress()
        return True

    async def _handle_binary_audio(self, raw: bytes) -> str | None:
        """Validate + forward one PCM16 message. Returns a stop reason."""
        if len(raw) > self._settings.max_websocket_message_bytes:
            self._send_channel_error(
                "message_too_large", "message exceeds size limit"
            )
            return "message_too_large"
        try:
            payload = webrtc_audio_to_upstream_payload(raw)
        except ValueError:
            log.warning("webrtc audio chunk rejected: not Int16 PCM bytes")
            self._send_channel_error(
                "invalid_event", "audio payload is not Int16 PCM"
            )
            return None
        if not self._count_audio_bytes(len(raw), source="pcm16"):
            return None
        ok = await self._forward_upstream(payload)
        if not ok:
            return "upstream_send_failed"
        return None

    async def _handle_text_event(self, text: str) -> str | None:
        """Handle one JSON text message. Returns a stop reason, if any."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self._send_channel_error(
                "invalid_event", "message must be a JSON object event"
            )
            return None
        if not isinstance(data, dict):
            self._send_channel_error(
                "invalid_event", "message must be a JSON object event"
            )
            return None
        try:
            event = parse_client_event(data)
        except ValueError:
            self._send_channel_error(
                "invalid_event", "unsupported or malformed event"
            )
            return None
        if isinstance(event, LanguageConfigEvent):
            return await self._apply_language_config(event)
        if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
            try:
                import base64 as _b64

                size = len(_b64.b64decode(event.audio, validate=True))
            except Exception:
                self._send_channel_error(
                    "invalid_event", "audio payload is not valid base64"
                )
                return None
            if not self._count_audio_bytes(size, source="base64"):
                return None
            payload = json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": event.audio,
                }
            )
            ok = await self._forward_upstream(payload)
            if not ok:
                return "upstream_send_failed"
            return None
        # Commit (audio.commit / input_audio_buffer.commit): accepted for
        # extension compatibility but swallowed, never forwarded. The
        # translation session is continuous and has no commit lifecycle.
        log.debug("commit swallowed type=%s", event.type)
        return None

    async def _channel_to_upstream(self) -> str:
        idle = self._settings.session_idle_timeout_seconds
        while True:
            try:
                message = await asyncio.wait_for(
                    self._queue.get(), timeout=idle
                )
            except (asyncio.TimeoutError, TimeoutError):
                self._send_channel_error(
                    "idle_timeout", "session closed due to inactivity"
                )
                return "idle_timeout"
            if message is None:
                # DataChannel closed. The translation session is
                # continuous, so there is nothing to flush: the tail of
                # the recording was already streamed.
                return "channel_closed"
            if isinstance(message, str):
                stop = await self._handle_text_event(message)
            elif isinstance(message, (bytes, bytearray, memoryview)):
                stop = await self._handle_binary_audio(bytes(message))
            else:
                self._send_channel_error(
                    "invalid_event", "unsupported message type"
                )
                continue
            if stop is not None:
                return stop

    async def _upstream_to_channel(self) -> str:
        while True:
            try:
                raw = await self._upstream.recv_text()
            except Exception:
                return "upstream_disconnected"
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("webrtc upstream sent malformed JSON")
                continue
            if not isinstance(event, dict):
                continue
            self._upstream_events += 1
            try:
                mapped = map_upstream_event(event)
            except UpstreamError as exc:
                log.warning("webrtc upstream error code=%s", exc.code)
                self._send_channel_error(exc.code, exc.message)
                return f"upstream_error:{exc.code}"
            if mapped is None:
                continue
            if isinstance(mapped, TranscriptDeltaEvent):
                log.info(
                    "webrtc transcript delta item_id=%s delta=%s",
                    mapped.item_id,
                    mapped.delta,
                )
            elif isinstance(mapped, TranscriptCompletedEvent):
                log.info(
                    "webrtc transcript completed item_id=%s transcript=%s",
                    mapped.item_id,
                    mapped.transcript,
                )
            try:
                self._channel.send(mapped.model_dump_json())
            except Exception:
                return "channel_closed"
