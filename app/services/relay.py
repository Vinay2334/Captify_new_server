"""Bidirectional relay between one extension client and one upstream session.

Two concurrent tasks per connection, no queues, no full-recording buffer:

- client -> upstream: validate each client event, enforce size / throughput
  limits, forward permitted audio events upstream (base64 payloads pass
  through untouched, never decoded and re-encoded).
- upstream -> client: map supported transcription events downstream, drop
  everything else, convert upstream errors to safe client errors.

When either side ends, both tasks are cancelled and both sockets closed.
"""

import asyncio
import base64
import binascii
import json
import time
from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.events import (
    AudioAppendEvent,
    ErrorEvent,
    TranscriptCompletedEvent,
    TranscriptDeltaEvent,
    UpstreamAudioAppendEvent,
    UpstreamError,
    map_upstream_event,
    parse_client_event,
)

log = get_logger(__name__)

UPSTREAM_SEND_TIMEOUT_SECONDS = 10.0

# Audio flow visibility: log an INFO summary at most this often / this many
# chunks so the terminal proves audio is arriving without spamming per-chunk.
# Only sizes, counts, and rates are logged - never base64 payloads.
AUDIO_LOG_INTERVAL_SECONDS = 5.0
AUDIO_LOG_INTERVAL_CHUNKS = 25


class ClientSocket(Protocol):
    async def send_text(self, message: str) -> None: ...
    async def receive_text(self) -> str: ...
    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class UpstreamSocket(Protocol):
    async def send_text(self, payload: str) -> None: ...
    async def recv_text(self) -> str: ...
    async def close(self) -> None: ...


class AudioThroughputLimiter:
    """Per-second cap on decoded audio bytes (1s sliding window)."""

    def __init__(self, max_bytes_per_second: int) -> None:
        self._max = max_bytes_per_second
        self._window_start = time.monotonic()
        self._used = 0

    def allow(self, num_bytes: int) -> bool:
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            self._window_start = now
            self._used = 0
        if self._used + num_bytes > self._max:
            return False
        self._used += num_bytes
        return True


def decode_audio_size(audio_b64: str) -> int:
    """Return decoded byte size of a base64 audio payload. Raises ValueError."""
    try:
        return len(base64.b64decode(audio_b64, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 audio payload") from exc


def to_upstream_payload(event: Any) -> str:
    """Map a validated client event to its upstream JSON (audio untouched)."""
    if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
        return json.dumps(
            {"type": "input_audio_buffer.append", "audio": event.audio}
        )
    return json.dumps({"type": "input_audio_buffer.commit"})


class RelaySession:
    def __init__(
        self,
        client: ClientSocket,
        upstream: UpstreamSocket,
        settings: Settings,
    ) -> None:
        self._client = client
        self._upstream = upstream
        self._settings = settings
        self._limiter = AudioThroughputLimiter(settings.max_audio_bytes_per_second)
        # Audio flow counters (metadata only, never payloads).
        self._audio_chunks = 0
        self._audio_bytes = 0
        self._commits = 0
        self._upstream_events = 0
        self._session_start = time.monotonic()
        self._last_audio_log = self._session_start
        self._last_logged_chunks = 0
        self._last_logged_bytes = 0

    def _log_audio_progress(self, *, force: bool = False) -> None:
        """Emit a throttled INFO summary proving audio is arriving."""
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
            "audio flow chunks=%d bytes=%d commits=%d "
            "avg_chunk=%.0fB overall=%.0fB/s interval=%.0fB/s elapsed=%.1fs",
            self._audio_chunks,
            self._audio_bytes,
            self._commits,
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
            "audio session ended reason=%s chunks=%d bytes=%d commits=%d "
            "upstream_events=%d elapsed=%.1fs overall=%.0fB/s",
            reason,
            self._audio_chunks,
            self._audio_bytes,
            self._commits,
            self._upstream_events,
            elapsed,
            self._audio_bytes / max(elapsed, 1e-6),
        )

    async def run(self) -> str:
        """Run both relay directions until either side ends. Returns reason."""
        log.info("audio session started")
        c2u = asyncio.create_task(
            self._client_to_upstream(), name="relay-client-to-upstream"
        )
        u2c = asyncio.create_task(
            self._upstream_to_client(), name="relay-upstream-to-client"
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
                except Exception as exc:  # relay task failure -> safe shutdown
                    log.warning(
                        "relay task failed type=%s", type(exc).__name__
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
            await self._close_both()

    async def _close_both(self) -> None:
        for sock in (self._client, self._upstream):
            try:
                await sock.close()
            except Exception:
                pass

    async def _send_client_error(self, code: str, message: str) -> None:
        try:
            await self._client.send_text(
                ErrorEvent(code=code, message=message).model_dump_json()
            )
        except Exception:
            pass

    async def _client_to_upstream(self) -> str:
        idle = self._settings.session_idle_timeout_seconds
        max_bytes = self._settings.max_websocket_message_bytes
        while True:
            try:
                raw = await asyncio.wait_for(
                    self._client.receive_text(), timeout=idle
                )
            except (asyncio.TimeoutError, TimeoutError):
                await self._send_client_error(
                    "idle_timeout", "session closed due to inactivity"
                )
                await self._client.close(code=1000, reason="idle timeout")
                return "idle_timeout"
            except Exception:
                # Client disconnected (Starlette raises on receive).
                return "client_disconnected"

            if len(raw.encode("utf-8")) > max_bytes:
                await self._send_client_error(
                    "message_too_large", "message exceeds size limit"
                )
                try:
                    await self._client.close(code=1009, reason="message too large")
                except Exception:
                    pass
                return "message_too_large"

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_client_error(
                    "invalid_event", "message must be a JSON object event"
                )
                continue
            if not isinstance(data, dict):
                await self._send_client_error(
                    "invalid_event", "message must be a JSON object event"
                )
                continue

            try:
                event = parse_client_event(data)
            except ValueError:
                await self._send_client_error(
                    "invalid_event", "unsupported or malformed event"
                )
                continue

            if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
                try:
                    size = decode_audio_size(event.audio)
                except ValueError:
                    log.warning("audio chunk rejected: invalid base64 payload")
                    await self._send_client_error(
                        "invalid_event", "audio payload is not valid base64"
                    )
                    continue
                if not self._limiter.allow(size):
                    log.warning(
                        "audio chunk rate-limited size=%dB budget=%dB/s "
                        "chunks=%d total_bytes=%d",
                        size,
                        self._settings.max_audio_bytes_per_second,
                        self._audio_chunks,
                        self._audio_bytes,
                    )
                    await self._send_client_error(
                        "rate_limited", "audio throughput limit exceeded"
                    )
                    continue
                self._audio_chunks += 1
                self._audio_bytes += size
                if self._audio_chunks == 1:
                    log.info(
                        "audio stream started first_chunk=%dB type=%s",
                        size,
                        event.type,
                    )
                else:
                    log.debug(
                        "audio chunk #%d size=%dB total=%dB",
                        self._audio_chunks,
                        size,
                        self._audio_bytes,
                    )
                self._log_audio_progress()

            is_commit = event.type in (
                "audio.commit",
                "input_audio_buffer.commit",
            )
            try:
                await asyncio.wait_for(
                    self._upstream.send_text(to_upstream_payload(event)),
                    timeout=UPSTREAM_SEND_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                log.warning(
                    "upstream send failed type=%s", type(exc).__name__
                )
                await self._send_client_error(
                    "upstream_error", "transcription service unavailable"
                )
                return "upstream_send_failed"
            if is_commit:
                self._commits += 1
                log.info(
                    "audio commit #%d forwarded after chunks=%d bytes=%d",
                    self._commits,
                    self._audio_chunks,
                    self._audio_bytes,
                )
            else:
                log.debug("forwarded upstream type=%s", event.type)

    async def _upstream_to_client(self) -> str:
        while True:
            try:
                raw = await self._upstream.recv_text()
            except Exception:
                return "upstream_disconnected"
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("upstream sent malformed JSON")
                continue
            if not isinstance(event, dict):
                continue
            self._upstream_events += 1
            try:
                mapped = map_upstream_event(event)
            except UpstreamError as exc:
                log.warning("upstream error code=%s", exc.code)
                try:
                    await self._client.send_text(
                        exc.to_client_event().model_dump_json()
                    )
                except Exception:
                    pass
                return f"upstream_error:{exc.code}"
            if mapped is None:
                log.debug(
                    "upstream event ignored type=%s", event.get("type")
                )
                continue
            if isinstance(mapped, TranscriptDeltaEvent):
                log.info(
                    "transcript delta item_id=%s delta=%s",
                    mapped.item_id,
                    mapped.delta,
                )
            elif isinstance(mapped, TranscriptCompletedEvent):
                log.info(
                    "transcript completed item_id=%s transcript=%s",
                    mapped.item_id,
                    mapped.transcript,
                )
            try:
                await self._client.send_text(mapped.model_dump_json())
            except Exception:
                return "client_disconnected"
