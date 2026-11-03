"""Bidirectional relay between one extension client and one upstream session.

Two concurrent tasks per connection, no queues, no full-recording buffer:

- client -> upstream: validate each client event, enforce size / throughput
  limits, forward permitted audio events upstream (base64 payloads pass
  through untouched, never decoded and re-encoded). The translation
  session is continuous, so in translation mode client commit events are
  accepted (the extension still sends them) but swallowed, never
  forwarded. In transcription mode commits close the turn and are
  forwarded upstream.
- upstream -> client: map supported translation transcript events
  downstream, drop everything else, convert upstream errors to safe
  client errors.

When either side ends, both tasks are cancelled and both sockets closed.
"""

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.events import (
    AudioAppendEvent,
    AudioCommitEvent,
    ErrorEvent,
    LanguageConfigEvent,
    TranscriptCompletedEvent,
    TranscriptDeltaEvent,
    UpstreamAudioAppendEvent,
    UpstreamAudioCommitEvent,
    UpstreamError,
    map_upstream_event,
    is_same_language,
    parse_client_event,
)
from app.services.stt_factory import STTWorkflowFactory
from app.services.subtitle_factory import SubtitleError, SubtitleFactory
from app.services.openai_realtime import (
    build_session_update,
    build_transcription_session_update,
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


def _transcription_payload(event: Any) -> str | None:
    """Upstream JSON for transcription mode (commit/turn lifecycle)."""
    if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
        return json.dumps(
            {"type": "input_audio_buffer.append", "audio": event.audio}
        )
    if isinstance(event, (AudioCommitEvent, UpstreamAudioCommitEvent)):
        return json.dumps({"type": "input_audio_buffer.commit"})
    return None


def to_upstream_payload(
    event: Any, transcription_mode: bool = False
) -> str | None:
    """Map a validated client event to its upstream JSON (audio untouched).

    In translation mode returns None for commit events (the translation
    session is continuous, no commit lifecycle). In transcription mode
    commits close the turn and use the commit-lifecycle event names.
    Language-config events never reach here; they are applied by
    ``_apply_language_config`` before this mapping runs.
    """
    if transcription_mode:
        return _transcription_payload(event)
    if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
        return json.dumps(
            {"type": "session.input_audio_buffer.append", "audio": event.audio}
        )
    return None


class RelaySession:
    def __init__(
        self,
        client: ClientSocket,
        upstream: UpstreamSocket,
        settings: Settings,
        subtitle: Any | None = None,
    ) -> None:
        self._client = client
        self._upstream = upstream
        self._settings = settings
        self._workflow = STTWorkflowFactory.get_workflow(settings)
        self._subtitle = subtitle or SubtitleFactory.get_translator(settings)
        self._subtitle_target_language = settings.translation_target_language
        self._subtitle_source = settings.translation_source_language or None
        self._limiter = AudioThroughputLimiter(settings.max_audio_bytes_per_second)
        # Session mode: the deployment-wide transcribe switch, or a
        # same-language session (the translation model omits same-language
        # audio). Drives append/commit event names, session updates, and
        # upstream transcript mapping below.
        self._transcription_mode = settings.transcribe_mode or is_same_language(
            settings.translation_source_language,
            settings.translation_target_language,
        )
        self._transcription_languages = settings.transcription_language_hint()
        # Target language already applied upstream by the initial
        # session.update sent at connect time. Repeat language-config
        # events for the same language are skipped, never re-sent.
        self._applied_target_language = settings.translation_target_language
        # Growing-window subtitle state: ``_subtitle_rev`` bumps on every
        # (re)translation so stale tasks drop their pieces; only the
        # latest rev may emit. ``_subtitle_task`` is the in-flight
        # provisional translation, cancelled whenever a newer word
        # arrives. ``_subtitle_fatal`` latches a misconfiguration
        # (e.g. missing API key) so later words pass through quietly.
        self._subtitle_rev = 0
        self._subtitle_task: asyncio.Task | None = None
        self._subtitle_fatal = False
        # Audio flow counters (metadata only, never payloads).
        self._audio_chunks = 0
        self._audio_bytes = 0
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
            "audio flow chunks=%d bytes=%d "
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
            "audio session ended reason=%s chunks=%d bytes=%d "
            "upstream_events=%d elapsed=%.1fs overall=%.0fB/s",
            reason,
            self._audio_chunks,
            self._audio_bytes,
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
            self._cancel_subtitle_task()
            for task in (c2u, u2c):
                if not task.done():
                    task.cancel()
            await asyncio.gather(c2u, u2c, return_exceptions=True)
            await self._join_subtitle_task()
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

    async def _apply_language_config(
        self, event: LanguageConfigEvent
    ) -> str | None:
        """Forward a target-language change upstream. Returns stop reason."""
        self._update_subtitle_target(event.target_lang)
        self._subtitle_source = event.source_lang or self._subtitle_source
        if self._workflow.name == "soniox":
            return self._apply_soniox_language_config(event)
        if self._settings.transcribe_mode or is_same_language(
            event.source_lang, event.target_lang
        ):
            return await self._apply_transcription_config(event.source_lang)
        if (
            event.target_lang == self._applied_target_language
            and not self._transcription_mode
        ):
            log.debug(
                "language unchanged target=%s", event.target_lang
            )
            return None
        payload = build_session_update(
            event.target_lang,
            self._settings.translation_transcription_model,
        )
        try:
            await asyncio.wait_for(
                self._upstream.send_text(payload),
                timeout=UPSTREAM_SEND_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            log.warning(
                "upstream language update failed type=%s",
                type(exc).__name__,
            )
            await self._send_client_error(
                "upstream_error", "translation service unavailable"
            )
            return "upstream_send_failed"
        self._transcription_mode = False
        self._applied_target_language = event.target_lang
        log.info(
            "translation target language updated target=%s",
            event.target_lang,
        )
        return None

    def _update_subtitle_target(self, target_language: str) -> None:
        if target_language != self._subtitle_target_language:
            self._subtitle_target_language = target_language
            self._cancel_subtitle_task()
            try:
                self._subtitle.clear_history()
            except Exception:
                pass

    def _cancel_subtitle_task(self) -> None:
        task, self._subtitle_task = self._subtitle_task, None
        if task is not None and not task.done():
            task.cancel()

    async def _join_subtitle_task(self) -> None:
        task, self._subtitle_task = self._subtitle_task, None
        if task is not None:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _subtitle_source_language(self) -> str | None:
        return self._subtitle_source

    def _subtitle_emit_delta(
        self, rev: int, seg: str, item_id: str, pieces: list[str]
    ) -> Callable[[str], Awaitable[None]]:
        """Build the streaming sink for one (re)translation revision.

        Only the latest ``rev`` may emit; stale revisions drop their
        pieces so the frontend never interleaves two snapshots. Each
        translated segment starts on a new line so consecutive captions
        don't run together on the client.
        """

        async def _emit(piece: str) -> None:
            if not piece:
                return
            if rev != self._subtitle_rev:
                return
            if not pieces and not piece.startswith("\n"):
                piece = "\n" + piece
            pieces.append(piece)
            try:
                await self._client.send_text(
                    TranscriptDeltaEvent(
                        delta=piece, item_id=item_id, seg_id=seg, rev=rev
                    ).model_dump_json()
                )
            except Exception:
                pass

        return _emit

    def _spawn_subtitle_provisional(
        self,
        *,
        snapshot: str,
        seg: str,
        rev: int,
        item_id: str,
        source: str | None,
        target: str,
    ) -> None:
        """Re-translate a growing snapshot without blocking the loop.

        The previous provisional translation is cancelled: its revision
        is stale and its pieces (if any already sent) are superseded by
        this revision, which the frontend uses as the replacement text.
        Failures are silent; the next word retriggers the translation.
        """
        self._cancel_subtitle_task()

        async def _run() -> None:
            pieces: list[str] = []
            try:
                await self._subtitle.translate_snapshot(
                    snapshot,
                    target_language=target,
                    source_language=source,
                    on_chunk=self._subtitle_emit_delta(rev, seg, item_id, pieces),
                )
            except asyncio.CancelledError:
                pass
            except SubtitleError as exc:
                if exc.code == "not_configured":
                    self._subtitle_fatal = True
                log.debug("subtitle provisional skipped code=%s", exc.code)
            except Exception as exc:
                log.debug(
                    "subtitle provisional failed type=%s", type(exc).__name__
                )

        self._subtitle_task = asyncio.create_task(_run())

    async def _translate_subtitle_final(
        self,
        snapshot: str,
        *,
        seg: str,
        rev: int,
        item_id: str,
        source: str | None,
        target: str,
    ) -> str | None:
        """Translate a finalized window, stream it, and send its completion.

        Emits streaming deltas for ``(seg, rev)`` followed by a
        ``transcript.completed`` carrying the same ``seg_id`` as the
        segment-final identifier. Returns the translated text, or None
        when translation failed and the caller must fall back.
        """
        pieces: list[str] = []
        try:
            translated = await self._subtitle.translate_snapshot(
                snapshot,
                target_language=target,
                source_language=source,
                on_chunk=self._subtitle_emit_delta(rev, seg, item_id, pieces),
            )
        except SubtitleError as exc:
            if exc.code == "not_configured":
                self._subtitle_fatal = True
            log.warning("subtitle final skipped code=%s", exc.code)
            return None
        except Exception as exc:
            log.warning(
                "subtitle final failed type=%s", type(exc).__name__
            )
            return None
        if not translated or not translated.strip():
            return None
        if not translated.startswith("\n"):
            translated = "\n" + translated
        try:
            await self._client.send_text(
                TranscriptCompletedEvent(
                    transcript=translated, item_id=item_id, seg_id=seg
                ).model_dump_json()
            )
        except Exception:
            return None
        log.debug(
            "subtitle segment done seg=%s chars=%d->%d streamed=%d",
            seg,
            len(snapshot),
            len(translated),
            len(pieces),
        )
        return translated

    async def _maybe_translate_subtitle(self, mapped: Any) -> Any:
        if not self._transcription_mode:
            return mapped
        if self._subtitle_fatal:
            return mapped
        if not getattr(self._subtitle, "enabled", False):
            return mapped
        if isinstance(mapped, TranscriptDeltaEvent):
            text, is_delta = mapped.delta, True
        elif isinstance(mapped, TranscriptCompletedEvent):
            text, is_delta = mapped.transcript, False
        else:
            return mapped
        if not text or not text.strip():
            return mapped
        source = self._subtitle_source_language()
        target = self._subtitle_target_language
        if source and is_same_language(source, target):
            return mapped
        translator = self._subtitle
        if is_delta:
            # Growing window: accumulate the word and re-translate the
            # whole snapshot immediately; the frontend replaces the
            # seg line with each new revision.
            snapshot = translator.push_fragment(text)
            if not snapshot.strip():
                return None
            if translator.should_finalize(snapshot):
                translator.take_window()
                head, tail = translator.split_for_finalize(snapshot)
                seg = translator.seg_id
                self._cancel_subtitle_task()
                self._subtitle_rev += 1
                translated = await self._translate_subtitle_final(
                    head,
                    seg=seg,
                    rev=self._subtitle_rev,
                    item_id=mapped.item_id,
                    source=source,
                    target=target,
                )
                if translated is None:
                    translator.set_window(snapshot)
                    return mapped
                translator.commit(head, translated.lstrip("\n"))
                if tail.strip():
                    translator.set_window(tail)
                    self._subtitle_rev += 1
                    self._spawn_subtitle_provisional(
                        snapshot=translator.current_snapshot(),
                        seg=translator.seg_id,
                        rev=self._subtitle_rev,
                        item_id=mapped.item_id,
                        source=source,
                        target=target,
                    )
                return None
            self._subtitle_rev += 1
            self._spawn_subtitle_provisional(
                snapshot=snapshot,
                seg=translator.seg_id,
                rev=self._subtitle_rev,
                item_id=mapped.item_id,
                source=source,
                target=target,
            )
            return None
        # Turn end: the window is the segment, extended by the
        # completion itself when it carries trailing words.
        snapshot = translator.take_final(text)
        if not snapshot.strip():
            return mapped
        seg = translator.seg_id
        self._cancel_subtitle_task()
        self._subtitle_rev += 1
        translated = await self._translate_subtitle_final(
            snapshot,
            seg=seg,
            rev=self._subtitle_rev,
            item_id=mapped.item_id,
            source=source,
            target=target,
        )
        if translated is None:
            return mapped
        translator.commit(snapshot, translated.lstrip("\n"))
        return None
    async def _apply_transcription_config(self, source_lang: str) -> str | None:
        """Forward a transcription language hint upstream. Returns stop reason."""
        if self._workflow.name == "soniox":
            return self._apply_soniox_transcription_config(source_lang)
        if self._transcription_mode and [source_lang] == self._transcription_languages:
            log.debug("transcription language unchanged hint=%s", source_lang)
            return None
        payload = build_transcription_session_update(
            self._settings.openai_transcription_model, [source_lang]
        )
        try:
            await asyncio.wait_for(
                self._upstream.send_text(payload),
                timeout=UPSTREAM_SEND_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            log.warning(
                "upstream transcription update failed type=%s",
                type(exc).__name__,
            )
            await self._send_client_error(
                "upstream_error", "translation service unavailable"
            )
            return "upstream_send_failed"
        self._transcription_mode = True
        self._transcription_languages = [source_lang]
        log.info("transcription language hint updated hint=%s", source_lang)
        return None

    def _apply_soniox_language_config(self, event: LanguageConfigEvent) -> str | None:
        """Track target switches for Soniox (target is fixed at connect).

        A Soniox translation session already carries the original tokens
        alongside the translated ones, so switching modes only flips the
        local emission filter: no upstream send is needed.
        """
        if self._settings.transcribe_mode or is_same_language(
            event.source_lang, event.target_lang
        ):
            self._transcription_mode = True
            self._transcription_languages = [event.source_lang]
            log.info(
                "transcription language hint updated hint=%s", event.source_lang
            )
        else:
            self._transcription_mode = False
            self._applied_target_language = event.target_lang
            log.info(
                "translation target language updated target=%s",
                event.target_lang,
            )
        return None

    def _apply_soniox_transcription_config(self, source_lang: str) -> str | None:
        """Track transcription hints for Soniox (fixed at connect)."""
        self._transcription_mode = True
        self._transcription_languages = [source_lang]
        log.info("transcription language hint updated hint=%s", source_lang)
        return None

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
            except Exception as exc:
                if str(exc) == "heartbeat":
                    log.debug("client heartbeat received")
                    continue
                raise

            if isinstance(event, LanguageConfigEvent):
                stop = await self._apply_language_config(event)
                if stop is not None:
                    return stop
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

            payload = self._workflow.to_upstream_payload(event, self._transcription_mode)
            if payload is None:
                # Translation sessions are continuous: client commits are
                # accepted for extension compatibility but never forwarded.
                log.debug("commit swallowed type=%s", event.type)
                continue
            try:
                await asyncio.wait_for(
                    self._upstream.send_text(payload),
                    timeout=UPSTREAM_SEND_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                log.warning(
                    "upstream send failed type=%s", type(exc).__name__
                )
                await self._send_client_error(
                    "upstream_error", "translation service unavailable"
                )
                return "upstream_send_failed"
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
                mapped = self._workflow.map_upstream_event(event, self._transcription_mode)
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
            mapped = await self._maybe_translate_subtitle(mapped)
            if mapped is None:
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
