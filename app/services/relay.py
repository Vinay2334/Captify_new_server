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
from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.events import (
    AudioAppendEvent,
    AudioCommitEvent,
    ErrorEvent,
    LanguageConfigEvent,
    SessionReadyEvent,
    TranscriptCompletedEvent,
    TranscriptDeltaEvent,
    UpstreamAudioAppendEvent,
    UpstreamAudioCommitEvent,
    UpstreamError,
    map_upstream_event,
    is_same_language,
    normalize_model_tier,
    parse_client_event,
)
from app.services.stt_factory import STTWorkflowFactory
from app.services.subtitle_factory import SubtitleError, SubtitleFactory
from app.services.openai_realtime import (
    build_session_update,
    build_transcription_session_update,
)
from app.services.quota import SessionQuotaTracker

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
        quota: SessionQuotaTracker | None = None,
        model_tier: str | None = None,
        upstream_factory: Any | None = None,
    ) -> None:
        self._client = client
        self._upstream = upstream
        self._settings = settings
        self._quota = quota
        # Factory for on-the-fly engine swaps (tier, ) -> upstream client.
        # Direct-constructed sessions (tests) may omit it; a tier flip
        # then ends the session instead of swapping.
        self._upstream_factory = upstream_factory
        self._upstream_gen = 0
        # Transcription tiers ("base" Soniox / "pro" OpenAI live-transcribe)
        # pick the speech-to-text engine only. When a session carries an
        # explicit tier, every session transcribed from connect time, so
        # the workflow and transcription flags start tiered instead of
        # the legacy translate-unless-flagged behavior below.
        self._model_tier = normalize_model_tier(model_tier)
        self._workflow = STTWorkflowFactory.get_workflow(settings, self._model_tier)
        self._is_tiered = self._workflow.tier is not None
        self._subtitle = subtitle or SubtitleFactory.get_translator(settings)
        self._subtitle_target_language = settings.translation_target_language
        self._subtitle_source = settings.translation_source_language or None
        self._limiter = AudioThroughputLimiter(settings.max_audio_bytes_per_second)
        # Session mode: tiered sessions always transcribe (source-language
        # captions; translation happens later as text-to-text). Legacy
        # sessions keep the old behavior: the deployment-wide transcribe
        # switch, or a same-language session (the translation model omits
        # same-language audio). Drives append/commit event names, session
        # updates, and upstream transcript mapping below.
        if self._is_tiered:
            self._transcription_mode = True
            self._transcription_languages: list[str] = []
        else:
            self._transcription_mode = settings.transcribe_mode or is_same_language(
                settings.translation_source_language,
                settings.translation_target_language,
            )
            self._transcription_languages = settings.transcription_language_hint()
        # Mode explicitly requested by the extension for this session
        # ("translation" / "transcription"); None until it asks. An
        # explicit transcription request disables subtitle translation
        # so captions stay in the auto-detected source language.
        self._session_mode: str | None = None
        # Target language already applied upstream by the initial
        # session.update sent at connect time. Repeat language-config
        # events for the same language are skipped, never re-sent.
        self._applied_target_language = settings.translation_target_language
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
        tasks = {c2u, u2c}
        if self._quota is not None:
            # Push the opening quota state right after session.ready so
            # the extension UI shows live numbers from the first second.
            try:
                await self._client.send_text(
                    json.dumps(self._quota.quota_event())
                )
            except Exception:
                pass
            watch = asyncio.create_task(
                self._quota_watch(), name="relay-quota-watch"
            )
            commit = asyncio.create_task(
                self._quota_checkpoint_loop(), name="relay-quota-commit"
            )
            tasks = {c2u, u2c, watch, commit}
        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
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
            if self._quota is not None:
                await self._finish_quota(reason)
            return reason
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._close_both()
            await self._close_subtitle()

    async def _quota_checkpoint_loop(self) -> str:
        """Background worker: commit quota to the DB every 15s.

        Each tick commits this session's billed-so-far seconds to the DB
        (QUOTA_COMMIT_SECONDS, default 15), then fetches the fresh quota
        — picking up other tabs' commits and plan changes — and cuts the
        session off if it has overshot. Runs on its own cadence,
        independent of the refresh/watch tick below.
        """
        assert self._quota is not None
        interval = max(
            getattr(self._settings, "quota_commit_seconds", 15.0), 1.0
        )
        while True:
            await asyncio.sleep(interval)
            await self._quota.checkpoint()
            await self._quota.maybe_refresh(force=True)
            try:
                await self._client.send_text(
                    json.dumps(self._quota.quota_event())
                )
            except Exception:
                return "client_disconnected"
            if self._quota.effective_remaining_seconds <= 0:
                await self._send_client_error(
                    "quota_exceeded",
                    "Subtitle quota exhausted. Upgrade your plan to continue.",
                )
                try:
                    await self._client.close(code=1008, reason="quota exceeded")
                except Exception:
                    pass
                return "quota_exceeded"

    async def _quota_watch(self) -> str:
        """Cut the session off once effective remaining quota hits zero."""
        assert self._quota is not None
        interval = max(self._settings.quota_refresh_seconds, 1.0)
        while True:
            await asyncio.sleep(interval)
            await self._quota.maybe_refresh(force=True)
            try:
                await self._client.send_text(
                    json.dumps(self._quota.quota_event())
                )
            except Exception:
                return "client_disconnected"
            if self._quota.effective_remaining_seconds <= 0:
                await self._send_client_error(
                    "quota_exceeded",
                    "Subtitle quota exhausted. Upgrade your plan to continue.",
                )
                try:
                    await self._client.close(code=1008, reason="quota exceeded")
                except Exception:
                    pass
                return "quota_exceeded"

    async def _finish_quota(self, reason: str) -> None:
        """Commit billed seconds exactly once, then report the result."""
        assert self._quota is not None
        try:
            final = await self._quota.finish(reason)
        except Exception as exc:
            log.warning("quota finish failed type=%s", type(exc).__name__)
            return
        if final is None:
            return
        try:
            await self._client.send_text(
                json.dumps({"type": "quota.update", **final.to_dict()})
            )
        except Exception:
            pass

    # Subtitle translator lifecycle.
    async def _close_subtitle(self) -> None:
        aclose = getattr(self._subtitle, 'aclose', None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

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
        """Apply a mode/language change upstream. Returns stop reason."""
        if event.model_tier is not None:
            # The tier names the transcription engine per request: when it
            # changes, swap the upstream engine on the fly. The transport
            # stays up; audio queued behind this message flows to the new
            # engine in order.
            requested_tier = normalize_model_tier(event.model_tier)
            if requested_tier is not None and requested_tier != self._model_tier:
                stop = await self._swap_transcription_tier(requested_tier)
                if stop is not None:
                    return stop
        mode = event.effective_mode
        if event.mode is None and self._settings.transcribe_mode:
            # Legacy clients without a mode keep the deployment switch.
            mode = "transcription"
        self._session_mode = mode
        if mode == "transcription":
            # Source is auto-detected upstream: no hint and no target.
            self._subtitle_source = None
            if self._workflow.name == "soniox":
                return self._apply_soniox_transcription_config(None)
            return await self._apply_transcription_config(None)
        target = (
            event.target_lang or self._settings.translation_target_language
        )
        self._update_subtitle_target(target)
        # Translation also auto-detects the source; only target is set.
        # Legacy picks keep their source for subtitle context only.
        self._subtitle_source = (
            None
            if event.mode is not None
            else (event.source_lang or self._subtitle_source)
        )
        if self._is_tiered or (
            self._settings.transcribe_mode
            and getattr(self._subtitle, "enabled", False)
        ):
            # This deployment transcribes upstream and translates the
            # subtitle text instead: keep transcribing (auto-detected
            # source) and let _maybe_translate_subtitle render target.
            return await self._apply_transcription_config(None)
        if self._workflow.name == "soniox":
            return self._apply_soniox_language_config(event)
        if (
            target == self._applied_target_language
            and not self._transcription_mode
        ):
            log.debug("language unchanged target=%s", target)
            return None
        payload = build_session_update(
            target,
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
        self._applied_target_language = target
        log.info(
            "translation target language updated target=%s",
            target,
        )
        return None

    async def _swap_transcription_tier(self, tier: str) -> str | None:
        """Swap the upstream transcription engine without dropping transport.

        Connects the new tier's engine first, then flips the relay over:
        audio already queued behind the triggering message flows to the
        new engine in order, and the reader task picks it up via the
        generation guard in ``_upstream_to_client``. Returns a stop
        reason when the swap cannot complete (the session then ends and
        the client's reconnect machinery heals it with a fresh offer).
        """
        if not self._settings.upstream_api_key_for_tier(tier):
            log.warning("tier switch rejected: tier API key is not set")
            await self._send_client_error(
                "not_configured", "translation service is not configured"
            )
            return "not_configured"
        factory = self._upstream_factory
        if factory is None:
            await self._send_client_error(
                "model_tier_mismatch",
                "Model tier changed mid-session. Reconnect to switch tiers.",
            )
            return "model_tier_mismatch"
        try:
            new_upstream = factory(tier)
        except TypeError:
            # Legacy zero-arg test fakes cannot build a tiered upstream.
            await self._send_client_error(
                "model_tier_mismatch",
                "Model tier changed mid-session. Reconnect to switch tiers.",
            )
            return "model_tier_mismatch"
        connect = getattr(new_upstream, "connect", None)
        if connect is not None:
            try:
                await connect()
            except UpstreamError as exc:
                log.warning(
                    "tier switch upstream connect failed code=%s", exc.code
                )
                await self._send_client_error(exc.code, exc.message)
                return f"upstream_error:{exc.code}"
            except Exception as exc:
                log.warning(
                    "tier switch upstream connect failed type=%s",
                    type(exc).__name__,
                )
                await self._send_client_error(
                    "upstream_error", "translation service unavailable"
                )
                return "upstream_send_failed"
        old_upstream = self._upstream
        self._upstream = new_upstream
        self._upstream_gen += 1
        self._workflow = STTWorkflowFactory.get_workflow(self._settings, tier)
        self._model_tier = normalize_model_tier(tier)
        self._is_tiered = True
        self._transcription_mode = True
        self._transcription_languages = []
        try:
            await old_upstream.close()
        except Exception:
            pass
        try:
            await self._client.send_text(
                SessionReadyEvent(
                    model=self._settings.active_model_for_tier(tier)
                ).model_dump_json()
            )
        except Exception:
            return "client_disconnected"
        log.info(
            "transcription tier switched tier=%s model=%s",
            tier,
            self._settings.active_model_for_tier(tier),
        )
        return None

    def _update_subtitle_target(self, target_language: str) -> None:
        if target_language != self._subtitle_target_language:
            self._subtitle_target_language = target_language
            try:
                self._subtitle.clear_history()
            except Exception:
                pass

    def _subtitle_source_language(self) -> str | None:
        return self._subtitle_source

    async def _maybe_translate_subtitle(self, mapped: Any) -> Any:
        if not self._transcription_mode:
            return mapped
        if self._session_mode == "transcription":
            # Explicit transcription: captions are verbatim in the
            # auto-detected source language, never text-translated.
            return mapped
        if not getattr(self._subtitle, "enabled", False):
            return mapped
        if isinstance(mapped, TranscriptDeltaEvent):
            text, store, field = mapped.delta, False, "delta"
        elif isinstance(mapped, TranscriptCompletedEvent):
            text, store, field = mapped.transcript, True, "transcript"
        else:
            return mapped
        if not text or not text.strip():
            return mapped
        source = self._subtitle_source_language()
        target = self._subtitle_target_language
        if source and is_same_language(source, target):
            return mapped
        streamed: list[str] = []

        async def _emit_piece(piece: str) -> None:
            if not piece:
                return
            # Each translated sentence starts on a new line so consecutive
            # captions don't run together on the client.
            if not streamed and not piece.startswith("\n"):
                piece = "\n" + piece
            streamed.append(piece)
            try:
                await self._client.send_text(
                    TranscriptDeltaEvent(
                        delta=piece, item_id=mapped.item_id
                    ).model_dump_json()
                )
            except Exception:
                pass

        try:
            translated = await self._subtitle.translate(
                text,
                target_language=target,
                source_language=source,
                store=store,
                item_id=mapped.item_id,
                on_chunk=_emit_piece,
            )
        except SubtitleError as exc:
            log.warning("subtitle translation skipped code=%s", exc.code)
            return mapped
        except Exception as exc:
            log.warning(
                "subtitle translation failed type=%s", type(exc).__name__
            )
            return mapped
        if translated is None:
            return None
        if not translated.strip():
            return mapped
        log.debug(
            "subtitle translated chars=%d->%d store=%s streamed=%d",
            len(text),
            len(translated),
            store,
            len(streamed),
        )
        if not translated.startswith("\n"):
            translated = "\n" + translated
        if field == "delta":
            if streamed:
                return None
            return TranscriptDeltaEvent(delta=translated, item_id=mapped.item_id)
        return TranscriptCompletedEvent(
            transcript=translated, item_id=mapped.item_id
        )
    async def _apply_transcription_config(
        self, source_lang: str | None
    ) -> str | None:
        """Forward a transcription config upstream. Returns stop reason.

        ``source_lang=None`` sends an empty language list so the model
        auto-detects the spoken language instead of being hinted.
        """
        languages = [source_lang] if source_lang else []
        if self._workflow.name == "soniox":
            return self._apply_soniox_transcription_config(source_lang)
        if self._transcription_mode and languages == self._transcription_languages:
            log.debug("transcription language unchanged hint=%s", source_lang)
            return None
        payload = build_transcription_session_update(
            self._settings.openai_transcription_model, languages
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
        self._transcription_languages = languages
        log.info(
            "transcription language hint updated hint=%s",
            source_lang or "auto",
        )
        return None

    def _apply_soniox_language_config(self, event: LanguageConfigEvent) -> str | None:
        """Legacy Soniox target tracking (untiered sessions only).

        Tiered and current Soniox sessions are transcription-only, so this
        path is unreachable in practice (kept for backward compatibility):
        switching modes only flips the local emission filter, no upstream
        send is needed.
        """
        if event.effective_mode == "transcription":
            self._transcription_mode = True
            self._transcription_languages = (
                [event.source_lang] if event.source_lang else []
            )
            log.info(
                "transcription language hint updated hint=%s",
                event.source_lang or "auto",
            )
        else:
            self._transcription_mode = False
            self._applied_target_language = (
                event.target_lang
                or self._settings.translation_target_language
            )
            log.info(
                "translation target language updated target=%s",
                self._applied_target_language,
            )
        return None

    def _apply_soniox_transcription_config(
        self, source_lang: str | None
    ) -> str | None:
        """Track transcription hints for Soniox (fixed at connect)."""
        self._transcription_mode = True
        self._transcription_languages = [source_lang] if source_lang else []
        log.info(
            "transcription language hint updated hint=%s",
            source_lang or "auto",
        )
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
                # Bill only audio the relay kept (validated, under the
                # rate limit). Dropped chunks never reach this line.
                if self._quota is not None:
                    self._quota.add_audio_bytes(size)
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
            upstream = self._upstream
            gen = self._upstream_gen
            try:
                raw = await upstream.recv_text()
            except Exception:
                if gen != self._upstream_gen:
                    # The engine was hot-swapped underneath this receive
                    # (the old socket's close surfaced here): keep reading
                    # from the new engine instead of ending the session.
                    continue
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
            # A transcript reached the extension: this session generated
            # subtitles, so its audio time counts as usage at commit.
            if self._quota is not None and isinstance(
                mapped, (TranscriptDeltaEvent, TranscriptCompletedEvent)
            ):
                self._quota.note_transcript()
