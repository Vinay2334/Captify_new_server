"""Upstream Soniox Realtime WebSocket client (stt-rt-v5).

One instance per connected extension client. Opens
``wss://stt-rt.soniox.com/transcribe-websocket`` (configurable),
authenticates with the server-side API key via the ``Authorization``
header, and configures a session with a JSON first frame such as::

    {
        "model": "stt-rt-v5",
        "audio_format": "pcm_s16le",
        "sample_rate": 24000,
        "num_channels": 1,
        "language_hints": ["en"],
        "translation": {"type": "one_way", "target_language": "es"},
    }

Wire contract (Soniox Speech-to-Text WebSocket API docs, 2026):

- C->S first frame: JSON session config (``model``, ``audio_format``,
  ``language_hints``, optional ``translation`` block, endpoint options).
- C->S audio: binary WebSocket frames, or text frames holding standard
  base64-encoded bytes. This client sends audio as base64 text frames so
  the relay keeps its text-only upstream socket interface.
- S->C: JSON responses ``{"tokens": [...], "finished": false}`` where
  each token carries ``text``, ``is_final``, ``translation_status``
  (``"original"`` / ``"translation"`` / ``"none"``), ``language`` and,
  for spoken tokens, timestamps. Final tokens are returned exactly
  once; non-final tokens are provisional snapshots.
- C->S end of stream: an empty text frame. The server then sends a
  ``{"tokens": [], "finished": true}`` response and closes.

No third-party Soniox SDK is used: the ``websockets`` package already
in ``requirements.txt`` is the only dependency.
"""

import asyncio
import json
from typing import Any

from app.core.logging import get_logger
from app.schemas.events import (
    TranscriptDeltaEvent,
    UpstreamError,
)

log = get_logger(__name__)

SONIOX_DEFAULT_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_DEFAULT_MODEL = "stt-rt-v5"


def build_soniox_config(
    target_language: str,
    model: str = SONIOX_DEFAULT_MODEL,
    language_hints: list[str] | None = None,
    sample_rate: int = 24000,
    num_channels: int = 1,
    enable_endpoint_detection: bool = True,
) -> str:
    """Session config for one-way translation into ``target_language``."""
    return json.dumps(
        {
            "model": model,
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "language_hints": list(language_hints) if language_hints else [target_language],
            "enable_language_identification": True,
            "enable_endpoint_detection": enable_endpoint_detection,
            "translation": {
                "type": "one_way",
                "target_language": target_language,
            },
        }
    )


def build_soniox_transcribe_config(
    language_hint: str = "en",
    model: str = SONIOX_DEFAULT_MODEL,
    language_hints: list[str] | None = None,
    sample_rate: int = 24000,
    num_channels: int = 1,
    enable_endpoint_detection: bool = True,
) -> str:
    """Session config for transcription without translation.

    Used for the deployment-wide transcribe switch and same-language
    sessions. The relay then emits the original tokens instead of the
    translated ones; the model name is unchanged.
    """
    return json.dumps(
        {
            "model": model,
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "language_hints": list(language_hints) if language_hints else [language_hint],
            "enable_language_identification": True,
            "enable_endpoint_detection": enable_endpoint_detection,
        }
    )


def to_soniox_audio_frame(audio_b64: str) -> str:
    """Wrap validated base64 PCM16 audio as a Soniox audio text frame.

    Soniox accepts a text frame holding standard base64-encoded bytes,
    so the relay can stay on its text-only upstream socket interface.
    """
    return audio_b64


def _translated_text(tokens: list[dict]) -> str:
    return "".join(
        token.get("text", "")
        for token in tokens
        if token.get("is_final") and token.get("translation_status") == "translation"
    )


def _original_text(tokens: list[dict]) -> str:
    return "".join(
        token.get("text", "")
        for token in tokens
        if token.get("is_final")
        and token.get("translation_status", "none") in ("none", "original")
    )


def map_soniox_event(
    event: dict[str, Any], transcription_mode: bool = False
) -> TranscriptDeltaEvent | None:
    """Map one Soniox token response to an extension transcript delta.

    Only finalized tokens map downstream, so each spoken word is
    emitted exactly once: provisional (non-final) tokens are skipped
    here and surface once finalized. Translation sessions emit the
    translated tokens; transcription sessions emit the original ones.
    The ``finished`` response and empty token lists map to None, which
    matches the translation path (deltas only, no completed turns).
    """
    if event.get("type") == "error":
        raise UpstreamError.from_event(event)
    error = event.get("error")
    if isinstance(error, dict) and error:
        raise UpstreamError.from_event(event)
    if isinstance(error, str) and error:
        raise UpstreamError(
            code="upstream_error", message="translation service unavailable"
        )
    if event.get("finished"):
        return None
    tokens = event.get("tokens", [])
    if not isinstance(tokens, list) or not tokens:
        return None
    text = _original_text(tokens) if transcription_mode else _translated_text(tokens)
    if not text:
        return None
    return TranscriptDeltaEvent(delta=text, item_id="")


class SonioxRealtimeClient:
    """Dedicated Soniox upstream connection for one extension session."""

    def __init__(
        self,
        api_key: str,
        model: str = SONIOX_DEFAULT_MODEL,
        url: str = SONIOX_DEFAULT_URL,
        target_language: str = "en",
        source_language: str = "",
        language_hints: list[str] | None = None,
        sample_rate: int = 24000,
        num_channels: int = 1,
        enable_endpoint_detection: bool = True,
        connect_timeout: float = 15.0,
        is_transcription: bool = False,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._target_language = target_language
        self._source_language = source_language
        if language_hints is not None:
            self._language_hints = list(language_hints)
        elif source_language:
            self._language_hints = [source_language, target_language]
        else:
            self._language_hints = [target_language]
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._enable_endpoint_detection = enable_endpoint_detection
        self._connect_timeout = connect_timeout
        self._is_transcription = is_transcription
        self._ws: Any = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def is_transcription(self) -> bool:
        return self._is_transcription

    def _config_for_mode(self) -> str:
        if self._is_transcription:
            return build_soniox_transcribe_config(
                self._target_language,
                model=self._model,
                language_hints=self._language_hints,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                enable_endpoint_detection=self._enable_endpoint_detection,
            )
        return build_soniox_config(
            self._target_language,
            model=self._model,
            language_hints=self._language_hints,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            enable_endpoint_detection=self._enable_endpoint_detection,
        )

    async def connect(self) -> None:
        """Open the upstream socket and send the session config."""
        import websockets
        from websockets.exceptions import InvalidStatus

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            connect = websockets.connect(
                self._url,
                additional_headers=headers,
                max_size=None,
                open_timeout=self._connect_timeout,
            )
            self._ws = await connect
        except TypeError:
            # Older websockets (<14) uses extra_headers.
            connect = websockets.connect(
                self._url,
                extra_headers=headers,  # type: ignore[call-arg]
                max_size=None,
                open_timeout=self._connect_timeout,
            )
            self._ws = await connect
        except InvalidStatus as exc:
            raise self._status_to_error(exc) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise UpstreamError(
                code="connect_timeout",
                message="could not reach translation service",
            ) from exc
        except OSError as exc:
            raise UpstreamError(
                code="connect_failed",
                message="could not reach translation service",
            ) from exc

        try:
            await self._ws.send(self._config_for_mode())
        except Exception as exc:
            await self.close()
            log.warning(
                "soniox session configure failed type=%s",
                type(exc).__name__,
            )
            raise UpstreamError(
                code="connect_failed",
                message="could not start translation session",
            ) from exc
        log.info(
            "soniox session configured model=%s target_language=%s",
            self._model,
            self._target_language,
        )

    def _status_to_error(self, exc: Exception) -> UpstreamError:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return UpstreamError(
                code="auth_failed",
                message="translation service authentication failed",
            )
        if status == 429:
            return UpstreamError(
                code="rate_limited", message="translation service is busy"
            )
        return UpstreamError(
            code="upstream_error", message="translation service unavailable"
        )

    async def send_text(self, payload: str) -> None:
        if self._ws is None:
            raise UpstreamError(code="not_connected", message="upstream closed")
        await self._ws.send(payload)

    async def recv_text(self) -> str:
        if self._ws is None:
            raise UpstreamError(code="not_connected", message="upstream closed")
        message = await self._ws.recv()
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        return message

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                # An empty text frame ends the Soniox stream gracefully;
                # fall back to plain close when the socket is already gone.
                await ws.send("")
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
        log.info("soniox connection closed")
