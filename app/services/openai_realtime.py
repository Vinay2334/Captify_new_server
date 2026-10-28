"""Upstream OpenAI Realtime API WebSocket client.

One instance per connected extension client. Opens
``wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate``
(configurable), authenticates with the server-side API key, and configures a
translation session (``gpt-realtime-translate`` by default).

In transcription mode (``is_transcription=True``) it opens the
transcription endpoint instead and configures a transcription session
(``gpt-realtime-transcribe`` by default) with a commit / turn lifecycle.

Wire contract verified against the official OpenAI Realtime translation
guide and translation client/server event references
(developers.openai.com, 2026):
  C->S  {"type": "session.update", "session":
          {"audio": {"input": {"transcription": {"model": ...},
          "noise_reduction": {"type": "near_field"}},
          "output": {"language": "<target>"}}}}
  C->S  {"type": "session.input_audio_buffer.append", "audio": "<base64 PCM16>"}
There is no commit / turn lifecycle: translation streams continuously.
  S->C  {"type": "session.output_transcript.delta", "delta": ..., ...}
  S->C  {"type": "session.input_transcript.delta", "delta": ..., ...}
  S->C  {"type": "session.output_audio.delta", "delta": ..., ...}
"""

import asyncio
import json
from typing import Any

from app.core.logging import get_logger
from app.schemas.events import UpstreamError

log = get_logger(__name__)


def build_session_update(
    target_language: str,
    transcription_model: str = "gpt-realtime-whisper",
) -> str:
    """Session configuration for translation (continuous, 24 kHz PCM).

    Translation sessions have no commit / turn lifecycle: audio streams
    continuously with the ``session.``-prefixed append event, the source
    language is auto-detected, and only the output ``language`` is set.
    The transcription model only controls source-language transcript
    deltas; translation itself always runs from the input audio stream.
    """
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "transcription": {"model": transcription_model},
                        "noise_reduction": {"type": "near_field"},
                    },
                    "output": {"language": target_language},
                }
            },
        }
    )


def build_transcription_session_update(
    model: str = "gpt-realtime-transcribe",
    languages: list[str] | None = None,
) -> str:
    """Session configuration for transcription (continuous, 24 kHz PCM).

    Transcription sessions have a commit / turn lifecycle: audio streams
    with the ``input_audio_buffer.append`` event and each turn is closed
    with ``input_audio_buffer.commit``. The transcript events map to the
    same downstream ``transcript.delta`` / ``transcript.completed``
    stream the extension renders for translations.
    """
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": model,
                            "languages": list(languages) if languages else [],
                        },
                        "turn_detection": None,
                    }
                },
            },
        }
    )


class OpenAIRealtimeClient:
    """Dedicated upstream connection for one extension session."""

    def __init__(
        self,
        api_key: str,
        model: str,
        url: str,
        beta_header: str = "realtime=v1",
        connect_timeout: float = 15.0,
        target_language: str = "en",
        transcription_model: str = "gpt-realtime-whisper",
        is_transcription: bool = False,
        transcription_languages: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._beta_header = beta_header
        self._connect_timeout = connect_timeout
        self._target_language = target_language
        self._transcription_model = transcription_model
        self._is_transcription = is_transcription
        self._transcription_languages = list(transcription_languages or [])
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

    def _session_update_for_mode(
        self, target_language: str, transcription_model: str
    ) -> str:
        """Initial session.update for the active (translate/transcribe) mode."""
        if self._is_transcription:
            return build_transcription_session_update(
                self._model,
                self._transcription_languages or [target_language],
            )
        return build_session_update(target_language, transcription_model)

    async def connect(self) -> None:
        """Open the upstream socket and configure the session."""
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
            await self._ws.send(
                self._session_update_for_mode(
                    self._target_language, self._transcription_model
                )
            )
        except Exception as exc:
            await self.close()
            log.warning(
                "upstream session configure failed type=%s",
                type(exc).__name__,
            )
            raise UpstreamError(
                code="connect_failed",
                message="could not start translation session",
            ) from exc
        log.info(
            "upstream session configured model=%s target_language=%s",
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
                await ws.close()
            except Exception:
                pass
        log.info("upstream connection closed")
