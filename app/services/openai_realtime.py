"""Upstream OpenAI Realtime API WebSocket client.

One instance per connected extension client. Opens
``wss://api.openai.com/v1/realtime?intent=transcription`` (configurable),
authenticates with the server-side API key, and configures a transcription
session (``gpt-live-transcribe`` by default).

Wire contract verified against the official OpenAI Realtime transcription
guide (developers.openai.com, October 2025+):
  C->S  {"type": "session.update", "session": {"type": "transcription",
          "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
          "transcription": {"model": ...}, "turn_detection": null}}}}
  C->S  {"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
  C->S  {"type": "input_audio_buffer.commit"}
  S->C  {"type": "conversation.item.input_audio_transcription.delta", ...}
  S->C  {"type": "conversation.item.input_audio_transcription.completed", ...}
"""

import asyncio
import json
from typing import Any

from app.core.logging import get_logger
from app.schemas.events import UpstreamError

log = get_logger(__name__)


def build_session_update(model: str) -> str:
    """Session configuration for transcription (manual commit, 24 kHz PCM).

    ``turn_detection`` is null: ``gpt-live-transcribe`` does not support
    server VAD, so the extension commits each audio turn explicitly.
    """
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": model},
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
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._beta_header = beta_header
        self._connect_timeout = connect_timeout
        self._ws: Any = None

    @property
    def model(self) -> str:
        return self._model

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
                message="could not reach transcription service",
            ) from exc
        except OSError as exc:
            raise UpstreamError(
                code="connect_failed",
                message="could not reach transcription service",
            ) from exc

        try:
            await self._ws.send(build_session_update(self._model))
        except Exception as exc:
            await self.close()
            log.warning(
                "upstream session configure failed type=%s",
                type(exc).__name__,
            )
            raise UpstreamError(
                code="connect_failed",
                message="could not start transcription session",
            ) from exc
        log.info("upstream session configured model=%s", self._model)

    def _status_to_error(self, exc: Exception) -> UpstreamError:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return UpstreamError(
                code="auth_failed",
                message="transcription service authentication failed",
            )
        if status == 429:
            return UpstreamError(
                code="rate_limited", message="transcription service is busy"
            )
        return UpstreamError(
            code="upstream_error", message="transcription service unavailable"
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
