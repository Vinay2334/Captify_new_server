"""STT provider factory: drops in the correct model workflow per settings.

The global ``STT_PROVIDER`` setting (``"openai"`` or ``"soniox"``) picks
which upstream workflow a session uses. Each workflow bundles everything
that differs between providers behind one interface:

- upstream client construction (OpenAI Realtime vs Soniox ``stt-rt-v5``),
- client-audio -> upstream payload mapping (both transports),
- upstream-event -> extension-event mapping,
- runtime language-update payloads (``None`` means the switch needs no
  upstream send; the relay then only flips its local mode).

``RelaySession`` and ``WebRTCRelaySession`` resolve their workflow once
via ``STTWorkflowFactory.get_workflow(settings)`` and delegate all
provider-specific mapping to it, so adding a third provider only means
adding a workflow class plus a branch here. The pre-existing
module-level ``to_upstream_payload`` / ``map_upstream_event`` helpers
stay OpenAI-shaped for backward compatibility.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from app.schemas.events import (
    AudioAppendEvent,
    AudioCommitEvent,
    UpstreamAudioAppendEvent,
    UpstreamAudioCommitEvent,
    UpstreamError,
    map_upstream_event as map_openai_event,
)
from app.services.openai_realtime import (
    OpenAIRealtimeClient,
    build_session_update,
    build_transcription_session_update,
)
from app.services.soniox_realtime import (
    SonioxRealtimeClient,
    map_soniox_event,
)

PROVIDER_OPENAI = "openai"
PROVIDER_SONIOX = "soniox"


class OpenAIWorkflow:
    """Workflow for the OpenAI Realtime translation/transcription models."""

    name = PROVIDER_OPENAI

    def create_upstream_client(self, settings: Any) -> OpenAIRealtimeClient:
        if settings.starts_in_transcription():
            return OpenAIRealtimeClient(
                api_key=settings.openai_api_key,
                model=settings.openai_transcription_model,
                url=settings.openai_transcription_url,
                beta_header=settings.openai_beta_header,
                connect_timeout=settings.upstream_connect_timeout_seconds,
                target_language=settings.translation_target_language,
                transcription_model=settings.translation_transcription_model,
                is_transcription=True,
                transcription_languages=settings.transcription_language_hint(),
            )
        return OpenAIRealtimeClient(
            api_key=settings.openai_api_key,
            model=settings.openai_realtime_model,
            url=settings.openai_realtime_url,
            beta_header=settings.openai_beta_header,
            connect_timeout=settings.upstream_connect_timeout_seconds,
            target_language=settings.translation_target_language,
            transcription_model=settings.translation_transcription_model,
        )

    def to_upstream_payload(
        self, event: Any, transcription_mode: bool = False
    ) -> str | None:
        # Single source of truth lives in relay.py; imported lazily to
        # avoid a module cycle (relay delegates to this workflow).
        from app.services.relay import to_upstream_payload

        return to_upstream_payload(event, transcription_mode)

    def webrtc_audio_payload(
        self, raw: bytes, transcription_mode: bool = False
    ) -> str:
        from app.services.webrtc_session import webrtc_audio_to_upstream_payload

        return webrtc_audio_to_upstream_payload(raw, transcription_mode)

    def map_upstream_event(
        self, event: dict[str, Any], transcription_mode: bool = False
    ):  # -> BaseModel | None
        return map_openai_event(event, transcription_mode)

    def build_language_update(
        self, target_language: str, settings: Any
    ) -> str | None:
        return build_session_update(
            target_language, settings.translation_transcription_model
        )

    def build_transcription_update(
        self, model: str, languages: list[str]
    ) -> str | None:
        return build_transcription_session_update(model, languages)


class SonioxWorkflow:
    """Workflow for the Soniox ``stt-rt-v5`` realtime model.

    Audio travels as base64 text frames (accepted by Soniox as audio
    content); translation targets are fixed at connect time, so runtime
    language switches only flip which tokens the relay emits and need
    no upstream send (``None`` payloads below).
    """

    name = PROVIDER_SONIOX

    def create_upstream_client(self, settings: Any) -> SonioxRealtimeClient:
        return SonioxRealtimeClient(
            api_key=settings.soniox_api_key,
            model=settings.soniox_model,
            url=settings.soniox_url,
            target_language=settings.translation_target_language,
            source_language=settings.translation_source_language,
            language_hints=settings.soniox_language_hints(),
            sample_rate=settings.soniox_sample_rate,
            num_channels=settings.soniox_num_channels,
            enable_endpoint_detection=settings.soniox_enable_endpoint_detection,
            connect_timeout=settings.upstream_connect_timeout_seconds,
            is_transcription=settings.starts_in_transcription(),
        )

    def to_upstream_payload(
        self, event: Any, transcription_mode: bool = False
    ) -> str | None:
        del transcription_mode  # Soniox audio frames are mode-independent.
        if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
            return event.audio
        # Soniox has no commit lifecycle; the stream ends with an empty
        # text frame sent by the client close path instead.
        return None

    def webrtc_audio_payload(
        self, raw: bytes, transcription_mode: bool = False
    ) -> str:
        del transcription_mode
        if not raw or len(raw) % 2 != 0:
            raise ValueError("audio payload must be non-empty Int16 PCM bytes")
        try:
            return base64.b64encode(raw).decode("ascii")
        except (binascii.Error, ValueError) as exc:
            raise ValueError("could not encode audio payload") from exc

    def map_upstream_event(
        self, event: dict[str, Any], transcription_mode: bool = False
    ):  # -> BaseModel | None
        if event.get("type") == "error" and "tokens" not in event:
            # OpenAI-shaped error (defensive; Soniox sends token
            # responses). Never fall through to the token mapper.
            raise UpstreamError.from_event(event)
        return map_soniox_event(event, transcription_mode)

    def build_language_update(
        self, target_language: str, settings: Any
    ) -> str | None:
        del target_language, settings
        # Fixed at connect; the relay flips its local emission filter.
        return None

    def build_transcription_update(
        self, model: str, languages: list[str]
    ) -> str | None:
        del model, languages
        return None


class STTWorkflowFactory:
    """Drops in the workflow for the configured ``STT_PROVIDER``."""

    @staticmethod
    def get_workflow(settings: Any) -> OpenAIWorkflow | SonioxWorkflow:
        if getattr(settings, "stt_provider", PROVIDER_OPENAI) == PROVIDER_SONIOX:
            return SonioxWorkflow()
        return OpenAIWorkflow()

    @staticmethod
    def provider_name(settings: Any) -> str:
        return STTWorkflowFactory.get_workflow(settings).name

    @staticmethod
    def create_upstream_client(settings: Any) -> Any:
        return STTWorkflowFactory.get_workflow(settings).create_upstream_client(
            settings
        )
