"""STT provider factory: drops in the correct transcription workflow.

Transcription (speech-to-text) and translation (text-to-text) are
decoupled:

- This factory picks only the transcription engine. ``base`` uses
  Soniox ``stt-rt-v5`` transcription; ``pro`` uses OpenAI
  ``gpt-live-transcribe`` (``OPENAI_TRANSCRIPTION_MODEL``). Both emit
  verbatim source-language ``transcript.delta`` / ``transcript.completed``
  events, so the downstream extension protocol is identical either way.
- Translation is a separate text-to-text step over those transcripts
  (``app.services.subtitle_factory``), applied by the relay sessions
  only when the session mode is ``translation``. Changing the tier never
  changes the translation step, and changing the translation provider
  never changes transcription.

Per-session tier selection: sessions that carry an explicit
``model_tier`` (extension dropdown, WebRTC offer, or ``?model_tier=``)
resolve their workflow here with that tier. Sessions without one keep
the legacy global ``STT_PROVIDER`` behavior so older clients and tests
are unchanged.
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
    normalize_model_tier,
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

MODEL_TIER_BASE = "base"
MODEL_TIER_PRO = "pro"


class OpenAIWorkflow:
    """Legacy OpenAI workflow (global STT_PROVIDER=openai, no tier).

    Preserves the pre-tier behavior: translation sessions open on
    ``gpt-realtime-translate`` unless the deployment transcribe switch
    or a same-language config starts the session in transcription.
    Tiered (``pro``) sessions use :class:`OpenAITranscriptionWorkflow`
    instead, which is transcription-only.
    """

    name = PROVIDER_OPENAI
    tier = None

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
        from app.services.relay import to_upstream_payload

        return to_upstream_payload(event, transcription_mode)

    def webrtc_audio_payload(
        self, raw: bytes, transcription_mode: bool = False
    ) -> str:
        from app.services.webrtc_session import webrtc_audio_to_upstream_payload

        return webrtc_audio_to_upstream_payload(raw, transcription_mode)

    def map_upstream_event(
        self, event: dict[str, Any], transcription_mode: bool = False
    ):
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


class OpenAITranscriptionWorkflow(OpenAIWorkflow):
    """Pro tier: OpenAI ``gpt-live-transcribe`` transcription only.

    Always opens the transcription endpoint; translation (if any) is
    the relay's text-to-text subtitle step over the transcripts.
    """

    tier = MODEL_TIER_PRO

    def create_upstream_client(self, settings: Any) -> OpenAIRealtimeClient:
        return OpenAIRealtimeClient(
            api_key=settings.openai_api_key,
            model=settings.openai_transcription_model,
            url=settings.openai_transcription_url,
            beta_header=settings.openai_beta_header,
            connect_timeout=settings.upstream_connect_timeout_seconds,
            target_language=settings.translation_target_language,
            transcription_model=settings.translation_transcription_model,
            is_transcription=True,
            transcription_languages=[],
        )


class SonioxWorkflow:
    """Soniox ``stt-rt-v5`` workflow: transcription only.

    The former Soniox one-way translation workflow is removed: the
    session config never carries a ``translation`` block and only
    original (source-language) final tokens map downstream. Audio
    travels as base64 text frames; language switches need no upstream
    send (``None`` payloads below) because transcription has no target.
    """

    name = PROVIDER_SONIOX
    tier = MODEL_TIER_BASE

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
            is_transcription=True,
        )

    def to_upstream_payload(
        self, event: Any, transcription_mode: bool = False
    ) -> str | None:
        del transcription_mode  # Soniox audio frames are mode-independent.
        if isinstance(event, (AudioAppendEvent, UpstreamAudioAppendEvent)):
            return event.audio
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
    ):
        if event.get("type") == "error" and "tokens" not in event:
            raise UpstreamError.from_event(event)
        return map_soniox_event(event, transcription_mode)

    def build_language_update(
        self, target_language: str, settings: Any
    ) -> str | None:
        del target_language, settings
        return None

    def build_transcription_update(
        self, model: str, languages: list[str]
    ) -> str | None:
        del model, languages
        return None


# The base tier is the Soniox transcription workflow.
SonioxTranscriptionWorkflow = SonioxWorkflow


class STTWorkflowFactory:
    """Drops in the transcription workflow for a tier or legacy config."""

    @staticmethod
    def get_workflow(
        settings: Any, model_tier: str | None = None
    ) -> OpenAIWorkflow | SonioxWorkflow:
        tier = normalize_model_tier(model_tier)
        if tier == MODEL_TIER_BASE:
            return SonioxWorkflow()
        if tier == MODEL_TIER_PRO:
            return OpenAITranscriptionWorkflow()
        if getattr(settings, "stt_provider", PROVIDER_OPENAI) == PROVIDER_SONIOX:
            return SonioxWorkflow()
        return OpenAIWorkflow()

    @staticmethod
    def provider_name(settings: Any, model_tier: str | None = None) -> str:
        return STTWorkflowFactory.get_workflow(settings, model_tier).name

    @staticmethod
    def create_upstream_client(
        settings: Any, model_tier: str | None = None
    ) -> Any:
        return STTWorkflowFactory.get_workflow(
            settings, model_tier
        ).create_upstream_client(settings)
