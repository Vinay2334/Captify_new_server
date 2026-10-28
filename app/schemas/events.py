"""Event schemas for the extension <-> backend <-> OpenAI relay.

Extension protocol (client <-> backend), JSON text frames:
  C->S  {"type": "audio.append", "audio": "<base64 PCM16 mono 24kHz>"}
  C->S  {"type": "audio.commit"}
  S->C  {"type": "session.ready", "model": "..."}
  S->C  {"type": "transcript.delta", "delta": "...", "item_id": "..."}
  S->C  {"type": "transcript.completed", "transcript": "...", "item_id": "..."}
  S->C  {"type": "error", "code": "...", "message": "..."}

The backend also accepts the OpenAI-native ``input_audio_buffer.append`` /
``input_audio_buffer.commit`` and ``session.input_audio_buffer.append`` /
``session.input_audio_buffer.commit`` events from the extension: appends
are forwarded as translation appends after validation, commits are
swallowed. An extension that already speaks the Realtime wire format
keeps working.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- client -> backend ---

class AudioAppendEvent(BaseModel):
    type: Literal["audio.append"]
    audio: str = Field(min_length=1)


class AudioCommitEvent(BaseModel):
    type: Literal["audio.commit"]


class UpstreamAudioAppendEvent(BaseModel):
    """OpenAI-native audio chunk, accepted from the extension as-is.

    Both the legacy transcription name and the translation
    ``session.``-prefixed name are accepted; the relay maps them to the
    translation or transcription append event for the session mode.
    """

    type: Literal["input_audio_buffer.append", "session.input_audio_buffer.append"]
    audio: str = Field(min_length=1)


class UpstreamAudioCommitEvent(BaseModel):
    """OpenAI-native commit, accepted but swallowed (no commit lifecycle)."""

    type: Literal["input_audio_buffer.commit", "session.input_audio_buffer.commit"]


class LanguageConfigEvent(BaseModel):
    """Per-session language selection sent by the extension.

    ``target_lang`` is forwarded upstream as a ``session.update`` output
    language. ``source_lang`` is accepted for forward compatibility but
    currently unused: the translation session always auto-detects the
    source language and the API exposes no source-language field.
    """

    type: Literal["language-config"]
    source_lang: str = Field(min_length=1)
    target_lang: str = Field(min_length=1)


CLIENT_EVENT_TYPES = {
    "audio.append",
    "audio.commit",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "session.input_audio_buffer.append",
    "session.input_audio_buffer.commit",
    "language-config",
}


def normalize_language_code(code: str) -> str:
    """Normalize a BCP-47 language tag for comparison."""
    return code.strip().lower()


def _primary_subtag(code: str) -> str:
    """Primary language subtag (``en-US`` -> ``en``)."""
    normalized = normalize_language_code(code)
    for sep in ("-", "_"):
        if sep in normalized:
            return normalized.split(sep)[0]
    return normalized


def is_same_language(source: str, target: str) -> bool:
    """Whether two language tags refer to the same spoken language."""
    src, tgt = _primary_subtag(source), _primary_subtag(target)
    return bool(src) and src == tgt


def parse_client_event(data: dict[str, Any]) -> BaseModel:
    """Validate one decoded client JSON object. Raises ValueError."""
    event_type = data.get("type")
    if event_type == "audio.append":
        return AudioAppendEvent.model_validate(data)
    if event_type == "audio.commit":
        return AudioCommitEvent.model_validate(data)
    if event_type in (
        "input_audio_buffer.append",
        "session.input_audio_buffer.append",
    ):
        return UpstreamAudioAppendEvent.model_validate(data)
    if event_type in (
        "input_audio_buffer.commit",
        "session.input_audio_buffer.commit",
    ):
        return UpstreamAudioCommitEvent.model_validate(data)
    if event_type == "language-config":
        return LanguageConfigEvent.model_validate(data)
    if event_type == "heartbeat":
        return Exception("heartbeat")
    raise ValueError(f"unsupported event type: {event_type!r}")


# --- backend -> client ---

class SessionReadyEvent(BaseModel):
    type: Literal["session.ready"] = "session.ready"
    model: str


class TranscriptDeltaEvent(BaseModel):
    type: Literal["transcript.delta"] = "transcript.delta"
    delta: str
    item_id: str = ""


class TranscriptCompletedEvent(BaseModel):
    type: Literal["transcript.completed"] = "transcript.completed"
    transcript: str
    item_id: str = ""


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


# --- upstream -> backend mapping ---

# Upstream events relayed (in mapped form) to the extension.
# Translation emits deltas only (no completed turns): the translated
# ``session.output_transcript.delta`` text becomes the extension's
# ``transcript.delta`` captions. Translation sessions have no ``item_id``,
# so the downstream field stays empty.
UPSTREAM_DELTA_TYPE = "session.output_transcript.delta"

# Upstream transcription events (gpt-realtime-transcribe). They carry the
# same spoken text the translation path would translate, so they map to
# the same downstream ``transcript.delta`` / ``transcript.completed``
# events the extension already renders.
TRANSCRIPTION_DELTA_TYPE = "conversation.item.input_audio_transcription.delta"
TRANSCRIPTION_COMPLETED_TYPE = (
    "conversation.item.input_audio_transcription.completed"
)

# Upstream events that are expected but carry nothing for the extension.
# They are consumed silently:
# - session lifecycle (created / updated / closed),
# - source-language ``session.input_transcript.delta`` (mapped only in
#   transcription mode; ignored when translating),
# - translated ``session.output_audio.delta`` (the downstream protocol is
#   text-only; no audio is forwarded).
UPSTREAM_IGNORED_TYPES = frozenset(
    {
        "session.created",
        "session.updated",
        "session.closed",
        "session.input_transcript.delta",
        "session.output_audio.delta",
    }
)


def map_upstream_event(
    event: dict[str, Any], transcription_mode: bool = False
) -> BaseModel | None:
    """Map one upstream event to an extension event.

    Transcription events map to the same downstream transcript events the
    extension renders for translations, so the client protocol is unchanged.
    ``session.input_transcript.delta`` additionally maps downstream only in
    transcription mode; when translating it stays ignored.

    Returns None for events the extension does not need. Raises
    UpstreamError for upstream ``error`` events.
    """
    event_type = event.get("type")
    if event_type == UPSTREAM_DELTA_TYPE:
        return TranscriptDeltaEvent(
            delta=event.get("delta", ""),
            item_id=event.get("item_id", ""),
        )
    if event_type == TRANSCRIPTION_DELTA_TYPE:
        return TranscriptDeltaEvent(
            delta=event.get("delta", ""),
            item_id=event.get("item_id", ""),
        )
    if event_type == TRANSCRIPTION_COMPLETED_TYPE:
        return TranscriptCompletedEvent(
            transcript=event.get("transcript", ""),
            item_id=event.get("item_id", ""),
        )
    if event_type == "session.input_transcript.delta" and transcription_mode:
        return TranscriptDeltaEvent(
            delta=event.get("delta", ""),
            item_id=event.get("item_id", ""),
        )
    if event_type in UPSTREAM_IGNORED_TYPES:
        return None
    if event_type == "error":
        raise UpstreamError.from_event(event)
    return None


class UpstreamError(Exception):
    """Safe, redacted upstream failure. Never carries raw payloads."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "UpstreamError":
        inner = event.get("error") if isinstance(event.get("error"), dict) else {}
        raw_code = str(inner.get("code", event.get("code", "upstream_error")))
        # Redact: only pass through short known-safe codes, else generic.
        code = raw_code if len(raw_code) <= 64 else "upstream_error"
        return cls(code=code, message="translation upstream error")

    def to_client_event(self) -> ErrorEvent:
        return ErrorEvent(code=self.code, message=self.message)
