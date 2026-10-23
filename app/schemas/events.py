"""Event schemas for the extension <-> backend <-> OpenAI relay.

Extension protocol (client <-> backend), JSON text frames:
  C->S  {"type": "audio.append", "audio": "<base64 PCM16 mono 24kHz>"}
  C->S  {"type": "audio.commit"}
  S->C  {"type": "session.ready", "model": "..."}
  S->C  {"type": "transcript.delta", "delta": "...", "item_id": "..."}
  S->C  {"type": "transcript.completed", "transcript": "...", "item_id": "..."}
  S->C  {"type": "error", "code": "...", "message": "..."}

The backend also accepts the OpenAI-native ``input_audio_buffer.append`` /
``input_audio_buffer.commit`` events from the extension and forwards them
after validation, so an extension that already speaks the Realtime wire
format keeps working.
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
    """OpenAI-native audio chunk, accepted from the extension as-is."""

    type: Literal["input_audio_buffer.append"]
    audio: str = Field(min_length=1)


class UpstreamAudioCommitEvent(BaseModel):
    type: Literal["input_audio_buffer.commit"]


CLIENT_EVENT_TYPES = {
    "audio.append",
    "audio.commit",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
}


def parse_client_event(data: dict[str, Any]) -> BaseModel:
    """Validate one decoded client JSON object. Raises ValueError."""
    event_type = data.get("type")
    if event_type == "audio.append":
        return AudioAppendEvent.model_validate(data)
    if event_type == "audio.commit":
        return AudioCommitEvent.model_validate(data)
    if event_type == "input_audio_buffer.append":
        return UpstreamAudioAppendEvent.model_validate(data)
    if event_type == "input_audio_buffer.commit":
        return UpstreamAudioCommitEvent.model_validate(data)
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
UPSTREAM_DELTA_TYPE = "conversation.item.input_audio_transcription.delta"
UPSTREAM_COMPLETED_TYPE = "conversation.item.input_audio_transcription.completed"

# Upstream events that are expected but carry nothing for the extension.
# They are consumed silently (session lifecycle, buffer acks).
UPSTREAM_IGNORED_TYPES = frozenset(
    {
        "session.created",
        "session.updated",
        "input_audio_buffer.committed",
        "input_audio_buffer.cleared",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.created",
        "rate_limits.updated",
    }
)


def map_upstream_event(event: dict[str, Any]) -> BaseModel | None:
    """Map one upstream event to an extension event.

    Returns None for events the extension does not need. Raises
    UpstreamError for upstream ``error`` events.
    """
    event_type = event.get("type")
    if event_type == UPSTREAM_DELTA_TYPE:
        return TranscriptDeltaEvent(
            delta=event.get("delta", ""),
            item_id=event.get("item_id", ""),
        )
    if event_type == UPSTREAM_COMPLETED_TYPE:
        return TranscriptCompletedEvent(
            transcript=event.get("transcript", ""),
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
        return cls(code=code, message="transcription upstream error")

    def to_client_event(self) -> ErrorEvent:
        return ErrorEvent(code=self.code, message=self.message)
