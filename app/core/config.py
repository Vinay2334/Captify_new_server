"""Application configuration loaded from environment variables.

All settings can be provided via environment variables or a ``.env`` file.
See ``.env.example`` for documented defaults.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.events import is_same_language


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI Realtime API (server side only - never sent to the extension).
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_realtime_model: str = Field(
        default="gpt-realtime-translate", alias="OPENAI_REALTIME_MODEL"
    )
    openai_realtime_url: str = Field(
        default="wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate",
        alias="OPENAI_REALTIME_URL",
    )
    # Target language for translated output audio and transcript deltas.
    # Source language is auto-detected. Supported codes include: es, pt,
    # fr, ja, ru, zh, de, ko, hi, id, vi, it, en.
    translation_target_language: str = Field(
        default="en", min_length=1, alias="OPENAI_TRANSLATION_TARGET_LANGUAGE"
    )
    # Transcription model used only for source-language transcript deltas.
    # Translation itself runs from the input audio stream regardless.
    translation_transcription_model: str = Field(
        default="gpt-realtime-whisper",
        alias="OPENAI_TRANSLATION_TRANSCRIPTION_MODEL",
    )
    # Deployment-wide mode switch: when True, every session transcribes
    # (gpt-realtime-transcribe) instead of translating. Default False
    # preserves the translation behavior.
    transcribe_mode: bool = Field(default=True, alias="TRANSCRIBE_MODE")
    # Source language for same-language detection. Empty means auto-detect
    # (translation path). When it matches the target language, sessions
    # transcribe instead of translating.
    translation_source_language: str = Field(
        default="", alias="OPENAI_TRANSLATION_SOURCE_LANGUAGE"
    )
    # Dedicated transcription model and endpoint for transcription
    # sessions (global transcribe mode or same-language sessions).
    openai_transcription_model: str = Field(
        default="gpt-live-transcribe", alias="OPENAI_TRANSCRIPTION_MODEL"
    )
    openai_transcription_url: str = Field(
        default="wss://api.openai.com/v1/realtime?intent=transcription",
        alias="OPENAI_TRANSCRIPTION_URL",
    )
    openai_beta_header: str = Field(default="realtime=v1", alias="OPENAI_BETA_HEADER")
    upstream_connect_timeout_seconds: float = Field(
        default=15.0, alias="UPSTREAM_CONNECT_TIMEOUT_SECONDS"
    )

    # Client-facing WebSocket limits.
    # Comma-separated origins, or "*" to allow any origin (local dev default).
    # Origin checks are NOT authentication.
    allowed_websocket_origins: str = Field(
        default="*", alias="ALLOWED_WEBSOCKET_ORIGINS"
    )
    max_websocket_message_bytes: int = Field(
        default=65536, alias="MAX_WEBSOCKET_MESSAGE_BYTES"
    )
    max_concurrent_sessions: int = Field(
        default=10, alias="MAX_CONCURRENT_SESSIONS"
    )
    session_idle_timeout_seconds: float = Field(
        default=120.0, alias="SESSION_IDLE_TIMEOUT_SECONDS"
    )
    # Mono PCM16 @ 24kHz is 48_000 bytes/s; default allows ~2x headroom.
    max_audio_bytes_per_second: int = Field(
        default=96000, alias="MAX_AUDIO_BYTES_PER_SECOND"
    )

    # WebRTC DataChannel audio (extensionRTC build). The label must match
    # DATA_CHANNEL_LABEL in extensionRTC/rtc-config.js. The extension
    # streams continuously; the upstream translation session is also
    # continuous (no commits), so audio is forwarded as it arrives.
    webrtc_data_channel_label: str = Field(
        default="audio", alias="WEBRTC_DATA_CHANNEL_LABEL"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator(
        "max_websocket_message_bytes",
        "max_concurrent_sessions",
        "max_audio_bytes_per_second",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("session_idle_timeout_seconds")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    def allowed_origins(self) -> list[str]:
        raw = self.allowed_websocket_origins.strip()
        if raw == "" or raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def is_origin_allowed(self, origin: str | None) -> bool:
        allowed = self.allowed_origins()
        if allowed == ["*"]:
            return True
        # Browser-less clients (tests, scripts) send no Origin header.
        if origin is None:
            return True
        return origin in allowed

    def starts_in_transcription(self) -> bool:
        """Whether new sessions open on the transcription path.

        True when the deployment-wide transcribe switch is on, or when
        the configured source language matches the target (same-language
        audio, which the translation model would omit).
        """
        return self.transcribe_mode or is_same_language(
            self.translation_source_language, self.translation_target_language
        )

    def transcription_language_hint(self) -> list[str]:
        """Spoken-language hint for the transcription session update."""
        return [
            self.translation_source_language or self.translation_target_language
        ]

    def active_model(self) -> str:
        """Model reported to clients (session.ready / health)."""
        if self.starts_in_transcription():
            return self.openai_transcription_model
        return self.openai_realtime_model


settings = Settings()
