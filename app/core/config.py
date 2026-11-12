"""Application configuration loaded from environment variables.

All settings can be provided via environment variables or a ``.env`` file.
See ``.env.example`` for documented defaults.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.events import is_same_language

# Valid values for the global STT provider switch (``STT_PROVIDER``).
STT_PROVIDERS = ("openai", "soniox")

# Valid values for the subtitle (text2text translation) provider switch
# (``SUBTITLE_PROVIDER``). ``"none"`` disables post-transcription
# translation; ``"deepseek"`` translates live-transcribe transcripts with
# a DeepSeek chat model (default ``deepseek-flash``) over the existing
# ``httpx`` dependency (OpenAI-compatible ``/chat/completions``).
SUBTITLE_PROVIDERS = ("none", "deepseek")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Global STT provider switch. "openai" preserves the existing OpenAI
    # Realtime workflow; "soniox" routes sessions through Soniox stt-rt-v5.
    stt_provider: str = Field(default="soniox", alias="STT_PROVIDER")

    # Soniox Realtime API (server side only - never sent to the extension).
    soniox_api_key: str = Field(default="", alias="SONIOX_API_KEY")
    soniox_model: str = Field(default="stt-rt-v5", alias="SONIOX_MODEL")
    soniox_url: str = Field(
        default="wss://stt-rt.soniox.com/transcribe-websocket",
        alias="SONIOX_URL",
    )
    # Extension audio is mono PCM16 @ 24 kHz; Soniox accepts this rate
    # natively, so no resampling is needed.
    soniox_sample_rate: int = Field(default=24000, alias="SONIOX_SAMPLE_RATE")
    soniox_num_channels: int = Field(default=1, alias="SONIOX_NUM_CHANNELS")
    soniox_enable_endpoint_detection: bool = Field(
        default=True, alias="SONIOX_ENABLE_ENDPOINT_DETECTION"
    )

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
        default="en", alias="OPENAI_TRANSLATION_SOURCE_LANGUAGE"
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

    # Subtitle (text2text translation) provider switch. "none" (default)
    # keeps transcription transcripts untranslated; "deepseek" translates
    # each live-transcribe transcript segment into the session target
    # language with a rolling window of previous segments as context.
    subtitle_provider: str = Field(default="none", alias="SUBTITLE_PROVIDER")
    # Model id sent to the DeepSeek chat API (OpenAI-compatible).
    subtitle_model: str = Field(default="deepseek-flash", alias="SUBTITLE_MODEL")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    # Rolling-window size: how many previous source/translation segment
    # pairs are sent as context alongside the segment being translated.
    subtitle_context_segments: int = Field(
        default=3, alias="SUBTITLE_CONTEXT_SEGMENTS"
    )
    subtitle_timeout_seconds: float = Field(
        default=15.0, alias="SUBTITLE_TIMEOUT_SECONDS"
    )

    subtitle_min_segment_chars: int = Field(
        default=20, alias="SUBTITLE_MIN_SEGMENT_CHARS"
    )
    subtitle_max_segment_chars: int = Field(
        default=300, alias="SUBTITLE_MAX_SEGMENT_CHARS"
    )
    subtitle_flush_seconds: float = Field(
        default=2.0, alias="SUBTITLE_FLUSH_SECONDS"
    )

    # --- Supabase authentication + quota management ---
    # Empty SUPABASE_URL keeps the previous local-dev behavior: endpoints
    # stay open (see lifespan warning in app.main). Setting SUPABASE_URL
    # (plus a JWT secret or JWKS) turns authentication and quota
    # enforcement on for both transports.
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    # Service role key: backend only, never sent to the extension. Used
    # for quota reads/writes (bypasses RLS by design).
    supabase_service_role_key: str = Field(
        default="", alias="SUPABASE_SERVICE_ROLE_KEY"
    )
    # HS256 JWT secret (Supabase dashboard -> project settings -> API).
    # When set, tokens are verified locally with it. When empty but
    # SUPABASE_URL is set, tokens are verified against the project's
    # JWKS document (RS256/ES256, current Supabase default).
    supabase_jwt_secret: str = Field(default="", alias="SUPABASE_JWT_SECRET")
    supabase_jwks_url: str = Field(default="", alias="SUPABASE_JWKS_URL")
    # Seconds of remaining quota required to start a session. Default 1:
    # any positive remainder may start; the session is cut off at zero.
    quota_min_start_seconds: int = Field(
        default=1, alias="QUOTA_MIN_START_SECONDS"
    )
    # How often a running session re-reads remaining quota (seconds).
    quota_refresh_seconds: float = Field(
        default=5.0, alias="QUOTA_REFRESH_SECONDS"
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

    @field_validator("stt_provider")
    @classmethod
    def _provider_supported(cls, value: str) -> str:
        if value not in STT_PROVIDERS:
            raise ValueError(f"must be one of {STT_PROVIDERS}")
        return value

    @field_validator("subtitle_provider")
    @classmethod
    def _subtitle_provider_supported(cls, value: str) -> str:
        if value not in SUBTITLE_PROVIDERS:
            raise ValueError(f"must be one of {SUBTITLE_PROVIDERS}")
        return value

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

    @field_validator("subtitle_context_segments")
    @classmethod
    def _non_negative_segments(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be zero or positive")
        return value

    @field_validator("session_idle_timeout_seconds")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("subtitle_min_segment_chars")
    @classmethod
    def _non_negative_subtitle_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be zero or positive")
        return value

    @field_validator("subtitle_max_segment_chars")
    @classmethod
    def _non_negative_subtitle_max(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be zero or positive")
        return value

    @field_validator("subtitle_flush_seconds")
    @classmethod
    def _positive_subtitle_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    def subtitle_enabled(self) -> bool:
        return self.subtitle_provider != "none"

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

    def is_soniox_provider(self) -> bool:
        """Whether sessions route through Soniox instead of OpenAI."""
        return self.stt_provider == "soniox"

    def auth_enabled(self) -> bool:
        """Whether Supabase auth/quota enforcement is on."""
        return bool(self.supabase_url.strip())

    def quota_enforced(self) -> bool:
        """Whether subtitle time is billed against a user quota."""
        return self.auth_enabled() and bool(
            self.supabase_service_role_key.strip()
        )

    def resolved_jwks_url(self) -> str:
        if self.supabase_jwks_url.strip():
            return self.supabase_jwks_url.strip()
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    def upstream_api_key(self) -> str:
        """API key for the active provider (never sent to the extension)."""
        if self.is_soniox_provider():
            return self.soniox_api_key
        return self.openai_api_key

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

    def soniox_language_hints(self) -> list[str]:
        """Spoken-language hints for the Soniox session config."""
        hints: list[str] = []
        for code in (
            self.translation_source_language,
            self.translation_target_language,
        ):
            if code and code not in hints:
                hints.append(code)
        return hints or [self.translation_target_language]

    def active_model(self) -> str:
        """Model reported to clients (session.ready / health)."""
        if self.is_soniox_provider():
            return self.soniox_model
        if self.starts_in_transcription():
            return self.openai_transcription_model
        return self.openai_realtime_model


settings = Settings()
