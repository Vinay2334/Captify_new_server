"""Application configuration loaded from environment variables.

All settings can be provided via environment variables or a ``.env`` file.
See ``.env.example`` for documented defaults.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI Realtime API (server side only - never sent to the extension).
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_realtime_model: str = Field(
        default="gpt-live-transcribe", alias="OPENAI_REALTIME_MODEL"
    )
    openai_realtime_url: str = Field(
        default="wss://api.openai.com/v1/realtime?intent=transcription",
        alias="OPENAI_REALTIME_URL",
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


settings = Settings()
