"""FastAPI application entrypoint."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.websocket import build_upstream_client, router
from app.core.config import Settings, settings as default_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not app.state.settings.openai_api_key:
        log.warning(
            "OPENAI_API_KEY is not set: transcription sessions will be "
            "rejected until it is configured"
        )
    log.warning(
        "transcription endpoint has no user authentication: anyone able to "
        "reach it may consume OpenAI API quota; keep it on localhost"
    )
    yield
    log.info("shutdown complete")


def create_app(app_settings: Settings | None = None) -> FastAPI:
    resolved = app_settings or default_settings
    configure_logging(resolved.log_level)
    app = FastAPI(title="Captify Transcription Backend", lifespan=lifespan)
    app.state.settings = resolved
    app.state.slot_lock = asyncio.Lock()
    app.state.active_sessions = 0
    app.state.upstream_factory = lambda: build_upstream_client(resolved)
    app.include_router(router)

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "active_sessions": app.state.active_sessions,
            "model": resolved.openai_realtime_model,
        }

    return app


app = create_app()
