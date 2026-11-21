"""FastAPI application entrypoint."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import build_upstream_client, router
from app.api.webrtc import router as webrtc_router
from app.api.quota import router as quota_router
from app.api.billing import router as billing_router
from app.api.plans import router as plans_router
from app.core.config import Settings, settings as default_settings
from app.core.logging import configure_logging, get_logger
from app.services.quota import build_quota_store

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not app.state.settings.upstream_api_key():
        log.warning(
            "translation API key is not set: translation sessions will be "
            "rejected until it is configured"
        )
    if not app.state.settings.auth_enabled():
        log.warning(
            "translation endpoint has no user authentication: anyone able to "
            "reach it may consume OpenAI API quota; keep it on localhost"
        )
    else:
        log.info(
            "user authentication enabled; quota enforcement=%s",
            app.state.settings.quota_enforced(),
        )
    yield
    peers = list(getattr(app.state, "webrtc_peers", set()))
    for pc in peers:
        try:
            await pc.close()
        except Exception:
            pass
    tasks = list(getattr(app.state, "background_tasks", set()))
    for task in tasks:
        if not task.done():
            task.cancel()
    log.info("shutdown complete")


def create_app(app_settings: Settings | None = None) -> FastAPI:
    resolved = app_settings or default_settings
    if resolved.is_production() and not resolved.auth_enabled():
        # Fail fast: a production deployment without Supabase Auth would
        # serve subtitle time anonymously, which cannot be billed to quota.
        raise RuntimeError(
            "Refusing to start: ENVIRONMENT=production requires SUPABASE_URL "
            "(and a JWT secret or JWKS) so every session is authenticated."
        )
    configure_logging(resolved.log_level)
    app = FastAPI(title="Captify Translation Backend", lifespan=lifespan)
    app.state.settings = resolved
    app.state.slot_lock = asyncio.Lock()
    app.state.active_sessions = 0
    app.state.webrtc_peers = set()
    app.state.background_tasks = set()
    app.state.quota_store = build_quota_store(resolved)
    app.state.upstream_factory = lambda model_tier=None: build_upstream_client(
        resolved, model_tier
    )
    app.include_router(router)
    app.include_router(webrtc_router)
    app.include_router(quota_router)
    app.include_router(billing_router)
    app.include_router(plans_router)

    # Browser access for the static site (pricing page fetches the plan
    # catalog and the signed-in user's quota). Bearer tokens ride the
    # Authorization header, so no cookies/credentials are involved.
    website_origins = resolved.website_origin_list()
    if website_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=website_origins,
            allow_methods=["GET", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            max_age=600,
        )

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "active_sessions": app.state.active_sessions,
            "model": resolved.active_model(),
        }

    return app


app = create_app()
