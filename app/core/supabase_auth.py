"""Supabase JWT verification for protected backend endpoints.

Supports both Supabase token formats:
- HS256, verified locally with ``SUPABASE_JWT_SECRET`` (legacy secrets), and
- RS256/ES256, verified against the project's JWKS document (current
  Supabase default) fetched once and cached.

The extension only ever holds the user's own access token (anon key is
public by design). The service role key never leaves the backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

_auth_scheme = HTTPBearer(auto_error=False)

# Cached JWKS client per URL: key rotation is picked up by PyJWT's keyset
# refresh on unknown kids only when we rebuild; cache TTL keeps it simple.
_jwks_clients: dict[str, tuple[float, jwt.PyJWKClient]] = {}
_JWKS_CACHE_SECONDS = 10 * 60


@dataclass(frozen=True)
class AuthUser:
    """Verified Supabase user extracted from an access token."""

    id: str
    email: str | None = None


class AuthError(Exception):
    """Expected rejection: expired/invalid token (maps to 401/4401)."""

    def __init__(self, message: str = "invalid or expired token") -> None:
        super().__init__(message)
        self.message = message


def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    now = time.monotonic()
    cached = _jwks_clients.get(jwks_url)
    if cached and now - cached[0] < _JWKS_CACHE_SECONDS:
        return cached[1]
    client = jwt.PyJWKClient(jwks_url, timeout=10)
    _jwks_clients[jwks_url] = (now, client)
    return client


def verify_supabase_token(token: str, settings: Settings) -> AuthUser:
    """Verify a Supabase access token. Raises AuthError on any failure."""
    if not token or not token.strip():
        raise AuthError("missing token")
    token = token.strip()

    # 1) HS256 fast path (legacy JWT secret configured).
    secret = settings.supabase_jwt_secret.strip()
    if secret:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
                leeway=30,
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("token expired")
        except jwt.PyJWTError:
            raise AuthError("invalid token")
        return _to_auth_user(payload)

    # 2) JWKS path (current Supabase default, RS256/ES256).
    try:
        signing_key = _jwks_client(settings.resolved_jwks_url()).get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False},
            leeway=30,
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("token expired")
    except jwt.PyJWTError as exc:
        log.warning("supabase token verification failed type=%s", type(exc).__name__)
        raise AuthError("invalid token")
    except (httpx.HTTPError, OSError) as exc:
        log.warning("supabase JWKS fetch failed type=%s", type(exc).__name__)
        raise AuthError("authentication unavailable")
    return _to_auth_user(payload)


def _to_auth_user(payload: dict) -> AuthUser:
    user_id = str(payload.get("sub") or "")
    if not user_id:
        raise AuthError("invalid token")
    email = payload.get("email")
    return AuthUser(id=user_id, email=str(email) if email else None)


def extract_bearer(request_or_ws) -> str | None:
    """Bearer token from Authorization header or ?token= query param.

    Query-param support exists because a WebSocket handshake (and some
    DataChannel signaling POSTs) cannot always set headers; the header is
    preferred whenever the client can send it.
    """
    auth = request_or_ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    params = getattr(request_or_ws, "query_params", None)
    if params is not None:
        token = params.get("token") or params.get("access_token")
        if token:
            return token
    return None


def settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


async def require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_auth_scheme),
) -> AuthUser:
    """FastAPI dependency: verified user or HTTP 401."""
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled():
        raise HTTPException(status_code=503, detail="authentication is not configured")
    token = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if not token:
        token = extract_bearer(request)
    try:
        return verify_supabase_token(token or "", settings)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=exc.message)


def verify_websocket_user(websocket: WebSocket, settings: Settings) -> AuthUser:
    """Verify a WebSocket handshake. Raises AuthError (caller closes 4401)."""
    if not settings.auth_enabled():
        raise AuthError("authentication is not configured")
    return verify_supabase_token(extract_bearer(websocket) or "", settings)


def clear_jwks_cache() -> None:
    """Test hook: drop cached JWKS clients."""
    _jwks_clients.clear()
