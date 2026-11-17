"""Dodo Payments webhook handling: verify, normalize, apply plan changes.

The pricing site links out to Dodo's hosted checkout (no card data ever
touches us). Dodo then tells this backend about money movement via
``POST /api/v1/billing/dodo/webhook``; this module holds everything that
endpoint needs except the FastAPI shell:

- ``verify_dodo_signature``: Standard Webhooks (Svix-compatible) HMAC
  check over the raw body, stdlib only — no new dependency.
- ``parse_dodo_event``: normalize Dodo's envelope into one BillingEvent
  (activate / deactivate / ignore) that only ever names a plan.
- ``find_user_id_by_email`` / ``set_user_plan``: service_role Supabase
  REST calls (the only writer of plan/quota, per the migration's RLS).

Idempotency is structural: applying the same event twice writes the same
``plan_id``, so Dodo retries can never double-apply. Unknown event types
and unmapped products resolve to ``ignore`` — retrying those would never
succeed, so the endpoint answers 200 for them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Dodo signs with the Standard Webhooks scheme and timestamps are
# time-checked with a 5 minute tolerance (per Dodo's docs).
WEBHOOK_TOLERANCE_SECONDS = 5 * 60

# Events that grant the mapped paid plan. `payment.succeeded` covers the
# first charge (and one-off products); subscription events cover renewals.
ACTIVATE_EVENTS = frozenset(
    {
        "payment.succeeded",
        "subscription.active",
        "subscription.renewed",
    }
)

# Events that drop the account back to the free plan.
DEACTIVATE_EVENTS = frozenset(
    {
        "subscription.cancelled",
        "subscription.expired",
    }
)

# Upper bound when resolving an email through the paginated Auth Admin API
# (50 users/page). Past this we stop and log instead of paging forever.
_MAX_ADMIN_PAGES = 20
_ADMIN_PAGE_SIZE = 50


@dataclass(frozen=True)
class BillingEvent:
    """Normalized webhook: what to do, for whom, on which plan."""

    action: str  # "activate" | "deactivate" | "ignore"
    product_id: str | None = None
    plan_id: str | None = None
    user_id: str | None = None
    email: str | None = None
    event_id: str | None = None


def _webhook_key(secret: str) -> bytes:
    """Decode a Standard Webhooks secret (optional `whsec_` prefix)."""
    raw = secret.strip()
    if raw.lower().startswith("whsec_"):
        raw = raw[len("whsec_"):]
    try:
        return base64.b64decode(raw)
    except (binascii.Error, ValueError):
        # Not base64: fall back to the raw bytes (test secrets, odd setups).
        return raw.encode("utf-8")


def verify_dodo_signature(
    raw_body: bytes,
    webhook_id: str | None,
    webhook_timestamp: str | None,
    webhook_signature: str | None,
    secret: str,
    *,
    now: float | None = None,
) -> bool:
    """Check the Standard Webhooks signature of a Dodo delivery.

    Signed content is ``{webhook-id}.{webhook-timestamp}.{raw_body}`` and
    the header holds one or more space-separated ``v1,<base64>`` values.
    Returns False (never raises) when anything is missing, malformed,
    stale, or mismatched.
    """
    if not secret or not secret.strip():
        return False
    if not webhook_id or not webhook_timestamp or not webhook_signature:
        return False
    try:
        issued_at = int(webhook_timestamp)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - issued_at) > WEBHOOK_TOLERANCE_SECONDS:
        return False
    try:
        payload = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    signed = f"{webhook_id}.{webhook_timestamp}.{payload}".encode("utf-8")
    expected = base64.b64encode(
        hmac.new(_webhook_key(secret), signed, hashlib.sha256).digest()
    ).decode("ascii")
    for candidate in webhook_signature.split():
        _version, _, signature = candidate.partition(",")
        if signature and hmac.compare_digest(signature, expected):
            return True
    return False


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dig(data: object, *paths: str) -> str | None:
    """Fetch the first present dotted path (``a.b.c``) from nested dicts."""
    if not isinstance(data, dict):
        return None
    for path in paths:
        node: object = data
        for part in path.split("."):
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if isinstance(node, str) and node.strip():
            return node.strip()
    return None


def parse_dodo_event(payload: dict, plan_map: dict[str, str]) -> BillingEvent:
    """Normalize a Dodo webhook envelope into an actionable event.

    ``plan_map`` is Dodo product id -> Supabase plans.id. The product id
    and customer identity are read defensively from several likely spots
    (top-level and nested subscription/payment objects) because Dodo's
    per-event shapes differ; anything unrecognized becomes ``ignore``.
    """
    event_type = payload.get("type") if isinstance(payload, dict) else None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(event_type, str) or not isinstance(data, dict):
        return BillingEvent(action="ignore")

    event_id = _first_str(payload.get("id"), data.get("webhook_id"))
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    event = BillingEvent(
        action="ignore",
        product_id= metadata.get("plan_id"),
        user_id=_first_str(metadata.get("user_id"), metadata.get("userId")),
        email=_dig(
            data,
            "customer.email",
            "customer_email",
            "email",
            "subscription.customer.email",
            "payment.customer.email",
        ),
        event_id=event_id,
    )

    if event_type in DEACTIVATE_EVENTS:
        return BillingEvent(
            action="deactivate",
            product_id=event.product_id,
            user_id=event.user_id,
            email=event.email,
            event_id=event.event_id,
        )
    if event_type in ACTIVATE_EVENTS:
        plan_id = plan_map.get(event.product_id or "")
        if not plan_id:
            log.warning(
                "dodo event for unmapped product type=%s product=%s",
                event_type,
                event.product_id,
            )
            return BillingEvent(
                action="ignore",
                product_id=event.product_id,
                user_id=event.user_id,
                email=event.email,
                event_id=event.event_id,
            )
        return BillingEvent(
            action="activate",
            product_id=event.product_id,
            plan_id=plan_id,
            user_id=event.user_id,
            email=event.email,
            event_id=event.event_id,
        )
    return event


def _supabase_headers(settings: Settings) -> dict[str, str]:
    key = settings.supabase_service_role_key.strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def find_user_id_by_email(
    email: str, settings: Settings, *, client: httpx.AsyncClient | None = None
) -> str | None:
    """Resolve a Supabase user id from an email via the Auth Admin API.

    ``client`` is a test seam: production passes None (a real client is
    built); tests inject a fake transport so no network is touched.
    """
    base = settings.supabase_url.rstrip("/")
    headers = _supabase_headers(settings)
    wanted = email.strip().lower()
    owned = client is not None
    http = client or httpx.AsyncClient(
        base_url=base, headers=headers, timeout=10.0
    )
    try:
        for page in range(1, _MAX_ADMIN_PAGES + 1):
            if owned:
                response = await http.get(
                    f"{base}/auth/v1/admin/users",
                    params={"page": page, "per_page": _ADMIN_PAGE_SIZE},
                    headers=headers,
                )
            else:
                response = await http.get(
                    "/auth/v1/admin/users",
                    params={"page": page, "per_page": _ADMIN_PAGE_SIZE},
                )
            if response.status_code >= 400:
                log.warning(
                    "dodo user lookup failed status=%s", response.status_code
                )
                return None
            try:
                body = response.json()
            except ValueError:
                return None
            users = body.get("users", []) if isinstance(body, dict) else []
            for user in users:
                if (
                    isinstance(user, dict)
                    and str(user.get("email", "")).strip().lower() == wanted
                    and user.get("id")
                ):
                    return str(user["id"])
            if len(users) < _ADMIN_PAGE_SIZE:
                return None
        log.warning("dodo user lookup hit page cap email=%s", email)
        return None
    except httpx.HTTPError as exc:
        log.warning("dodo user lookup transport failed type=%s", type(exc).__name__)
        return None
    finally:
        if not owned:
            await http.aclose()


async def set_user_plan(
    user_id: str, plan_id: str, settings: Settings, *, client=None
) -> bool:
    """Move a profile to ``plan_id`` (service_role bypasses RLS by design).

    Returns True on success. A missing ``plans`` row fails closed (False)
    via the foreign key instead of inventing a plan.
    """
    from datetime import datetime, timezone

    base = settings.supabase_url.rstrip("/")
    headers = _supabase_headers(settings)
    owned = client is not None
    http = client or httpx.AsyncClient(
        base_url=base, headers=headers, timeout=10.0
    )
    try:
        if owned:
            response = await http.patch(
                f"{base}/rest/v1/profiles?user_id=eq.{user_id}",
                json={
                    "plan_id": plan_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                headers=headers,
            )
        else:
            response = await http.patch(
                f"/rest/v1/profiles?user_id=eq.{user_id}",
                json={
                    "plan_id": plan_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        if response.status_code >= 400:
            log.warning(
                "dodo plan update failed user=%s plan=%s status=%s",
                user_id,
                plan_id,
                response.status_code,
            )
            return False
        return True
    except httpx.HTTPError as exc:
        log.warning("dodo plan update transport failed type=%s", type(exc).__name__)
        return False
    finally:
        if not owned:
            await http.aclose()
