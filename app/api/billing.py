"""Dodo Payments webhook: paid checkout -> Supabase plan/quota.

``POST /api/v1/billing/dodo/webhook`` is called server-to-server by Dodo
(Developer -> Webhooks in the dashboard). The Standard Webhooks signature
IS the authentication — no Supabase user token is involved, so this route
deliberately skips ``require_user``. An empty ``DODO_WEBHOOK_SECRET``
disables it (503) rather than accepting unsigned money events.

Status contract (tuned for Dodo's retries):
- 401 invalid/missing signature — never applied.
- 200 applied or deliberately ignored (unknown type, unmapped product,
  no identifiable user): retrying would not help.
- 502 Supabase read/write failed — transient, worth a Dodo retry.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.services.billing import (
    BillingEvent,
    find_user_id_by_email,
    parse_dodo_event,
    set_user_plan,
    verify_dodo_signature,
)

log = get_logger(__name__)

router = APIRouter()


async def _resolve_user_id(settings, event: BillingEvent) -> str | None:
    if event.user_id:
        return event.user_id
    if event.email:
        return await find_user_id_by_email(event.email, settings)
    return None


@router.post("/api/v1/billing/dodo/webhook")
async def dodo_webhook(request: Request):
    settings = request.app.state.settings
    if not settings.billing_enabled():
        return JSONResponse(
            status_code=503, content={"ok": False, "reason": "billing_disabled"}
        )
    raw_body = await request.body()
    if not verify_dodo_signature(
        raw_body,
        request.headers.get("webhook-id"),
        request.headers.get("webhook-timestamp"),
        request.headers.get("webhook-signature"),
        settings.dodo_webhook_secret,
    ):
        log.warning("dodo webhook rejected: bad signature")
        return JSONResponse(
            status_code=401, content={"ok": False, "reason": "bad_signature"}
        )
    try:
        import json

        payload = json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400, content={"ok": False, "reason": "bad_json"}
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400, content={"ok": False, "reason": "bad_json"}
        )

    event = parse_dodo_event(payload, settings.dodo_plans())
    log.info(
        "dodo webhook received type=%s action=%s event=%s",
        payload.get("type"),
        event.action,
        event.event_id,
    )
    if event.action == "ignore":
        return JSONResponse(
            status_code=200, content={"ok": True, "action": "ignored"}
        )

    plan_id = (
        event.plan_id
        if event.action == "activate"
        else settings.dodo_free_plan_id.strip() or "free"
    )
    try:
        user_id = await _resolve_user_id(settings, event)
    except Exception as exc:  # lookup transport blew up: retryable
        log.warning("dodo user resolve failed type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=502, content={"ok": False, "reason": "lookup_failed"}
        )
    if not user_id:
        log.warning(
            "dodo webhook without attributable user type=%s event=%s",
            payload.get("type"),
            event.event_id,
        )
        return JSONResponse(
            status_code=200,
            content={"ok": True, "action": "ignored",
                     "reason": "unknown_user"},
        )
    try:
        applied = await set_user_plan(user_id, plan_id, settings)
    except Exception as exc:
        log.warning("dodo plan apply failed type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=502, content={"ok": False, "reason": "plan_update_failed"}
        )
    if not applied:
        return JSONResponse(
            status_code=502, content={"ok": False, "reason": "plan_update_failed"}
        )
    log.info(
        "dodo webhook applied action=%s user=%s plan=%s event=%s",
        event.action,
        user_id,
        plan_id,
        event.event_id,
    )
    return JSONResponse(
        status_code=200,
        content={"ok": True, "action": event.action, "plan_id": plan_id},
    )
