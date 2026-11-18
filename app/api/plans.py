"""Public plan catalog for the static pricing site.

``GET /api/v1/plans`` returns every row of ``public.plans`` shaped for
rendering (price, minutes, features, order, highlight flag) plus the Dodo
product id each paid plan checks out with. No authentication: prices and
product ids are public (they already ship in the site's static config).

Plan content (name, price, features, order) is edited in the Supabase
dashboard — no website redeploy. The Dodo product linkage still comes
from ``DODO_PLAN_MAP`` (backend env): changing which Dodo product backs a
plan needs a backend restart, everything else is DB-only.

Reads go through the service_role key server-side, so RLS stays closed
(no anon access to the table). Works pre-migration too: when the catalog
columns (price_monthly_usd, features, display_order, is_featured) are
absent, the endpoint falls back to the legacy column set with defaults.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

CATALOG_COLUMNS = (
    "id,name,quota_limit_seconds,price_label,tagline,"
    "price_monthly_usd,features,display_order,is_featured"
)
LEGACY_COLUMNS = "id,name,quota_limit_seconds,price_label,tagline"


def _supabase_headers(settings: Settings) -> dict[str, str]:
    key = settings.supabase_service_role_key.strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def fetch_plan_rows(settings: Settings) -> list[dict] | None:
    """Raw ``plans`` rows, newest schema first with a legacy fallback.

    Returns None when Supabase cannot be reached at all (transient,
    worth a retry). A 400 on the new column set means the catalog
    migration has not been applied: retry with the legacy columns.
    """
    base = settings.supabase_url.rstrip("/")
    headers = _supabase_headers(settings)
    try:
        async with httpx.AsyncClient(
            base_url=base, headers=headers, timeout=10.0
        ) as client:
            response = await client.get(
                "/rest/v1/plans",
                params={"select": CATALOG_COLUMNS, "order": "display_order"},
            )
            if response.status_code == 400:
                response = await client.get(
                    "/rest/v1/plans",
                    params={"select": LEGACY_COLUMNS, "order": "id"},
                )
            if response.status_code >= 400:
                log.warning(
                    "plans catalog read failed status=%s",
                    response.status_code,
                )
                return None
            body = response.json()
            return body if isinstance(body, list) else None
    except httpx.HTTPError as exc:
        log.warning(
            "plans catalog read transport failed type=%s", type(exc).__name__
        )
        return None


def _clean_features(value: object) -> list[dict]:
    """Normalize the ``features`` jsonb column to [{text, dim}]."""
    if not isinstance(value, list):
        return []
    cleaned: list[dict] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            cleaned.append({"text": item.strip(), "dim": False})
        elif isinstance(item, dict) and str(item.get("text", "")).strip():
            cleaned.append(
                {
                    "text": str(item.get("text")).strip(),
                    "dim": bool(item.get("dim", False)),
                }
            )
    return cleaned


def normalize_plan(
    row: dict, dodo_product_by_plan: dict[str, str], order: int
) -> dict:
    """Shape one DB row for the pricing page (missing keys get defaults)."""
    limit = int(row.get("quota_limit_seconds", 0) or 0)
    plan_id = str(row.get("id", ""))
    try:
        price = float(row.get("price_monthly_usd", 0) or 0)
    except (TypeError, ValueError):
        price = 0.0
    display = row.get("display_order")
    return {
        "id": plan_id,
        "name": str(row.get("name", plan_id)),
        "price_label": str(row.get("price_label", "")),
        "price_monthly_usd": price,
        "tagline": str(row.get("tagline", "")),
        "minutes": limit // 60,
        "quota_limit_seconds": limit,
        "features": _clean_features(row.get("features")),
        "display_order": int(display) if display is not None else order,
        "is_featured": bool(row.get("is_featured", False)),
        "dodo_product_id": dodo_product_by_plan.get(plan_id),
    }


@router.get("/api/v1/plans")
async def list_plans(request: Request):
    settings: Settings = request.app.state.settings
    if not settings.quota_enforced():
        return JSONResponse(
            status_code=503, content={"ok": False, "reason": "billing_disabled"}
        )
    rows = await fetch_plan_rows(settings)
    if rows is None:
        return JSONResponse(
            status_code=502, content={"ok": False, "reason": "plans_unavailable"}
        )
    # Reverse of DODO_PLAN_MAP (product -> plan): the product each paid
    # plan checks out with. Later entries win on duplicate plan ids.
    dodo_product_by_plan = {
        plan_id: product_id
        for product_id, plan_id in settings.dodo_plans().items()
    }
    plans = [
        normalize_plan(row, dodo_product_by_plan, order)
        for order, row in enumerate(rows)
        if isinstance(row, dict) and row.get("id")
    ]
    plans.sort(key=lambda p: (p["display_order"], p["id"]))
    return {"plans": plans}
