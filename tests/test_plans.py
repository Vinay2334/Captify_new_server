"""Public plan catalog tests (no network, no Supabase).

Covers the website's data source for dynamic pricing:
- rows shaped for rendering (minutes, features, order, highlight flag)
- Dodo product attached from the reverse DODO_PLAN_MAP
- legacy rows (pre-migration) degrade to defaults
- 502 Supabase failure, 503 without service key
- CORS headers for the configured website origins
"""

import pytest
from fastapi.testclient import TestClient

import app.api.plans as plans_api
from app.core.config import Settings
from app.main import create_app


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SUBTITLE_PROVIDER": "none",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
        "SUPABASE_JWT_SECRET": "test-jwt-secret",
        "DODO_WEBHOOK_SECRET": "test-secret",
        "DODO_PLAN_MAP": "prod_pro=pro,prod_studio=studio",
    }
    base.update(overrides)
    return Settings(**base)


FULL_ROWS = [
    {
        "id": "studio",
        "name": "Studio",
        "quota_limit_seconds": 240000,
        "price_label": "$29 / month",
        "tagline": "For teams",
        "price_monthly_usd": 29,
        "features": [{"text": "3 seats", "dim": False}, "Everything in Pro"],
        "display_order": 2,
        "is_featured": False,
    },
    {
        "id": "free",
        "name": "Starter",
        "quota_limit_seconds": 6000,
        "price_label": "$0 / month",
        "tagline": "For trying",
        "price_monthly_usd": 0,
        "features": [],
        "display_order": 0,
        "is_featured": False,
    },
    {
        "id": "pro",
        "name": "Pro",
        "quota_limit_seconds": 60000,
        "price_label": "$12 / month",
        "tagline": "Popular",
        "price_monthly_usd": 12,
        "features": [{"text": "Priority speed", "dim": False}],
        "display_order": 1,
        "is_featured": True,
    },
]


def make_client(monkeypatch, rows):
    async def fake_fetch(settings):
        return rows

    monkeypatch.setattr(plans_api, "fetch_plan_rows", fake_fetch)
    return TestClient(create_app(make_settings()))


def test_lists_plans_ordered_with_dodo_products(monkeypatch):
    client = make_client(monkeypatch, FULL_ROWS)
    resp = client.get("/api/v1/plans")
    assert resp.status_code == 200
    plans = resp.json()["plans"]
    assert [p["id"] for p in plans] == ["free", "pro", "studio"]
    pro = plans[1]
    assert pro["minutes"] == 1000
    assert pro["is_featured"] is True
    assert pro["dodo_product_id"] == "prod_pro"
    assert pro["features"] == [{"text": "Priority speed", "dim": False}]
    # String features normalize to included bullets.
    assert plans[2]["features"] == [
        {"text": "3 seats", "dim": False},
        {"text": "Everything in Pro", "dim": False},
    ]
    # Free plan has no Dodo product: never a checkout button.
    assert plans[0]["dodo_product_id"] is None


def test_legacy_rows_degrade_to_defaults(monkeypatch):
    client = make_client(
        monkeypatch,
        [{"id": "free", "name": "Free", "quota_limit_seconds": 6000,
          "price_label": "$0", "tagline": "Try"}],
    )
    resp = client.get("/api/v1/plans")
    assert resp.status_code == 200
    (plan,) = resp.json()["plans"]
    assert plan["price_monthly_usd"] == 0
    assert plan["features"] == []
    assert plan["is_featured"] is False
    assert plan["minutes"] == 100


def test_supabase_failure_is_502(monkeypatch):
    async def fake_fetch(settings):
        return None

    monkeypatch.setattr(plans_api, "fetch_plan_rows", fake_fetch)
    client = TestClient(create_app(make_settings()))
    resp = client.get("/api/v1/plans")
    assert resp.status_code == 502


def test_disabled_without_service_key(monkeypatch):
    async def fake_fetch(settings):  # pragma: no cover - never reached
        raise AssertionError("must not read Supabase")

    monkeypatch.setattr(plans_api, "fetch_plan_rows", fake_fetch)
    settings = make_settings(SUPABASE_SERVICE_ROLE_KEY="")
    client = TestClient(create_app(settings))
    resp = client.get("/api/v1/plans")
    assert resp.status_code == 503


def test_cors_allows_configured_website_origin(monkeypatch):
    client = make_client(monkeypatch, [])
    resp = client.get(
        "/api/v1/plans", headers={"Origin": "http://localhost:8080"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == (
        "http://localhost:8080"
    )


def test_cors_blocks_unknown_origin(monkeypatch):
    client = make_client(monkeypatch, [])
    resp = client.get(
        "/api/v1/plans", headers={"Origin": "https://evil.example"}
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_website_origin_list_defaults_and_parsing():
    assert make_settings().website_origin_list() == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    assert make_settings(WEBSITE_ORIGINS="").website_origin_list() == []
    assert make_settings(
        WEBSITE_ORIGINS=" https://site.example ,https://www.site.example "
    ).website_origin_list() == ["https://site.example", "https://www.site.example"]
