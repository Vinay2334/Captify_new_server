"""Authenticated account endpoints: plan + quota for the extension UI."""

from fastapi import APIRouter, Depends, Request

from app.core.supabase_auth import AuthUser, require_user

router = APIRouter()


@router.get("/api/v1/me/quota")
async def my_quota(request: Request, user: AuthUser = Depends(require_user)) -> dict:
    """Current plan + usage for the signed-in user (owns this row only)."""
    store = request.app.state.quota_store
    snapshot = await store.get_quota(user.id)
    return {
        "user_id": user.id,
        "email": user.email,
        **snapshot.to_dict(),
    }
