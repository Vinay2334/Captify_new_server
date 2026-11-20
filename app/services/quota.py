"""User quota tracking billed in whole subtitle seconds.

Billing rule (server-side only, never trusts client-supplied numbers):
every validated inbound audio byte counts ``1 / PCM_BYTES_PER_SECOND`` of
a second; the commit rounds up to whole seconds. Paused/muted stretches
send no audio, so they bill nothing; an idle-but-open session with no
audio bills nothing.

Safety properties:
- ``authorize`` runs before a session starts and refuses when no quota
  remains (``Remaining = limit - used``).
- ``commit`` is idempotent on ``client_session_id``: reconnects/retries
  that reuse the key can never double-bill, and concurrent sessions are
  serialized (row lock in Postgres, asyncio lock in memory).
- Sessions that produced no transcript (upstream failure before any
  output, pure silence, rejected audio) commit 0 seconds and are marked
  ``failed``/unprocessed instead of ``completed``.
- A running session re-reads remaining quota every few seconds; at zero
  it is cut off with ``quota_exceeded`` so it cannot consume past its
  allocation.
- Every ``QUOTA_COMMIT_SECONDS`` (default 15) a running session also
  checkpoints: it commits the seconds billed so far to the DB (session
  stays ``active``) and then re-reads quota, so another tab/session sees
  this usage within one interval instead of only at session end. Between
  checkpoints the session's own uncommitted audio already counts against
  its effective remainder, so a single session cannot overshoot either.
  Checkpoints carry the cumulative billed total, so retries apply only
  the delta since the last checkpoint and never double-bill; a session
  that ultimately bills 0 (no transcript / upstream failure) refunds any
  checkpointed seconds at ``finish``.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Extension audio is mono PCM16 @ 24 kHz: 24_000 samples/s * 2 bytes.
PCM_BYTES_PER_SECOND = 48_000

# Quota for users when the backend has no Supabase project configured
# (local dev / tests). Matches the SQL default plan (100 minutes).
MEMORY_DEFAULT_PLAN = "Free"
MEMORY_DEFAULT_LIMIT_SECONDS = 6000


def billed_seconds_for_bytes(num_bytes: int) -> int:
    """Whole seconds billed for validated audio bytes (rounds up)."""
    if num_bytes <= 0:
        return 0
    return int(math.ceil(num_bytes / PCM_BYTES_PER_SECOND))


@dataclass(frozen=True)
class QuotaSnapshot:
    user_id: str
    plan: str
    used_seconds: int
    limit_seconds: int

    @property
    def remaining_seconds(self) -> int:
        return max(self.limit_seconds - self.used_seconds, 0)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "plan": self.plan,
            "used_seconds": self.used_seconds,
            "limit_seconds": self.limit_seconds,
            "remaining_seconds": self.remaining_seconds,
        }


class QuotaDenied(Exception):
    """Raised when a session may not start (no quota left)."""

    def __init__(self, snapshot: QuotaSnapshot, reason: str = "quota_exceeded") -> None:
        super().__init__(reason)
        self.snapshot = snapshot
        self.reason = reason


class QuotaStore:
    """Persistence behind quota checks/commits (Supabase or memory)."""

    async def get_quota(self, user_id: str) -> QuotaSnapshot:
        raise NotImplementedError

    async def authorize(
        self, user_id: str, client_session_id: str, min_start_seconds: int = 1
    ) -> QuotaSnapshot:
        """Reserve a session slot. Raises QuotaDenied when not allowed."""
        raise NotImplementedError

    async def commit(
        self, client_session_id: str, billed_seconds: int, status: str
    ) -> QuotaSnapshot | None:
        """Idempotent finalize. Returns the fresh snapshot (None if unknown)."""
        raise NotImplementedError

    async def checkpoint(
        self, client_session_id: str, billed_seconds: int
    ) -> QuotaSnapshot | None:
        """Commit cumulative billed seconds mid-session (stays active).

        ``billed_seconds`` is the session total so far, not a delta: only
        the increase since the last checkpoint/finalize is applied, so a
        retried checkpoint never double-bills. Returns the fresh snapshot
        (None if the session is unknown or already finalized).
        """
        raise NotImplementedError


class MemoryQuotaStore(QuotaStore):
    """In-process store: local dev fallback and hermetic tests."""

    def __init__(self, default_limit_seconds: int = MEMORY_DEFAULT_LIMIT_SECONDS) -> None:
        self._lock = asyncio.Lock()
        self._profiles: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}
        self._default_limit = default_limit_seconds

    def set_plan(self, user_id: str, plan: str, limit_seconds: int) -> None:
        """Dashboard equivalent for tests: change a user's plan + limit."""
        profile = self._profiles.setdefault(
            user_id,
            {"plan": MEMORY_DEFAULT_PLAN, "limit": self._default_limit, "used": 0},
        )
        profile["plan"] = plan
        profile["limit"] = limit_seconds

    def _snapshot(self, user_id: str) -> QuotaSnapshot:
        profile = self._profiles.setdefault(
            user_id,
            {"plan": MEMORY_DEFAULT_PLAN, "limit": self._default_limit, "used": 0},
        )
        return QuotaSnapshot(
            user_id=user_id,
            plan=profile["plan"],
            used_seconds=profile["used"],
            limit_seconds=profile["limit"],
        )

    async def get_quota(self, user_id: str) -> QuotaSnapshot:
        async with self._lock:
            return self._snapshot(user_id)

    async def authorize(
        self, user_id: str, client_session_id: str, min_start_seconds: int = 1
    ) -> QuotaSnapshot:
        async with self._lock:
            existing = self._sessions.get(client_session_id)
            if existing is not None and existing["status"] == "active":
                # Reconnect reuses the key: resume without double-reserving.
                return self._snapshot(user_id)
            snapshot = self._snapshot(user_id)
            if snapshot.remaining_seconds < max(min_start_seconds, 1):
                raise QuotaDenied(snapshot)
            self._sessions[client_session_id] = {
                "user_id": user_id,
                "status": "active",
                "billed": 0,
            }
            return snapshot

    async def checkpoint(
        self, client_session_id: str, billed_seconds: int
    ) -> QuotaSnapshot | None:
        async with self._lock:
            session = self._sessions.get(client_session_id)
            if session is None or session["status"] in ("completed", "failed"):
                return None
            billed = max(int(billed_seconds or 0), 0)
            delta = max(billed - session["billed"], 0)
            if delta:
                session["billed"] = billed
                profile = self._profiles.setdefault(
                    session["user_id"],
                    {"plan": MEMORY_DEFAULT_PLAN, "limit": self._default_limit, "used": 0},
                )
                profile["used"] += delta
            return self._snapshot(session["user_id"])

    async def commit(
        self, client_session_id: str, billed_seconds: int, status: str
    ) -> QuotaSnapshot | None:
        async with self._lock:
            session = self._sessions.get(client_session_id)
            if session is None:
                return None
            if session["status"] in ("completed", "failed"):
                return self._snapshot(session["user_id"])
            # Delta against checkpointed seconds: a session billed at
            # checkpoints only pays the remainder here, and one that
            # ultimately bills 0 refunds what it checkpointed.
            final_billed = max(int(billed_seconds or 0), 0)
            delta = final_billed - session["billed"]
            session["status"] = status
            session["billed"] = final_billed
            profile = self._profiles.setdefault(
                session["user_id"],
                {"plan": MEMORY_DEFAULT_PLAN, "limit": self._default_limit, "used": 0},
            )
            profile["used"] = max(profile["used"] + delta, 0)
            return self._snapshot(session["user_id"])


class SupabaseQuotaStore(QuotaStore):
    """Postgres-backed store via Supabase REST (service role key).

    Prefers the atomic ``authorize_session`` / ``commit_session_usage``
    RPCs from the migration; falls back to plain REST reads/writes when
    the RPCs are absent so a partially applied migration still serves
    quota reads.
    """

    def __init__(self, settings: Settings) -> None:
        base = settings.supabase_url.rstrip("/")
        key = settings.supabase_service_role_key.strip()
        self._base = base
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self._min_start = max(settings.quota_min_start_seconds, 1)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base, headers=self._headers, timeout=10.0
        )

    async def _rpc(self, name: str, params: dict) -> dict | None:
        try:
            async with self._client() as client:
                response = await client.post(f"/rest/v1/rpc/{name}", json=params)
        except httpx.HTTPError as exc:
            log.warning("quota RPC %s transport failed type=%s", name, type(exc).__name__)
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            log.warning("quota RPC %s failed status=%s", name, response.status_code)
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def _rest_profile(self, user_id: str) -> QuotaSnapshot:
        async with self._client() as client:
            profile_resp = await client.get(
                "/rest/v1/profiles",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "quota_used_seconds,plan_id",
                },
            )
            if profile_resp.status_code >= 400:
                raise RuntimeError(f"profile read failed: {profile_resp.status_code}")
            rows = profile_resp.json()
            if not rows:
                return QuotaSnapshot(
                    user_id=user_id, plan="Free", used_seconds=0, limit_seconds=0
                )
            row = rows[0]
            plan_resp = await client.get(
                "/rest/v1/plans",
                params={
                    "id": f"eq.{row.get('plan_id', 'free')}",
                    "select": "name,quota_limit_seconds",
                },
            )
            name, limit = "Free", 0
            if plan_resp.status_code < 400:
                plans = plan_resp.json()
                if plans:
                    name = plans[0].get("name", "Free")
                    limit = int(plans[0].get("quota_limit_seconds", 0) or 0)
            return QuotaSnapshot(
                user_id=user_id,
                plan=name,
                used_seconds=int(row.get("quota_used_seconds", 0) or 0),
                limit_seconds=limit,
            )

    @staticmethod
    def _snapshot_from_rpc(user_id: str, data: dict) -> QuotaSnapshot:
        return QuotaSnapshot(
            user_id=user_id,
            plan=str(data.get("plan", "Free")),
            used_seconds=int(data.get("used_seconds", 0) or 0),
            limit_seconds=int(data.get("limit_seconds", 0) or 0),
        )

    async def get_quota(self, user_id: str) -> QuotaSnapshot:
        return await self._rest_profile(user_id)

    async def authorize(
        self, user_id: str, client_session_id: str, min_start_seconds: int = 1
    ) -> QuotaSnapshot:
        data = await self._rpc(
            "authorize_session",
            {"p_user_id": user_id, "p_client_session_id": client_session_id},
        )
        if data is not None:
            snapshot = self._snapshot_from_rpc(user_id, data)
            if not data.get("allowed", False):
                raise QuotaDenied(snapshot)
            return snapshot
        # Fallback without RPCs: read + plain insert (less atomic, but
        # keeps quota reads working on a partially applied migration).
        snapshot = await self._rest_profile(user_id)
        if snapshot.remaining_seconds < max(min_start_seconds, self._min_start):
            raise QuotaDenied(snapshot)
        try:
            async with self._client() as client:
                await client.post(
                    "/rest/v1/subtitle_sessions",
                    json={
                        "user_id": user_id,
                        "client_session_id": client_session_id,
                        "status": "active",
                    },
                )
        except httpx.HTTPError as exc:
            log.warning("session insert failed type=%s", type(exc).__name__)
        return snapshot

    async def checkpoint(
        self, client_session_id: str, billed_seconds: int
    ) -> QuotaSnapshot | None:
        data = await self._rpc(
            "checkpoint_session_usage",
            {
                "p_client_session_id": client_session_id,
                "p_billed_seconds": max(int(billed_seconds or 0), 0),
            },
        )
        # None covers a DB without the checkpoint migration yet: the
        # tracker keeps the seconds uncommitted and retries next interval
        # (its effective-remainder check still bounds this session).
        if data is None:
            return None
        used = data.get("used_seconds")
        if used is None:
            return None
        return QuotaSnapshot(
            user_id="",
            plan=str(data.get("plan", "Free")),
            used_seconds=int(used or 0),
            limit_seconds=int(data.get("limit_seconds", 0) or 0),
        )

    async def commit(
        self, client_session_id: str, billed_seconds: int, status: str
    ) -> QuotaSnapshot | None:
        data = await self._rpc(
            "commit_session_usage",
            {
                "p_client_session_id": client_session_id,
                "p_billed_seconds": max(int(billed_seconds or 0), 0),
                "p_status": status if status in ("completed", "failed") else "completed",
            },
        )
        if data is None:
            return None
        used = data.get("used_seconds")
        if used is None:
            return None
        return QuotaSnapshot(
            user_id="",
            plan=str(data.get("plan", "Free")),
            used_seconds=int(used or 0),
            limit_seconds=int(data.get("limit_seconds", 0) or 0),
        )


def build_quota_store(settings: Settings) -> QuotaStore:
    if settings.quota_enforced():
        return SupabaseQuotaStore(settings)
    return MemoryQuotaStore()


@dataclass
class SessionQuotaTracker:
    """Per-session billing accumulator owned by one relay session.

    Only audio bytes the relay actually kept (post-validation,
    post-rate-limit) are added. ``finish`` converts to whole seconds and
    commits exactly once; unknown sessions commit nothing.
    """

    store: QuotaStore
    user_id: str
    client_session_id: str
    snapshot: QuotaSnapshot
    audio_bytes: int = 0
    transcripts_emitted: int = 0
    refresh_interval: float = 5.0
    checkpoint_interval: float = 15.0
    _finished: bool = field(default=False, init=False)
    _last_refresh: float = field(default_factory=time.monotonic, init=False)
    _committed_seconds: int = field(default=0, init=False)
    _last_checkpoint: float = field(default_factory=time.monotonic, init=False)

    @property
    def billed_seconds(self) -> int:
        return billed_seconds_for_bytes(self.audio_bytes)

    @property
    def remaining_seconds(self) -> int:
        return self.snapshot.remaining_seconds

    @property
    def committed_seconds(self) -> int:
        """Seconds already checkpointed/committed to the DB."""
        return self._committed_seconds

    @property
    def uncommitted_seconds(self) -> int:
        """Billed seconds not yet in the DB snapshot."""
        return max(self.billed_seconds - self._committed_seconds, 0)

    @property
    def effective_remaining_seconds(self) -> int:
        """DB remainder minus this session's uncommitted audio.

        The DB snapshot alone never shrinks while this session runs, so
        the cutoff must use this value to fire mid-session.
        """
        return max(self.snapshot.remaining_seconds - self.uncommitted_seconds, 0)

    def add_audio_bytes(self, num_bytes: int) -> None:
        if num_bytes > 0:
            self.audio_bytes += num_bytes

    def note_transcript(self) -> None:
        self.transcripts_emitted += 1

    def quota_event(self) -> dict:
        event = self.snapshot.to_dict()
        event["used_seconds"] = self.snapshot.used_seconds + self.uncommitted_seconds
        event["remaining_seconds"] = self.effective_remaining_seconds
        return {"type": "quota.update", **event}

    async def checkpoint(self) -> QuotaSnapshot:
        """Commit the billed-so-far total to the DB now (stays active).

        Unconditional: the 15s background worker calls this every tick,
        even with no new whole second or no transcript yet, so usage
        reaches the DB on cadence rather than only when convenient. The
        store applies only the delta since the last checkpoint, so
        repeat commits never double-bill; if the session later bills 0
        at finish (no transcript / upstream failure), ``finish`` refunds
        whatever was committed here.
        """
        if self._finished:
            return self.snapshot
        self._last_checkpoint = time.monotonic()
        billed = self.billed_seconds
        try:
            fresh = await self.store.checkpoint(self.client_session_id, billed)
        except Exception as exc:
            log.warning("quota checkpoint failed type=%s", type(exc).__name__)
            return self.snapshot
        if fresh is None:
            # Unknown session or checkpoint RPC not deployed yet: keep
            # the seconds uncommitted and retry on the next tick.
            log.warning("quota checkpoint unavailable for session")
            return self.snapshot
        if not fresh.user_id:
            fresh = QuotaSnapshot(
                user_id=self.user_id,
                plan=fresh.plan,
                used_seconds=fresh.used_seconds,
                limit_seconds=fresh.limit_seconds,
            )
        self.snapshot = fresh
        self._committed_seconds = billed
        return self.snapshot

    async def maybe_checkpoint(
        self, *, force: bool = False, interval: float | None = None
    ) -> QuotaSnapshot:
        """Checkpoint if the interval has elapsed (or ``force``).

        Kept for interval-driven callers; the transports instead run a
        dedicated 15s worker that calls :meth:`checkpoint` directly.
        """
        if self._finished:
            return self.snapshot
        now = time.monotonic()
        due = self.checkpoint_interval if interval is None else interval
        if not force and now - self._last_checkpoint < max(due, 0.0):
            return self.snapshot
        return await self.checkpoint()

    async def maybe_refresh(self, *, force: bool = False) -> QuotaSnapshot:
        now = time.monotonic()
        if not force and now - self._last_refresh < self.refresh_interval:
            return self.snapshot
        self._last_refresh = now
        try:
            self.snapshot = await self.store.get_quota(self.user_id)
        except Exception as exc:
            log.warning("quota refresh failed type=%s", type(exc).__name__)
        return self.snapshot

    async def finish(self, reason: str) -> QuotaSnapshot | None:
        """Commit usage exactly once. Failed/unprocessed sessions bill 0."""
        if self._finished:
            return None
        self._finished = True
        failed = reason.startswith(
            ("upstream_error", "relay_error", "internal_error", "upstream_send_failed")
        )
        if failed or self.transcripts_emitted == 0:
            billed, status = 0, "failed"
        else:
            billed, status = self.billed_seconds, "completed"
        try:
            final = await self.store.commit(self.client_session_id, billed, status)
        except Exception as exc:
            log.warning("quota commit failed type=%s", type(exc).__name__)
            return None
        if final is not None and not final.user_id:
            final = QuotaSnapshot(
                user_id=self.user_id,
                plan=final.plan,
                used_seconds=final.used_seconds,
                limit_seconds=final.limit_seconds,
            )
        if final is not None:
            self._committed_seconds = billed
            self.snapshot = final
        log.info(
            "quota session billed user=%s billed=%ds transcripts=%d status=%s reason=%s",
            self.user_id,
            billed,
            self.transcripts_emitted,
            status,
            reason,
        )
        return final


async def begin_tracked_session(
    store: QuotaStore,
    user_id: str,
    client_session_id: str,
    *,
    min_start_seconds: int = 1,
) -> SessionQuotaTracker:
    """Authorize + build a tracker. Raises QuotaDenied when not allowed."""
    snapshot = await store.authorize(
        user_id, client_session_id, min_start_seconds=min_start_seconds
    )
    return SessionQuotaTracker(
        store=store, user_id=user_id, client_session_id=client_session_id, snapshot=snapshot
    )
