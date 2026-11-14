# Supabase Auth + Quota Setup

This guide wires user authentication (Google OAuth + email) and subtitle
quota management into Captify. Estimated: 15–20 minutes in the Supabase
dashboard, no code changes required.

## 1. Create the Supabase project

1. Create a project at [supabase.com](https://supabase.com) (free tier is enough).
2. Open **Project Settings → API** and note:
   - `Project URL` → `SUPABASE_URL`
   - `anon public` key → extension `SUPABASE_ANON_KEY` (public by design)
   - `service_role` key → backend `SUPABASE_SERVICE_ROLE_KEY` (**secret**)
3. Open **Project Settings → API → JWT**: if your project shows a **JWT
   Secret** (HS256, legacy projects), copy it to backend
   `SUPABASE_JWT_SECRET`. Current projects use RS256/ES256 with a JWKS
   document — leave `SUPABASE_JWT_SECRET` empty and the backend verifies
   tokens against `<SUPABASE_URL>/auth/v1/.well-known/jwks.json`.

## 2. Run the SQL migration

Open **SQL Editor** and run
[`supabase/migrations/0001_auth_quota.sql`](supabase/migrations/0001_auth_quota.sql).
It creates:

| Object | Purpose |
|---|---|
| `plans` | Plan catalog (`free` = 6000 s / 100 min default). Edit limits here; the backend reads them live. |
| `profiles` | One row per user (`user_id` → `auth.users.id`), plan + used seconds. Auto-created on signup via the `handle_new_user` trigger. |
| `subtitle_sessions` | One row per logical session, keyed by idempotency key `client_session_id`. Exactly-once billing. |
| RLS policies | Users read only their own rows. **No** write policy for plan/quota/usage — only `service_role` (the backend) changes them. |
| `authorize_session` / `commit_session_usage` | Atomic RPCs: pre-start quota check + idempotent usage commit (row locks serialize concurrent sessions). |

Verify: **Table Editor** should show `plans` with the `free` row.

## 3. Configure the backend

Add to `Captify_new_server/.env` (see `.env.example`):

```env
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...service_role...
# SUPABASE_JWT_SECRET=...   # only for legacy HS256 projects
ENVIRONMENT=production
```

`ENVIRONMENT=production` makes the backend refuse to boot without
Supabase Auth, so anonymous (unbilled) subtitle time is impossible in
any real deployment. Keep `development` only for local work.

Restart: `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`.
With `SUPABASE_URL` set, both transports require a verified token and bill
usage; without it they keep the previous open local-dev behavior.

## 4. Configure the extension

1. Edit `Captify_rtc/src/caption-overlay/lib/supabase-config.js`:
   `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `BACKEND_BASE_URL`
   (must match where uvicorn listens).
2. Rebuild: `npm run build` (output in `dist/`), then reload the
   unpacked extension at `chrome://extensions`.
3. The extension never holds the service role key — only the user's own
   access token (persisted in `chrome.storage.local`, refreshed before
   expiry, restored on browser restart).

## 5. Google OAuth

1. **Google Cloud Console**: create OAuth 2.0 credentials (Web application),
   add `https://YOUR-PROJECT.supabase.co/auth/v1/callback` as an authorized
   redirect URI.
2. **Supabase → Authentication → Providers → Google**: enable, paste the
   Google client ID + secret.
3. **Supabase → Authentication → Providers**: keep only **Email** and
   **Google** enabled. Leave **Anonymous** OFF — and even if it is ever
   flipped on by accident, the backend rejects `is_anonymous` tokens
   (`anonymous_disabled`) on every transport, so anonymous users can
   never start sessions or consume quota.
4. **Supabase → Authentication → URL Configuration**: add as Redirect URLs:
   - `https://YOUR-EXTENSION-ID.chromiumapp.org/` (from
     `chrome.identity.getRedirectURL()`; the ID is stable per installations
     from the same codebase, or use your Chrome Web Store ID)
   - `http://localhost:8000/**` (local testing only)
5. The sign-in popup's **Continue with Google** opens the
   consent flow and returns to the extension with a Supabase session.
   If Google is not configured, email sign-up/sign-in works on its own
   (enable it under **Authentication → Providers → Email**; with
   *Confirm email* on, new users click the emailed link first).

## 6. Testing plan/quota changes from the dashboard

- **Change a user's plan**: Table Editor → `profiles` → set `plan_id`
  (e.g. add a `pro` row in `plans` first). Takes effect on the next
  session start and on the next menu open (the menu re-reads quota when
  opened, and the backend pushes live `quota.update` events while running).
- **Top up / reset usage**: edit `profiles.quota_used_seconds`.
- **Audit billing**: `subtitle_sessions` shows per-session billed seconds
  and status (`completed` vs `failed` — failed/unprocessed sessions bill 0).
- Never edit `quota` from the extension: RLS blocks it; the menu is
  display-only.

## 9. Notes for local testing

- Run the migration exactly once per project **before** starting the
  backend with `SUPABASE_URL` set. Until the tables exist, session start
  fails (the backend cannot authorize or bill). Order: SQL/`db push`
  first, `uvicorn` second.
- `pytest` is hermetic for Supabase connectivity (`tests/conftest.py`
  pins empty Supabase vars unless a test passes its own): your real
  `.env` keys never make tests hit the network. The remaining red tests
  in the suite are pre-existing config-default drift, unrelated to auth.
- `supabase migration list --remote` shows `20261112000000` under Local
  with an empty Remote column until you push.

## 7. Security model

- Access tokens are verified on **every** protected call (offer POST,
  WebSocket handshake, `/me/quota`); expired tokens get 401/`auth_expired`,
  exhausted quota gets 402/`quota_exceeded`.
- Billing counts only audio bytes the relay kept (validated, under the
  rate limit), converted server-side (`48_000` B = 1 s, rounded up).
- `client_session_id` makes reconnects/retries idempotent; concurrent
  sessions serialize on the profile row; a running session is cut off when
  remaining quota hits zero.
- Failed sessions (upstream error before any transcript, silent sessions,
  rejected audio) commit 0 seconds.

## 8. Adding payments later

No quota-flow changes needed: add a `stripe_customer_id` column to
`profiles`, new rows to `plans`, and a webhook endpoint that updates
`profiles.plan_id` — `authorize`/`commit` and RLS already key off those
tables.
