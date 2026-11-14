-- Captify authentication + quota management.
--
-- SOURCE OF TRUTH for `supabase db push` (timestamped name is what the
-- Supabase CLI expects). Apply once per project BEFORE starting the
-- backend with SUPABASE_URL set:
--   supabase db push   (project linked; see SUPABASE_SETUP.md)
-- or paste this file into the Supabase dashboard SQL editor.
--
-- Design notes:
--   * Supabase Auth owns identities (auth.users). This migration only adds
--     application tables that reference auth.users(id).
--   * Plan + quota limits live in `plans` so they can be changed from the
--     Supabase dashboard; the backend always reads them live (never
--     hardcoded). Changing a user's plan = updating profiles.plan_id.
--   * Usage is recorded in whole seconds: billed_seconds / quota_used_seconds.
--   * subtitle_sessions.client_session_id is the idempotency key generated
--     by the extension per logical session and reused across reconnects /
--     retries, so a retry never bills twice and failed sessions bill 0.
--   * Only the service_role key (backend) may change plan/quota/usage.
--     RLS gives users read access to their own rows only.
--   * Payment-gateway ready: add columns (e.g. stripe_customer_id) to
--     profiles/plans later without touching the quota flow.

-- ---------------------------------------------------------------- plans ---
create table if not exists public.plans (
  id text primary key,
  name text not null,
  quota_limit_seconds integer not null check (quota_limit_seconds >= 0),
  price_label text not null default '',
  tagline text not null default '',
  created_at timestamptz not null default now()
);

-- Default plan assigned to every newly registered user.
-- 6000 seconds = 100 minutes (matches the extension's quota display).
insert into public.plans (id, name, quota_limit_seconds, price_label, tagline)
values ('free', 'Free', 6000, '$0 / month', 'For trying Captify')
on conflict (id) do nothing;

-- -------------------------------------------------------------- profiles ---
create table if not exists public.profiles (
  user_id uuid primary key references auth.users (id) on delete cascade,
  first_name text,
  last_name text,
  plan_id text not null default 'free' references public.plans (id),
  quota_used_seconds integer not null default 0 check (quota_used_seconds >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- ----------------------------------------------------- subtitle_sessions ---
create table if not exists public.subtitle_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  -- Idempotency key: one logical subtitle session, stable across
  -- reconnects/retries. Unique so a retried commit can never double-bill.
  client_session_id text not null unique,
  status text not null default 'active'
    check (status in ('active', 'completed', 'failed')),
  billed_seconds integer not null default 0 check (billed_seconds >= 0),
  started_at timestamptz not null default now(),
  ended_at timestamptz
);

create index if not exists subtitle_sessions_user_id_idx
  on public.subtitle_sessions (user_id);

-- ------------------------------------------- auto-provision new signups ---
-- Every new auth user gets the default plan + zero usage, so the
-- "same default plan and quota for every user" rule holds without
-- backend write races on first login.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  default_plan text;
begin
  select id into default_plan from public.plans where id = 'free';
  if default_plan is null then
    select id into default_plan from public.plans order by quota_limit_seconds asc limit 1;
  end if;
  insert into public.profiles (user_id, first_name, last_name, plan_id, quota_used_seconds)
  values (new.id, new.raw_user_meta_data ->> 'first_name', new.raw_user_meta_data ->> 'last_name', coalesce(default_plan, 'free'), 0)
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ------------------------------------------------------------------ RLS ---
alter table public.plans enable row level security;
alter table public.profiles enable row level security;
alter table public.subtitle_sessions enable row level security;

-- Plans are readable by every authenticated user (needed to show the plan
-- name); only service_role can change them (no write policy for others).
drop policy if exists "plans_readable_by_authenticated" on public.plans;
create policy "plans_readable_by_authenticated"
  on public.plans for select
  to authenticated
  using (true);

-- Users can read only their own profile. No insert/update/delete policies
-- for authenticated users: plan and quota fields are backend-only.
drop policy if exists "profiles_read_own" on public.profiles;
create policy "profiles_read_own"
  on public.profiles for select
  to authenticated
  using (auth.uid() = user_id);

-- Users can read only their own session history. Writes are backend-only.
drop policy if exists "sessions_read_own" on public.subtitle_sessions;
create policy "sessions_read_own"
  on public.subtitle_sessions for select
  to authenticated
  using (auth.uid() = user_id);

-- --------------------------------- atomic quota RPCs (service_role use) ---
-- authorize_session: called BEFORE a subtitle session starts. Locks the
-- profile row, reuses the idempotency key on reconnect, and refuses when
-- no quota remains. One transaction => concurrent starts cannot overbook.
create or replace function public.authorize_session(
  p_user_id uuid,
  p_client_session_id text
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_plan_id text;
  v_used integer;
  v_limit integer;
  v_plan_name text;
  v_existing_id uuid;
  v_existing_status text;
  v_existing_billed integer;
begin
  -- Ensure a profile exists (covers users created before this migration).
  insert into public.profiles (user_id, plan_id, quota_used_seconds)
  values (p_user_id, 'free', 0)
  on conflict (user_id) do nothing;

  select plan_id, quota_used_seconds
    into v_plan_id, v_used
    from public.profiles
   where user_id = p_user_id
   for update;

  select quota_limit_seconds, name
    into v_limit, v_plan_name
    from public.plans
   where id = v_plan_id;
  if v_limit is null then
    v_limit := 0;
    v_plan_name := v_plan_id;
  end if;

  -- Reconnect with the same idempotency key: resume, never double-reserve.
  select id, status, billed_seconds
    into v_existing_id, v_existing_status, v_existing_billed
    from public.subtitle_sessions
   where client_session_id = p_client_session_id;

  if v_existing_id is not null then
    return jsonb_build_object(
      'allowed', (v_limit - v_used) > 0 or v_existing_status = 'active',
      'resumed', true,
      'session_id', v_existing_id,
      'status', v_existing_status,
      'billed_seconds', v_existing_billed,
      'used_seconds', v_used,
      'limit_seconds', v_limit,
      'remaining_seconds', greatest(v_limit - v_used, 0),
      'plan', v_plan_name
    );
  end if;

  if (v_limit - v_used) <= 0 then
    return jsonb_build_object(
      'allowed', false,
      'resumed', false,
      'used_seconds', v_used,
      'limit_seconds', v_limit,
      'remaining_seconds', 0,
      'plan', v_plan_name
    );
  end if;

  insert into public.subtitle_sessions (user_id, client_session_id, status)
  values (p_user_id, p_client_session_id, 'active')
  returning id into v_existing_id;

  return jsonb_build_object(
    'allowed', true,
    'resumed', false,
    'session_id', v_existing_id,
    'used_seconds', v_used,
    'limit_seconds', v_limit,
    'remaining_seconds', v_limit - v_used,
    'plan', v_plan_name
  );
end;
$$;

-- commit_session_usage: called ONCE when a session ends. Idempotent on
-- client_session_id: a retried commit returns the stored result without
-- adding usage again. Row lock serializes concurrent sessions.
create or replace function public.commit_session_usage(
  p_client_session_id text,
  p_billed_seconds integer,
  p_status text
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_session_id uuid;
  v_user_id uuid;
  v_old_status text;
  v_old_billed integer;
  v_used integer;
  v_limit integer;
  v_plan_name text;
  v_plan_id text;
begin
  if p_billed_seconds is null or p_billed_seconds < 0 then
    p_billed_seconds := 0;
  end if;
  if p_status not in ('completed', 'failed') then
    p_status := 'completed';
  end if;

  select id, user_id, status, billed_seconds
    into v_session_id, v_user_id, v_old_status, v_old_billed
    from public.subtitle_sessions
   where client_session_id = p_client_session_id
   for update;

  -- Unknown session (e.g. server restarted before authorize committed):
  -- nothing to bill; report zero without touching any profile.
  if v_session_id is null then
    return jsonb_build_object(
      'committed', false,
      'reason', 'unknown_session',
      'billed_seconds', 0
    );
  end if;

  -- Already finalized (retry / reconnect race): return stored outcome,
  -- never add usage a second time.
  if v_old_status in ('completed', 'failed') then
    select quota_used_seconds, plan_id
      into v_used, v_plan_id
      from public.profiles
     where user_id = v_user_id;
    select quota_limit_seconds into v_limit from public.plans where id = v_plan_id;
    return jsonb_build_object(
      'committed', false,
      'reason', 'already_finalized',
      'billed_seconds', v_old_billed,
      'used_seconds', coalesce(v_used, 0),
      'limit_seconds', coalesce(v_limit, 0),
      'remaining_seconds', greatest(coalesce(v_limit, 0) - coalesce(v_used, 0), 0)
    );
  end if;

  update public.subtitle_sessions
     set status = p_status,
         billed_seconds = p_billed_seconds,
         ended_at = now()
   where id = v_session_id;

  update public.profiles
     set quota_used_seconds = quota_used_seconds + p_billed_seconds,
         updated_at = now()
   where user_id = v_user_id
  returning quota_used_seconds, plan_id into v_used, v_plan_id;

  select quota_limit_seconds, name
    into v_limit, v_plan_name
    from public.plans
   where id = v_plan_id;

  return jsonb_build_object(
    'committed', true,
    'billed_seconds', p_billed_seconds,
    'used_seconds', v_used,
    'limit_seconds', coalesce(v_limit, 0),
    'remaining_seconds', greatest(coalesce(v_limit, 0) - v_used, 0),
    'plan', v_plan_name
  );
end;
$$;
