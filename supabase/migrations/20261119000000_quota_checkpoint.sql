-- Periodic quota checkpoints for running sessions.
--
-- Before this migration a session's usage only reached
-- profiles.quota_used_seconds when the session ended (commit_session_usage),
-- so a long session could run far past its quota: the periodic quota
-- re-read always saw the pre-session remainder, and concurrent sessions
-- (e.g. two tabs) each saw the full remainder until one of them ended.
--
-- checkpoint_session_usage commits the cumulative billed-so-far seconds
-- mid-session (the session stays 'active'); the backend calls it every
-- QUOTA_COMMIT_SECONDS (default 15) and then re-reads quota, so other
-- sessions see the usage within one interval. Only the delta since the
-- last checkpoint is applied, so retries never double-bill.
--
-- commit_session_usage is replaced with the delta-aware equivalent: its
-- p_billed_seconds is the session's final total, so it now applies
-- (final - already checkpointed) instead of the whole total. A final
-- total of 0 (failed / no-transcript session) therefore refunds any
-- checkpointed seconds. Sessions that never checkpoint behave exactly
-- as before (delta = total - 0).

-- ------------------------------------------- checkpoint_session_usage ---
create or replace function public.checkpoint_session_usage(
  p_client_session_id text,
  p_billed_seconds integer
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
  v_delta integer;
  v_used integer;
  v_limit integer;
  v_plan_name text;
  v_plan_id text;
begin
  if p_billed_seconds is null or p_billed_seconds < 0 then
    p_billed_seconds := 0;
  end if;

  select id, user_id, status, billed_seconds
    into v_session_id, v_user_id, v_old_status, v_old_billed
    from public.subtitle_sessions
   where client_session_id = p_client_session_id
   for update;

  if v_session_id is null then
    return jsonb_build_object(
      'committed', false,
      'reason', 'unknown_session',
      'billed_seconds', 0
    );
  end if;

  select quota_used_seconds, plan_id
    into v_used, v_plan_id
    from public.profiles
   where user_id = v_user_id;
  select quota_limit_seconds, name
    into v_limit, v_plan_name
    from public.plans
   where id = v_plan_id;

  if v_old_status in ('completed', 'failed') then
    return jsonb_build_object(
      'committed', false,
      'reason', 'already_finalized',
      'billed_seconds', v_old_billed,
      'used_seconds', coalesce(v_used, 0),
      'limit_seconds', coalesce(v_limit, 0),
      'remaining_seconds', greatest(coalesce(v_limit, 0) - coalesce(v_used, 0), 0),
      'plan', v_plan_name
    );
  end if;

  v_delta := greatest(p_billed_seconds - v_old_billed, 0);

  if v_delta > 0 then
    update public.subtitle_sessions
       set billed_seconds = p_billed_seconds
     where id = v_session_id;

    update public.profiles
       set quota_used_seconds = quota_used_seconds + v_delta,
           updated_at = now()
     where user_id = v_user_id
    returning quota_used_seconds into v_used;
  end if;

  return jsonb_build_object(
    'committed', v_delta > 0,
    'billed_seconds', greatest(v_old_billed, p_billed_seconds),
    'used_seconds', coalesce(v_used, 0),
    'limit_seconds', coalesce(v_limit, 0),
    'remaining_seconds', greatest(coalesce(v_limit, 0) - coalesce(v_used, 0), 0),
    'plan', v_plan_name
  );
end;
$$;

-- ------------------------------ commit_session_usage (delta-aware) ---
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
  v_delta integer;
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

  if v_session_id is null then
    return jsonb_build_object(
      'committed', false,
      'reason', 'unknown_session',
      'billed_seconds', 0
    );
  end if;

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

  -- Delta against checkpointed seconds (0 when never checkpointed):
  -- negative when the final total is 0 and checkpoints must be refunded.
  v_delta := p_billed_seconds - v_old_billed;

  update public.subtitle_sessions
     set status = p_status,
         billed_seconds = p_billed_seconds,
         ended_at = now()
   where id = v_session_id;

  update public.profiles
     set quota_used_seconds = greatest(quota_used_seconds + v_delta, 0),
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
