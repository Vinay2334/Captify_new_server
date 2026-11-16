-- Captify paid plans for Dodo Payments checkout.
--
-- SOURCE OF TRUTH for `supabase db push` (apply after
-- 20261112000000_auth_quota.sql). Additive only: inserts the `pro` and
-- `studio` rows the pricing site sells, matching website/index.html
-- (1,000 min = 60,000 s, 4,000 min = 240,000 s). Re-running is safe
-- (upsert on id also repairs limits/labels if edited by hand).

insert into public.plans (id, name, quota_limit_seconds, price_label, tagline)
values
  ('pro', 'Pro', 60000, '$12 / month', 'For daily viewers and learners'),
  ('studio', 'Studio', 240000, '$29 / month', 'For teams and heavy use')
on conflict (id) do update set
  name = excluded.name,
  quota_limit_seconds = excluded.quota_limit_seconds,
  price_label = excluded.price_label,
  tagline = excluded.tagline;
