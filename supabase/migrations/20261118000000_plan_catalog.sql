-- Captify plan catalog for the pricing site.
--
-- SOURCE OF TRUTH for `supabase db push` (apply after
-- 20261116000000_billing_plans.sql). Additive only.
--
-- The website renders its plans from GET /api/v1/plans, which reads this
-- table live: price, features, name, tagline, order and highlight can all
-- be edited in the Supabase dashboard with no website redeploy. The seed
-- updates below only fill columns that still hold their defaults, so
-- re-running never clobbers dashboard customizations.
--
-- The `features` column is a JSON array of {"text": ..., "dim": true/false}
-- bullets (dim renders the grey "not included" row). The site prepends the
-- minutes line itself from quota_limit_seconds, so features stay free of
-- derived numbers.

alter table public.plans
  add column if not exists price_monthly_usd numeric not null default 0,
  add column if not exists features jsonb not null default '[]',
  add column if not exists display_order integer not null default 0,
  add column if not exists is_featured boolean not null default false;

-- Starter (matches website/pricing.html copy).
update public.plans
   set price_monthly_usd = 0,
       display_order = 0,
       is_featured = false,
       features = '[
         {"text": "Real-time captions & translation", "dim": false},
         {"text": "11 languages, auto-detected source", "dim": false},
         {"text": "1 device", "dim": false},
         {"text": "Priority processing speed", "dim": true},
         {"text": "Caption history", "dim": true}
       ]'::jsonb
 where id = 'free'
   and price_monthly_usd = 0
   and features = '[]'::jsonb;

-- Pro (most popular).
update public.plans
   set price_monthly_usd = 12,
       display_order = 1,
       is_featured = true,
       features = '[
         {"text": "Real-time captions & translation", "dim": false},
         {"text": "11 languages, auto-detected source", "dim": false},
         {"text": "Priority processing speed", "dim": false},
         {"text": "Caption history", "dim": false},
         {"text": "1 device · priority support", "dim": false}
       ]'::jsonb
 where id = 'pro'
   and price_monthly_usd = 0
   and features = '[]'::jsonb;

-- Studio.
update public.plans
   set price_monthly_usd = 29,
       display_order = 2,
       is_featured = false,
       features = '[
         {"text": "Everything in Pro", "dim": false},
         {"text": "3 seats included", "dim": false},
         {"text": "Shared team workspace", "dim": false},
         {"text": "Priority processing speed", "dim": false},
         {"text": "Dedicated support", "dim": false}
       ]'::jsonb
 where id = 'studio'
   and price_monthly_usd = 0
   and features = '[]'::jsonb;
