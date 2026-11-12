"""Hermetic test environment for Supabase connectivity.

The repository's local `.env` may contain a real SUPABASE_URL and service
role key (needed to run the backend for real). The test-suite must not
depend on that: unless a test explicitly opts in by passing Supabase
settings to `Settings(...)`, auth stays disabled and the in-process
`MemoryQuotaStore` is used, so no test ever touches the network.

Pydantic-settings priority is init kwargs > environment > `.env` file, so
these process-level defaults neutralize `.env` while explicit test kwargs
(like `test_auth_quota.py`'s URL + JWT secret) still take effect.
"""

import os

os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "")
os.environ.setdefault("SUPABASE_JWT_SECRET", "")
os.environ.setdefault("SUPABASE_JWKS_URL", "")
