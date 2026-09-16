-- =====================================================================
-- integration_connections
-- Per-user, UI-managed connections for GitHub / Outlook / Slack / JIRA.
-- Mirrors the mcp_connections convention: user_id stores auth.uid(), the
-- secret (a GitHub PAT for now) is stored AES-GCM encrypted in *_ciphertext
-- + *_iv columns via the backend's encrypt_secret(); plaintext never lands
-- in a column. Non-secret config (repo, channel, mailbox) lives in `config`.
-- =====================================================================

create table if not exists public.integration_connections (
    id               uuid primary key default gen_random_uuid(),
    user_id          uuid not null references auth.users(id) on delete cascade,
    provider         text not null check (provider in ('github','outlook','slack','jira')),
    -- The card toggle in Settings → Integrations.
    enabled          boolean not null default false,
    -- Non-secret provider config, e.g. {"repo": "owner/repo"} for github,
    -- {"channel": "#reports"} for slack, {"mailbox": "..."} for outlook.
    config           jsonb not null default '{}'::jsonb,
    -- Encrypted secret (GitHub PAT / Slack bot token / etc). Both columns are
    -- written together by encrypt_secret(); decrypt_secret() reads them back.
    token_ciphertext text,
    token_iv         text,
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),
    -- One connection per provider per user. Required for the upsert
    -- (on_conflict="user_id,provider") the backend uses.
    unique (user_id, provider)
);

-- updated_at maintenance (so /integrations can show "last changed" if needed).
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists integration_connections_set_updated_at on public.integration_connections;
create trigger integration_connections_set_updated_at
before update on public.integration_connections
for each row execute function public.set_updated_at();

-- RLS. The backend uses the service-role key (bypasses RLS) and enforces
-- ownership in code, exactly like mcp_connections. These policies are a
-- safety net in case anything ever queries the table with a user JWT.
alter table public.integration_connections enable row level security;

drop policy if exists "integration_connections_select_own" on public.integration_connections;
create policy "integration_connections_select_own"
  on public.integration_connections for select
  using (user_id = auth.uid());

drop policy if exists "integration_connections_insert_own" on public.integration_connections;
create policy "integration_connections_insert_own"
  on public.integration_connections for insert
  with check (user_id = auth.uid());

drop policy if exists "integration_connections_update_own" on public.integration_connections;
create policy "integration_connections_update_own"
  on public.integration_connections for update
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

drop policy if exists "integration_connections_delete_own" on public.integration_connections;
create policy "integration_connections_delete_own"
  on public.integration_connections for delete
  using (user_id = auth.uid());
