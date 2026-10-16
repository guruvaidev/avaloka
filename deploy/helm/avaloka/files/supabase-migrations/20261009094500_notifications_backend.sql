-- Notifications (design doc D4). Matches the hosted project's table.
-- Idempotent: safe on a fresh database and on one that already has the table.
-- recipient_id / actor_id are profiles.id, NOT auth.uid(). Requires public.profiles.

create extension if not exists pgcrypto;

create table if not exists public.notifications (
    id             uuid        primary key default gen_random_uuid(),
    recipient_id   uuid        not null,
    actor_id       uuid,
    type           text        not null,
    title          text        not null,
    body           text,
    link           text,
    resource_type  text,
    resource_id    uuid,
    metadata       jsonb       default '{}'::jsonb,
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now(),
    read_at        timestamptz
);

alter table public.notifications add column if not exists actor_id      uuid;
alter table public.notifications add column if not exists body          text;
alter table public.notifications add column if not exists link          text;
alter table public.notifications add column if not exists resource_type text;
alter table public.notifications add column if not exists resource_id   uuid;
alter table public.notifications add column if not exists metadata      jsonb default '{}'::jsonb;
alter table public.notifications add column if not exists updated_at    timestamptz not null default now();
alter table public.notifications add column if not exists read_at       timestamptz;

do $$
begin
    if not exists (select 1 from pg_constraint
                   where conrelid = 'public.notifications'::regclass
                     and conname = 'notifications_recipient_fk') then
        alter table public.notifications
            add constraint notifications_recipient_fk
            foreign key (recipient_id) references public.profiles (id) on delete cascade;
    end if;
    if not exists (select 1 from pg_constraint
                   where conrelid = 'public.notifications'::regclass
                     and conname = 'notifications_actor_id_fkey') then
        alter table public.notifications
            add constraint notifications_actor_id_fkey
            foreign key (actor_id) references public.profiles (id) on delete set null;
    end if;
end
$$;

create index if not exists notifications_recipient_unread_idx
    on public.notifications (recipient_id, created_at desc) where read_at is null;
create index if not exists notifications_recipient_created_idx
    on public.notifications (recipient_id, created_at desc);
create index if not exists notifications_recipient_ref_idx
    on public.notifications (recipient_id, (metadata ->> 'ref'));

create or replace function public.notifications_set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at := now();
    return new;
end;
$$;

drop trigger if exists notifications_set_updated_at on public.notifications;
create trigger notifications_set_updated_at
    before update on public.notifications
    for each row execute function public.notifications_set_updated_at();

alter table public.notifications enable row level security;

drop policy if exists "own notifications select" on public.notifications;
create policy "own notifications select" on public.notifications for select to authenticated
    using (recipient_id = (select profiles.id from public.profiles where profiles.user_id = auth.uid()));

drop policy if exists "own notifications update" on public.notifications;
create policy "own notifications update" on public.notifications for update to authenticated
    using (recipient_id = (select profiles.id from public.profiles where profiles.user_id = auth.uid()))
    with check (recipient_id = (select profiles.id from public.profiles where profiles.user_id = auth.uid()));

drop policy if exists "own notifications delete" on public.notifications;
create policy "own notifications delete" on public.notifications for delete to authenticated
    using (recipient_id = (select profiles.id from public.profiles where profiles.user_id = auth.uid()));

drop policy if exists "authenticated notifications insert" on public.notifications;
create policy "authenticated notifications insert" on public.notifications for insert to authenticated
    with check (actor_id = (select profiles.id from public.profiles where profiles.user_id = auth.uid()));

grant select, insert, update, delete on public.notifications to authenticated;